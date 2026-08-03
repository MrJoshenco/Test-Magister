#!/usr/bin/env python3
"""
eval_saliency.py

Evalúa el rendimiento de algoritmos de segmentación / descubrimiento de objetos
no supervisado (Saliency / Object Discovery) comparando una carpeta de
predicciones contra una carpeta de Ground Truth (GT), sobre un dataset de
100 imágenes dividido en 4 categorías secuenciales de 25 imágenes cada una.

Métricas calculadas por imagen:
    1. MAE        - Mean Absolute Error entre el mapa continuo y el GT binario.
    2. Max F-beta - Máximo F-measure variando el umbral de binarización (0-255),
                    calculado con beta^2 = 0.3 (clásico saliency) y beta^2 = 1.0 (F1).
    3. mIoU       - IoU tras binarizar la predicción con umbral relativo (mean + k_std*std),
                    igual que el método usado en DINOv2.py para generar la máscara binaria.
    4. CorLoc     - 1 si el IoU (paso 3) >= 0.5, 0 en caso contrario.

    El umbral relativo por defecto usa k_std = 0.5, configurable con --k_std.

El script imprime:
    - Una tabla de resultados promedio POR CADA lote de 25 imágenes
      ("Categoría 1 (Fotos 1-25)", "Categoría 2 (Fotos 26-50)", ...).
    - Para cada métrica se muestra: media, desviación estándar y el intervalo
      de confianza (IC) al 95 % (configurable con --confidence).
    - Una tabla resumen final con el promedio total sobre las 100 imágenes.

Además, exporta todos los resultados a un archivo de texto (.txt) cuya ruta
se controla con --output (por defecto: eval_results.txt).

Uso:
    python eval_saliency.py --gt_dir /ruta/a/GT --pred_dir /ruta/a/predicciones

Requisitos:
    numpy, opencv-python
    (opcional) scipy - para cálculo exacto del intervalo de confianza con la
                       distribución t de Student. Si no está disponible, se usa
                       una tabla interpolada para IC 95 % o aproximación normal
                       para otros niveles de confianza.
"""

import os
import sys
import argparse

import numpy as np
import cv2

# scipy es opcional; si no está disponible se usa una tabla de t para IC 95 %
try:
    from scipy import stats
    HAS_SCIPY = True
except Exception:  # pragma: no cover - entorno sin scipy
    HAS_SCIPY = False
    stats = None

EPS = 1e-8
BATCH_SIZE = 25  # tamaño fijo de cada categoría
DEFAULT_OUTPUT = "eval_results.txt"
DEFAULT_CONFIDENCE = 0.95

# --------------------------------------------------------------------------- #
# Utilidades de carga y normalización
# --------------------------------------------------------------------------- #

def load_gray(path):
    """Carga una imagen en escala de grises. Devuelve None si falla."""
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def match_resolution(pred, gt):
    """
    Si pred y gt tienen distinta resolución, redimensiona pred al tamaño de gt
    usando interpolación bilineal.
    """
    if pred.shape != gt.shape:
        pred = cv2.resize(
            pred,
            (gt.shape[1], gt.shape[0]),  # cv2.resize espera (ancho, alto)
            interpolation=cv2.INTER_LINEAR,
        )
    return pred


def normalize01(arr):
    """Normaliza un arreglo a floats en [0, 1]."""
    arr = arr.astype(np.float64)
    mn, mx = arr.min(), arr.max()
    if mx - mn < EPS:
        return np.zeros_like(arr) if mx < 1 else np.ones_like(arr)
    return (arr - mn) / (mx - mn)


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #

def compute_mae(pred_norm, gt_norm):
    """Mean Absolute Error entre el mapa continuo (pred) y el GT binario."""
    return float(np.mean(np.abs(pred_norm - gt_norm)))


def compute_max_fbeta(pred_norm, gt_binary, beta2):
    """
    Calcula el Max F-beta variando el umbral de 0 a 255 mediante un enfoque
    de histograma acumulado (equivalente a iterar umbral por umbral, pero
    vectorizado y rápido).

    pred_norm  : arreglo float [0, 1]
    gt_binary  : arreglo booleano (True = objeto / foreground)
    beta2      : valor de beta^2 (0.3 o 1.0)
    """
    pred_255 = np.clip(np.round(pred_norm * 255), 0, 255).astype(np.uint8)

    gt_num = gt_binary.sum()
    if gt_num == 0:
        return 0.0

    fg_vals = pred_255[gt_binary]
    bg_vals = pred_255[~gt_binary]

    fg_hist, _ = np.histogram(fg_vals, bins=256, range=(0, 256))
    bg_hist, _ = np.histogram(bg_vals, bins=256, range=(0, 256))

    # Para cada umbral t (0..255): cuántos píxeles tienen valor >= t.
    tp = np.cumsum(fg_hist[::-1])[::-1].astype(np.float64)
    fp = np.cumsum(bg_hist[::-1])[::-1].astype(np.float64)
    fn = gt_num - tp

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)

    numerator = (1 + beta2) * precision * recall
    denominator = beta2 * precision + recall + EPS
    fbeta = numerator / denominator

    return float(np.max(fbeta))


def compute_miou_and_corloc(pred_norm, gt_binary, k_std=0.5):
    """
    Binariza la predicción con el método relativo usado en el pipeline:
    umbral = mean(pred) + k_std * std(pred).
    Calcula el IoU y CorLoc contra el GT.
    """
    # Umbral relativo: igual que make_binary_mask(method="relative") en DINOv2.py
    thresh_val = pred_norm.mean() + k_std * pred_norm.std()
    pred_bin = pred_norm > thresh_val

    intersection = np.logical_and(pred_bin, gt_binary).sum()
    union = np.logical_or(pred_bin, gt_binary).sum()

    iou = intersection / union if union > 0 else 0.0
    corloc = 1.0 if iou >= 0.5 else 0.0

    return float(iou), float(corloc)


# --------------------------------------------------------------------------- #
# Evaluación por imagen
# --------------------------------------------------------------------------- #

def evaluate_pair(pred_path, gt_path, k_std=0.5):
    """Calcula las 4 métricas para un par (predicción, GT). Devuelve dict o None."""
    gt = load_gray(gt_path)
    pred = load_gray(pred_path)

    if gt is None:
        print(f"  [AVISO] No se pudo leer el GT: {gt_path} -> se omite.")
        return None
    if pred is None:
        print(f"  [AVISO] No se pudo leer la predicción: {pred_path} -> se omite.")
        return None

    pred = match_resolution(pred, gt)

    pred_norm = normalize01(pred)
    gt_norm = normalize01(gt)
    gt_binary = gt_norm > 0.5

    mae = compute_mae(pred_norm, gt_norm)
    max_fbeta_03 = compute_max_fbeta(pred_norm, gt_binary, beta2=0.3)
    max_fbeta_10 = compute_max_fbeta(pred_norm, gt_binary, beta2=1.0)
    iou, corloc = compute_miou_and_corloc(pred_norm, gt_binary, k_std=k_std)

    return {
        "mae": mae,
        "max_fbeta_0.3": max_fbeta_03,
        "max_fbeta_1.0": max_fbeta_10,
        "iou": iou,
        "corloc": corloc,
    }


# --------------------------------------------------------------------------- #
# Emparejamiento, chunking y evaluación del dataset completo
# --------------------------------------------------------------------------- #

def list_sorted_pngs(directory):
    """
    Lista los .png de un directorio ordenados alfabéticamente de forma estricta.
    Ignora archivos auxiliares de visualización (_heatmap.png y _overlay.png)
    para evaluar únicamente las máscaras binarias.
    """
    excluded_suffixes = ("_heatmap.png", "_overlay.png")
    files = [
        f for f in os.listdir(directory)
        if f.lower().endswith(".png")
        and not f.lower().endswith(excluded_suffixes)
    ]
    return sorted(files)  # orden alfabético estricto (str sort)


def chunk_list(lst, size):
    """Divide una lista en sublistas consecutivas de tamaño `size`."""
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def evaluate_dataset(gt_dir, pred_dir, batch_size=BATCH_SIZE, k_std=0.5):
    gt_files = list_sorted_pngs(gt_dir)
    pred_files = list_sorted_pngs(pred_dir)

    if not gt_files:
        print(f"No se encontraron imágenes PNG en el directorio GT: {gt_dir}")
        sys.exit(1)

    if len(gt_files) != len(pred_files):
        print(
            f"[AVISO] La cantidad de archivos GT ({len(gt_files)}) y de predicciones "
            f"({len(pred_files)}) no coincide. Se emparejará solo por nombre existente en ambos."
        )

    if len(gt_files) % batch_size != 0:
        print(
            f"[AVISO] El total de {len(gt_files)} imágenes GT no es múltiplo exacto de "
            f"{batch_size}. El último lote quedará incompleto."
        )

    pred_set = set(pred_files)
    batches = chunk_list(gt_files, batch_size)

    all_results = []          # resultados de las 100 imágenes (para el resumen total)
    batch_results_list = []   # lista de listas, una por categoría

    global_idx = 0
    for batch_num, batch_files in enumerate(batches, start=1):
        start_photo = global_idx + 1
        end_photo = global_idx + len(batch_files)
        print(f"\n--- Procesando Categoría {batch_num} (Fotos {start_photo}-{end_photo}) ---")

        batch_results = []
        for fname in batch_files:
            global_idx += 1
            print(f"[{global_idx}/{len(gt_files)}] Procesando: {fname}")

            gt_path = os.path.join(gt_dir, fname)

            if fname not in pred_set:
                print(f"  [AVISO] No existe la predicción correspondiente: {fname} -> se omite.")
                continue

            pred_path = os.path.join(pred_dir, fname)
            metrics = evaluate_pair(pred_path, gt_path, k_std=k_std)
            if metrics is not None:
                metrics["filename"] = fname
                batch_results.append(metrics)
                all_results.append(metrics)

        batch_results_list.append((batch_num, start_photo, end_photo, batch_results))

    return batch_results_list, all_results


# --------------------------------------------------------------------------- #
# Estadísticas: media, desviación estándar e intervalo de confianza
# --------------------------------------------------------------------------- #

# Tabla de valores t para IC 95 % (bilateral). Se usa solo si scipy no está
# instalado. Los puntos se interpolan linealmente para df intermedios.
_T_TABLE_DF = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
     35, 40, 45, 50, 60, 70, 80, 90, 100, 120, np.inf],
    dtype=np.float64,
)
_T_TABLE_T = np.array(
    [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
     2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
     2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
     2.030, 2.021, 2.014, 2.009, 2.000, 1.994, 1.990, 1.987, 1.984, 1.980,
     1.960],
    dtype=np.float64,
)


def t_critical(df, confidence=DEFAULT_CONFIDENCE):
    """
    Devuelve el valor crítico t para un intervalo de confianza bilateral.

    - Si scipy está disponible, usa stats.t.ppf exacto.
    - Si no, para IC 95 % usa una tabla interpolada de valores t.
    - Para otros niveles de confianza sin scipy, cae en aproximación normal z=1.96.
    """
    alpha = 1 - confidence
    if df < 1:
        return 0.0

    if HAS_SCIPY:
        return float(stats.t.ppf(1 - alpha / 2, df))

    # Sin scipy: si es exactamente 95 %, usar tabla interpolada
    if abs(confidence - 0.95) < 1e-6:
        if df >= _T_TABLE_DF[-2]:
            return float(_T_TABLE_T[-1])
        return float(np.interp(df, _T_TABLE_DF, _T_TABLE_T))

    # Fallback: aproximación normal (z=1.96)
    print(
        f"[AVISO] scipy no está instalado y --confidence={confidence} != 0.95. "
        "Usando aproximación normal z=1.96 para el intervalo de confianza."
    )
    return 1.96


def compute_stats(results, confidence=DEFAULT_CONFIDENCE):
    """
    Calcula media, desviación estándar (muestral) e intervalo de confianza
    para cada métrica. Devuelve un dict con la forma:

        {
          "mae": {"mean": ..., "std": ..., "sem": ..., "ci_low": ..., "ci_high": ...},
          ...
          "n": N
        }
    """
    if not results:
        return None

    metric_keys = ["mae", "max_fbeta_0.3", "max_fbeta_1.0", "iou", "corloc"]
    n = len(results)
    t_val = t_critical(n - 1, confidence) if n > 1 else 0.0

    stats_dict = {"n": n}
    for key in metric_keys:
        vals = np.array([r[key] for r in results], dtype=np.float64)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 0 else 0.0
        margin = t_val * sem
        stats_dict[key] = {
            "mean": mean,
            "std": std,
            "sem": sem,
            "ci_low": mean - margin,
            "ci_high": mean + margin,
        }
    return stats_dict


# --------------------------------------------------------------------------- #
# Formato de resultados (tablas de texto)
# --------------------------------------------------------------------------- #

def format_metrics_table(title, stats, k_std=0.5, confidence=DEFAULT_CONFIDENCE):
    """Genera la representación en texto de una tabla de métricas."""
    lines = []
    lines.append("")
    lines.append("=" * 76)
    lines.append(f" {title}")
    lines.append("=" * 76)

    if stats is None:
        lines.append(" No hay imágenes evaluadas correctamente en este bloque.")
        lines.append("=" * 76)
        return "\n".join(lines)

    pct = int(round(confidence * 100))
    lines.append(f"{'Métrica':<34} {'Media':>10} {'± Std':>10} {f'IC {pct}%':>18}")
    lines.append("-" * 76)

    def fmt(metric, label, display_k=None):
        s = stats[metric]
        ci_label = f"[{s['ci_low']:.4f}, {s['ci_high']:.4f}]"
        row_label = label
        if display_k is not None:
            row_label = f"{label} (k_std={display_k})"
        lines.append(
            f"{row_label:<34} {s['mean']:>10.4f} {s['std']:>10.4f} {ci_label:>18}"
        )

    fmt("mae", "MAE")
    fmt("max_fbeta_0.3", "Max F-beta (beta^2=0.3)")
    fmt("max_fbeta_1.0", "Max F-beta (beta^2=1.0 / F1)")
    fmt("iou", "mIoU", display_k=k_std)
    fmt("corloc", "CorLoc")

    lines.append("-" * 76)
    lines.append(f"{'Imágenes evaluadas':<34}{stats['n']:>42d}")
    lines.append("=" * 76)
    return "\n".join(lines)


def print_all_results(batch_results_list, all_results, k_std=0.5,
                      confidence=DEFAULT_CONFIDENCE):
    """Imprime y devuelve el texto completo de los resultados."""
    output_parts = []

    # Tablas por categoría
    for batch_num, start_photo, end_photo, batch_results in batch_results_list:
        stats = compute_stats(batch_results, confidence=confidence)
        title = f"Resultados Categoría {batch_num} (Fotos {start_photo}-{end_photo})"
        table = format_metrics_table(title, stats, k_std=k_std, confidence=confidence)
        output_parts.append(table)
        print(table)

    # Tabla resumen total
    total_stats = compute_stats(all_results, confidence=confidence)
    total_title = f"RESUMEN TOTAL DEL DATASET ({len(all_results)} imágenes)"
    total_table = format_metrics_table(
        total_title, total_stats, k_std=k_std, confidence=confidence
    )
    output_parts.append(total_table)
    print(total_table)

    return "\n".join(output_parts)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evalúa predicciones de saliency/object discovery contra Ground Truth, "
            "en lotes de 25 imágenes (4 categorías) sobre un dataset de 100 imágenes. "
            "Imprime media, desviación estándar e intervalo de confianza, y exporta "
            "todo a un archivo de texto."
        )
    )
    parser.add_argument("--gt_dir", required=True, help="Ruta a la carpeta con las máscaras GT.")
    parser.add_argument("--pred_dir", required=True, help="Ruta a la carpeta con las predicciones.")
    parser.add_argument(
        "--k_std",
        type=float,
        default=0.5,
        help="Factor k_std para el umbral relativo mean + k_std*std usado en mIoU/CorLoc (default: 0.5).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Tamaño de cada lote/categoría (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=(
            "Nivel de confianza para el intervalo de confianza de las métricas "
            f"(default: {DEFAULT_CONFIDENCE})."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=(
            "Ruta del archivo de texto donde se exportarán los resultados "
            f"(default: {DEFAULT_OUTPUT})."
        ),
    )
    args = parser.parse_args()

    if not os.path.isdir(args.gt_dir):
        print(f"Error: el directorio GT no existe: {args.gt_dir}")
        sys.exit(1)
    if not os.path.isdir(args.pred_dir):
        print(f"Error: el directorio de predicciones no existe: {args.pred_dir}")
        sys.exit(1)

    if not (0 < args.confidence < 1):
        print("Error: --confidence debe estar entre 0 y 1.")
        sys.exit(1)

    if not HAS_SCIPY:
        print(
            "[AVISO] scipy no está instalado. Se usará la tabla de t para IC 95 % "
            "o aproximación normal para otros niveles de confianza."
        )

    print(f"GT dir:     {args.gt_dir}")
    print(f"Pred dir:   {args.pred_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"k_std:      {args.k_std}  (umbral relativo: mean + k_std*std)")
    print(f"Confidence: {args.confidence}")
    print(f"Output:     {args.output}")

    batch_results_list, all_results = evaluate_dataset(
        args.gt_dir, args.pred_dir, batch_size=args.batch_size, k_std=args.k_std
    )
    output_text = print_all_results(
        batch_results_list, all_results, k_std=args.k_std, confidence=args.confidence
    )

    # Exportar a archivo de texto
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"\n[INFO] Resultados exportados a: {args.output}")
    except OSError as exc:
        print(f"\n[ERROR] No se pudo escribir el archivo de salida {args.output}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
