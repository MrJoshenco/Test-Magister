#!/usr/bin/env python3
"""
eval_saliency_batches.py

Evalúa el rendimiento de algoritmos de segmentación / descubrimiento de objetos
no supervisado (Saliency / Object Discovery) comparando una carpeta de
predicciones contra una carpeta de Ground Truth (GT), sobre un dataset de
100 imágenes dividido en 4 categorías secuenciales de 25 imágenes cada una.

Métricas calculadas por imagen:
    1. MAE        - Mean Absolute Error entre el mapa continuo y el GT binario.
    2. Max F-beta - Máximo F-measure variando el umbral de binarización (0-255),
                    calculado con beta^2 = 0.3 (clásico saliency) y beta^2 = 1.0 (F1).
    3. mIoU       - IoU tras binarizar la predicción con el método de Otsu.
    4. CorLoc     - 1 si el IoU (paso 3) >= 0.5, 0 en caso contrario.

El script imprime:
    - Una tabla de resultados promedio POR CADA lote de 25 imágenes
      ("Categoría 1 (Fotos 1-25)", "Categoría 2 (Fotos 26-50)", ...).
    - Una tabla resumen final con el promedio total sobre las 100 imágenes.

Uso:
    python eval_saliency_batches.py --gt_dir /ruta/a/GT --pred_dir /ruta/a/predicciones

Requisitos:
    numpy, opencv-python
"""

import os
import sys
import argparse

import numpy as np
import cv2

EPS = 1e-8
BATCH_SIZE = 25  # tamaño fijo de cada categoría


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


def compute_miou_and_corloc(pred_norm, gt_binary):
    """
    Binariza la predicción con Otsu y calcula el IoU contra el GT.
    Devuelve (iou, corloc).
    """
    pred_255 = np.clip(np.round(pred_norm * 255), 0, 255).astype(np.uint8)

    try:
        _, pred_bin = cv2.threshold(
            pred_255, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        pred_bin = pred_bin.astype(bool)
    except cv2.error:
        pred_bin = np.zeros_like(pred_255, dtype=bool)

    intersection = np.logical_and(pred_bin, gt_binary).sum()
    union = np.logical_or(pred_bin, gt_binary).sum()

    iou = intersection / union if union > 0 else 0.0
    corloc = 1.0 if iou >= 0.5 else 0.0

    return float(iou), float(corloc)


# --------------------------------------------------------------------------- #
# Evaluación por imagen
# --------------------------------------------------------------------------- #

def evaluate_pair(pred_path, gt_path):
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
    iou, corloc = compute_miou_and_corloc(pred_norm, gt_binary)

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
    """Lista los .png de un directorio ordenados alfabéticamente de forma estricta."""
    files = [f for f in os.listdir(directory) if f.lower().endswith(".png")]
    return sorted(files)  # orden alfabético estricto (str sort)


def chunk_list(lst, size):
    """Divide una lista en sublistas consecutivas de tamaño `size`."""
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def evaluate_dataset(gt_dir, pred_dir, batch_size=BATCH_SIZE):
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
            metrics = evaluate_pair(pred_path, gt_path)
            if metrics is not None:
                metrics["filename"] = fname
                batch_results.append(metrics)
                all_results.append(metrics)

        batch_results_list.append((batch_num, start_photo, end_photo, batch_results))

    return batch_results_list, all_results


# --------------------------------------------------------------------------- #
# Impresión de resultados
# --------------------------------------------------------------------------- #

def compute_averages(results):
    """Calcula los promedios de las 4 métricas para una lista de resultados por imagen."""
    if not results:
        return None
    return {
        "mae": float(np.mean([r["mae"] for r in results])),
        "max_fbeta_0.3": float(np.mean([r["max_fbeta_0.3"] for r in results])),
        "max_fbeta_1.0": float(np.mean([r["max_fbeta_1.0"] for r in results])),
        "miou": float(np.mean([r["iou"] for r in results])),
        "corloc": float(np.mean([r["corloc"] for r in results])),
        "n": len(results),
    }


def print_metrics_table(title, avgs):
    print("\n" + "=" * 50)
    print(f" {title}")
    print("=" * 50)
    if avgs is None:
        print(" No hay imágenes evaluadas correctamente en este bloque.")
        print("=" * 50)
        return
    print(f"{'Métrica':<28}{'Valor':>20}")
    print("-" * 50)
    print(f"{'MAE':<28}{avgs['mae']:>20.4f}")
    print(f"{'Max F-beta (beta^2=0.3)':<28}{avgs['max_fbeta_0.3']:>20.4f}")
    print(f"{'Max F-beta (beta^2=1.0 / F1)':<28}{avgs['max_fbeta_1.0']:>20.4f}")
    print(f"{'mIoU (Otsu)':<28}{avgs['miou']:>20.4f}")
    print(f"{'CorLoc':<28}{avgs['corloc']:>20.4f}")
    print("-" * 50)
    print(f"{'Imágenes evaluadas':<28}{avgs['n']:>20d}")
    print("=" * 50)


def print_all_results(batch_results_list, all_results):
    # Tablas por categoría
    for batch_num, start_photo, end_photo, batch_results in batch_results_list:
        avgs = compute_averages(batch_results)
        title = f"Resultados Categoría {batch_num} (Fotos {start_photo}-{end_photo})"
        print_metrics_table(title, avgs)

    # Tabla resumen total
    total_avgs = compute_averages(all_results)
    print_metrics_table(
        f"RESUMEN TOTAL DEL DATASET ({len(all_results)} imágenes)", total_avgs
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evalúa predicciones de saliency/object discovery contra Ground Truth, "
            "en lotes de 25 imágenes (4 categorías) sobre un dataset de 100 imágenes."
        )
    )
    parser.add_argument("--gt_dir", required=True, help="Ruta a la carpeta con las máscaras GT.")
    parser.add_argument("--pred_dir", required=True, help="Ruta a la carpeta con las predicciones.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Tamaño de cada lote/categoría (default: {BATCH_SIZE}).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.gt_dir):
        print(f"Error: el directorio GT no existe: {args.gt_dir}")
        sys.exit(1)
    if not os.path.isdir(args.pred_dir):
        print(f"Error: el directorio de predicciones no existe: {args.pred_dir}")
        sys.exit(1)

    print(f"GT dir:     {args.gt_dir}")
    print(f"Pred dir:   {args.pred_dir}")
    print(f"Batch size: {args.batch_size}")

    batch_results_list, all_results = evaluate_dataset(
        args.gt_dir, args.pred_dir, batch_size=args.batch_size
    )
    print_all_results(batch_results_list, all_results)


if __name__ == "__main__":
    main()