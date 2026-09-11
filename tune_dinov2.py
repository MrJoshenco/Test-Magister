#!/usr/bin/env python3
"""
tune_dinov2.py

Búsqueda de hiperparámetros de DINOv2.py optimizada para una GPU con ~24 GB VRAM
(p. ej. RTX 3090).  El script:

  * Selecciona 50 imágenes aleatorias del dataset (con semilla fija).
  * Explora un grid denso de hiperparámetros.
  * Permite búsqueda aleatoria (random search) sobre ese grid para mantener el
    tiempo de ejecución razonable sin perder cobertura del espacio.
  * Aprovecha la VRAM disponible probando resoluciones más altas (770 px).
  * Guarda progreso parcial a CSV para recuperar resultados si se interrumpe.

Uso:
    .venv/bin/python tune_dinov2.py

Ajusta SUBSET_SIZE, RANDOM_SEED, PARAM_GRID, RANDOM_SEARCH y
N_RANDOM_SAMPLES directamente en este script.
"""

import sys
import os
import time
import shutil
import tempfile
import itertools
import csv
from pathlib import Path
import numpy as np
import torch

# ─── Paths del proyecto ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import DINOv2
import eval_saliency

INPUT_DIR = PROJECT_ROOT / "Dataset_tesis_numerado_ligero"
GT_DIR = PROJECT_ROOT / "got_masks_v2"

# ─── Configuración del experimento ──────────────────────────────────────────────

# Semilla para reproducibilidad (selección de imágenes y de combinaciones).
RANDOM_SEED = 14

# Número de imágenes aleatorias del dataset a evaluar.
SUBSET_SIZE = 30

# Grid denso de hiperparámetros a sintonizar.  Con una 3090 (24 GB VRAM) se puede
# probar MAX_RESOLUTION=770 sin problemas; 980 es factible para muchas imágenes,
# pero se deja comentado por si se desea un margen de seguridad.
PARAM_GRID = {
    "MAX_RESOLUTION": [518, 980],  # 518 es el default de DINOv2.py
    "TEMPERATURE": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "GUIDED_FILTER_RADIUS": [5, 8, 12, 15, 20, 25, 30],
    "K_STD": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
    "GAUSSIAN_SIGMA": [0.5, 1.0, 1.5],
    "PERCENTILE_LOW": [0, 1, 2],
    "PERCENTILE_HIGH": [98, 99, 100],
}

# Si True, en lugar de recorrer TODO el grid se muestrean N_RANDOM_SAMPLES
# combinaciones aleatorias.  Útil para hacer una búsqueda más exhaustiva del
# espacio sin que el tiempo explote.
RANDOM_SEARCH = True
N_RANDOM_SAMPLES = 100

# Parámetros fijos para esta corrida
FIXED_PARAMS = {
    "GUIDED_FILTER_EPS": 1e-3,
    "ALPHA_OVERLAY": 0.6,
    "PATCH_SIZE": 14,
}

# Archivo CSV donde se escribe el progreso parcial (útil si se interrumpe).
PARTIAL_CSV = PROJECT_ROOT / "tune_dinov2_partial.csv"

EPS = 1e-8


# ─── Utilidades ───────────────────────────────────────────────────────────────


def discover_input_ids():
    """Descubre todos los IDs numéricos de imágenes disponibles en INPUT_DIR."""
    ids = set()
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for path in INPUT_DIR.glob(ext):
            try:
                ids.add(int(path.stem))
            except ValueError:
                continue
    return sorted(ids)


def select_random_subset(all_ids, n, seed):
    """Selecciona n IDs aleatorios reproducibles."""
    rng = np.random.default_rng(seed)
    if n > len(all_ids):
        print(f"[AVISO] Se pidieron {n} imágenes pero solo hay {len(all_ids)}. Usando todas.")
        n = len(all_ids)
    chosen = rng.choice(all_ids, size=n, replace=False)
    return sorted(int(x) for x in chosen)


def apply_config(config):
    """Aplica el diccionario de configuración como constantes de módulo DINOv2."""
    for key, value in config.items():
        if hasattr(DINOv2, key):
            setattr(DINOv2, key, value)
        else:
            print(f"[AVISO] DINOv2 no tiene la constante '{key}'")


def load_model(device):
    """Carga DINOv2 una sola vez y devuelve (model, extractor)."""
    print("[Cargando modelo] DINOv2 ViT-Base con registers...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg").to(device)
    model.eval()
    extractor = DINOv2.DINOv2AttentionExtractor(model)
    return model, extractor


def get_image_paths(subset_ids):
    """Busca imágenes de entrada con los IDs indicados (.jpg, .jpeg o .png)."""
    paths = []
    for i in subset_ids:
        candidates = [
            INPUT_DIR / f"{i:03d}.jpg",
            INPUT_DIR / f"{i}.jpg",
            INPUT_DIR / f"{i:03d}.jpeg",
            INPUT_DIR / f"{i}.jpeg",
            INPUT_DIR / f"{i:03d}.png",
            INPUT_DIR / f"{i}.png",
        ]
        found = [p for p in candidates if p.exists()]
        if found:
            paths.append(str(found[0]))
        else:
            print(f"[AVISO] No se encontró imagen de entrada para ID {i}")
    return paths


def _numeric_id_from_filename(fname):
    """Extrae el ID numérico entero de un nombre de archivo, ej: '001.png' -> 1."""
    stem = Path(fname).stem
    for suffix in ("_heatmap", "_overlay"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    try:
        return int(stem)
    except ValueError:
        return None


def evaluate_output(output_dir, gt_dir, k_std):
    """
    Evalúa las máscaras generadas en output_dir contra gt_dir.
    Empareja por ID numérico, tolerando padding diferente.
    Devuelve un diccionario plano con las medias de las métricas.
    """
    gt_map = {}
    for fname in eval_saliency.list_sorted_pngs(gt_dir):
        img_id = _numeric_id_from_filename(fname)
        if img_id is not None:
            gt_map[img_id] = Path(gt_dir) / fname

    results = []
    for pred_fname in eval_saliency.list_sorted_pngs(output_dir):
        img_id = _numeric_id_from_filename(pred_fname)
        if img_id is None or img_id not in gt_map:
            continue
        pred_path = Path(output_dir) / pred_fname
        gt_path = gt_map[img_id]
        metrics = eval_saliency.evaluate_pair(str(pred_path), str(gt_path), k_std=k_std)
        if metrics is not None:
            results.append(metrics)

    stats = eval_saliency.compute_stats(results)
    if stats is None:
        return None

    return {
        "mae": stats["mae"]["mean"],
        "max_fbeta_0.3": stats["max_fbeta_0.3"]["mean"],
        "max_fbeta_1.0": stats["max_fbeta_1.0"]["mean"],
        "miou": stats["iou"]["mean"],
        "corloc": stats["corloc"]["mean"],
        "n": stats["n"],
    }


def run_experiment(image_paths, output_dir, config, model, extractor, device):
    """Ejecuta una combinación de hiperparámetros y devuelve cuántas imágenes OK."""
    apply_config(config)

    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    for path in image_paths:
        success = DINOv2.process_image(path, model, extractor, str(output_dir), device)
        if success:
            ok += 1
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return ok


def format_config_short(config):
    """Cadena corta para identificar una combinación."""
    return (
        f"R={config['MAX_RESOLUTION']:<4} "
        f"T={config['TEMPERATURE']:<4} "
        f"r={config['GUIDED_FILTER_RADIUS']:<3} "
        f"k={config['K_STD']:<4} "
        f"g={config['GAUSSIAN_SIGMA']:<4} "
        f"p={config['PERCENTILE_LOW']}/{config['PERCENTILE_HIGH']:<7}"
    )


def write_partial_csv(row, csv_path):
    """Añade una fila de resultados al CSV parcial, creando encabezado si es nuevo."""
    fieldnames = [
        "rank", "MAX_RESOLUTION", "TEMPERATURE", "GUIDED_FILTER_RADIUS", "K_STD",
        "GAUSSIAN_SIGMA", "PERCENTILE_LOW", "PERCENTILE_HIGH",
        "mae", "max_fbeta_0.3", "max_fbeta_1.0", "miou", "corloc", "n", "ok", "elapsed"
    ]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hardware] Usando: {device.type.upper()}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"[Hardware] GPU: {props.name}  VRAM: {vram_gb:.1f} GB")

    if not INPUT_DIR.exists():
        print(f"[ERROR] No existe el directorio de entrada: {INPUT_DIR}")
        return
    if not GT_DIR.exists():
        print(f"[ERROR] No existe el directorio GT: {GT_DIR}")
        return

    all_ids = discover_input_ids()
    if not all_ids:
        print(f"[ERROR] No se encontraron imágenes en {INPUT_DIR}")
        return

    SUBSET_IDS = select_random_subset(all_ids, SUBSET_SIZE, RANDOM_SEED)
    image_paths = get_image_paths(SUBSET_IDS)
    if not image_paths:
        print(f"[ERROR] No se encontraron imágenes de entrada en {INPUT_DIR}")
        return

    # Prepara combinaciones (grid completo o random search).
    keys = list(PARAM_GRID.keys())
    full_grid = list(itertools.product(*PARAM_GRID.values()))
    if RANDOM_SEARCH:
        rng = np.random.default_rng(RANDOM_SEED)
        if N_RANDOM_SAMPLES >= len(full_grid):
            combinations = full_grid
        else:
            idx = rng.choice(len(full_grid), size=N_RANDOM_SAMPLES, replace=False)
            combinations = [full_grid[i] for i in idx]
    else:
        combinations = full_grid

    n_comb = len(combinations)
    print(f"[Subset]   {len(image_paths)} imágenes aleatorias (seed={RANDOM_SEED}): "
          f"{[Path(p).name for p in image_paths[:5]]}{'...' if len(image_paths) > 5 else ''}")
    print(f"[GT]       {GT_DIR}")
    print(f"[Grid]     {n_comb} combinaciones de un espacio total de {len(full_grid)} "
          f"({'random search' if RANDOM_SEARCH else 'grid completo'})")
    print(f"[Parcial]  Resultados se guardan en: {PARTIAL_CSV}")
    print("-" * 80)

    model, extractor = load_model(device)

    results = []
    base_tmp = Path(tempfile.mkdtemp(prefix="tune_dinov2_"))
    start_global = time.time()

    # Elimina CSV parcial previo para empezar limpio.
    if PARTIAL_CSV.exists():
        PARTIAL_CSV.unlink()

    try:
        for idx, values in enumerate(combinations, 1):
            config = dict(FIXED_PARAMS)
            config.update(dict(zip(keys, values)))

            out_dir = base_tmp / f"comb_{idx:05d}"
            print(f"\n[{idx:>3}/{n_comb}] {format_config_short(config)}")

            start = time.time()
            try:
                ok = run_experiment(image_paths, out_dir, config, model, extractor, device)
                elapsed = time.time() - start

                avgs = evaluate_output(out_dir, GT_DIR, k_std=config["K_STD"])

                if avgs is not None:
                    result = {
                        "config": config,
                        "mae": avgs["mae"],
                        "max_fbeta_0.3": avgs["max_fbeta_0.3"],
                        "max_fbeta_1.0": avgs["max_fbeta_1.0"],
                        "miou": avgs["miou"],
                        "corloc": avgs["corloc"],
                        "n": avgs["n"],
                        "ok": ok,
                        "elapsed": elapsed,
                    }
                    results.append(result)
                    print(f"       MAE={avgs['mae']:.4f}  mIoU={avgs['miou']:.4f}  "
                          f"F1={avgs['max_fbeta_1.0']:.4f}  Fb03={avgs['max_fbeta_0.3']:.4f}  "
                          f"CorLoc={avgs['corloc']:.4f}  time={elapsed:.1f}s")

                    write_partial_csv({
                        "rank": len(results),
                        "MAX_RESOLUTION": config["MAX_RESOLUTION"],
                        "TEMPERATURE": config["TEMPERATURE"],
                        "GUIDED_FILTER_RADIUS": config["GUIDED_FILTER_RADIUS"],
                        "K_STD": config["K_STD"],
                        "GAUSSIAN_SIGMA": config["GAUSSIAN_SIGMA"],
                        "PERCENTILE_LOW": config["PERCENTILE_LOW"],
                        "PERCENTILE_HIGH": config["PERCENTILE_HIGH"],
                        "mae": avgs["mae"],
                        "max_fbeta_0.3": avgs["max_fbeta_0.3"],
                        "max_fbeta_1.0": avgs["max_fbeta_1.0"],
                        "miou": avgs["miou"],
                        "corloc": avgs["corloc"],
                        "n": avgs["n"],
                        "ok": ok,
                        "elapsed": elapsed,
                    }, PARTIAL_CSV)
                else:
                    print(f"       [AVISO] No se pudieron evaluar resultados.")

                # Estimación de tiempo restante.
                elapsed_global = time.time() - start_global
                avg_per_comb = elapsed_global / idx
                remaining = avg_per_comb * (n_comb - idx)
                print(f"       [ETA] ~{remaining / 3600:.1f}h restantes (promedio {avg_per_comb:.1f}s/comb)")
            except Exception as e:
                print(f"       [ERROR] {e}")
                continue
    finally:
        extractor.close()
        shutil.rmtree(base_tmp, ignore_errors=True)

    if not results:
        print("\n[ERROR] Ninguna combinación produjo resultados evaluables.")
        return

    # ─── Tabla ordenada por mIoU ─────────────────────────────────────────────
    results_sorted = sorted(results, key=lambda x: x["miou"], reverse=True)

    print("\n" + "=" * 110)
    print(" RESULTADOS ORDENADOS POR mIoU (umbral relativo: mean + k_std * std)")
    print("=" * 110)
    print(f"{'Rank':<5}{'R':<6}{'T':<6}{'r':<5}{'k':<6}{'g':<6}{'pl':<4}{'ph':<4}"
          f"{'mIoU':<8}{'F1':<8}{'Fβ0.3':<8}{'MAE':<8}{'CorLoc':<9}{'Time':<8}")
    print("-" * 110)
    for rank, r in enumerate(results_sorted, 1):
        c = r["config"]
        print(f"{rank:<5}"
              f"{c['MAX_RESOLUTION']:<6}"
              f"{c['TEMPERATURE']:<6.1f}"
              f"{c['GUIDED_FILTER_RADIUS']:<5}"
              f"{c['K_STD']:<6.2f}"
              f"{c['GAUSSIAN_SIGMA']:<6.1f}"
              f"{c['PERCENTILE_LOW']:<4}"
              f"{c['PERCENTILE_HIGH']:<4}"
              f"{r['miou']:<8.4f}"
              f"{r['max_fbeta_1.0']:<8.4f}"
              f"{r['max_fbeta_0.3']:<8.4f}"
              f"{r['mae']:<8.4f}"
              f"{r['corloc']:<9.4f}"
              f"{r['elapsed']:<8.1f}")
    print("=" * 110)

    # ─── Mejor configuración ─────────────────────────────────────────────────
    best = results_sorted[0]
    print("\nMejor configuración (por mIoU):")
    for key, value in sorted(best["config"].items()):
        print(f"  {key:<22} = {value}")
    print(f"  Métricas: mIoU={best['miou']:.4f}, F1={best['max_fbeta_1.0']:.4f}, "
          f"MAE={best['mae']:.4f}, CorLoc={best['corloc']:.4f}")


if __name__ == "__main__":
    main()
