#!/usr/bin/env python3
"""
tune_dinov2.py

Grid search de hiperparámetros ajustables de DINOv2.py.
Carga el modelo una sola vez, itera sobre combinaciones de hiperparámetros,
genera máscaras para un subconjunto de imágenes y evalúa contra el ground-truth.

Uso:
    .venv/bin/python tune_dinov2.py

Ajusta SUBSET_IDS, PARAM_GRID y FIXED_PARAMS directamente en este script.
"""

import sys
import os
import time
import shutil
import tempfile
import itertools
from pathlib import Path
import numpy as np
import torch

# ─── Paths del proyecto ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import DINOv2
import eval_saliency

INPUT_DIR = PROJECT_ROOT / "Dataset_tesis_numerado"
GT_DIR = PROJECT_ROOT / "got_masks_v2"

# ─── Configuración del experimento ────────────────────────────────────────────

# Subconjunto de imágenes a evaluar (IDs numéricos). Reducir para acortar tiempo.
SUBSET_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Grid de hiperparámetros a sintonizar
PARAM_GRID = {
    "TEMPERATURE": [0.3, 0.5, 0.7],
    "GUIDED_FILTER_RADIUS": [8, 15, 25],
    "K_STD": [0.3, 0.5, 0.7],
}

# Parámetros fijos para esta corrida
FIXED_PARAMS = {
    "MAX_RESOLUTION": 518,
    "GAUSSIAN_SIGMA": 1.0,
    "PERCENTILE_LOW": 2,
    "PERCENTILE_HIGH": 98,
    "GUIDED_FILTER_EPS": 1e-3,
    "ALPHA_OVERLAY": 0.6,
    "PATCH_SIZE": 14,
}

EPS = 1e-8


# ─── Utilidades ───────────────────────────────────────────────────────────────


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
    """Busca imágenes de entrada con los IDs indicados (.jpg o .png)."""
    paths = []
    for i in subset_ids:
        candidates = [
            INPUT_DIR / f"{i}.jpg",
            INPUT_DIR / f"{i}.jpeg",
            INPUT_DIR / f"{i}.png",
        ]
        found = [p for p in candidates if p.exists()]
        if found:
            paths.append(str(found[0]))
        else:
            print(f"[AVISO] No se encontró imagen de entrada para ID {i}")
    return paths


def _numeric_id_from_filename(fname):
    """Extrae el ID numérico entero de un nombre de archivo, ej: '001.png' -> 1, '42.png' -> 42."""
    stem = Path(fname).stem
    # Elimina sufijos de visualización si los hubiera
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
    Empareja por ID numérico, tolerando padding diferente (ej: pred 1.png vs GT 001.png).
    """
    # Mapa GT: id -> path
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
    return eval_saliency.compute_averages(results)


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
        f"T={config['TEMPERATURE']:<4} "
        f"r={config['GUIDED_FILTER_RADIUS']:<3} "
        f"k={config['K_STD']:<4}"
    )


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hardware] Usando: {device.type.upper()}")

    if not INPUT_DIR.exists():
        print(f"[ERROR] No existe el directorio de entrada: {INPUT_DIR}")
        return
    if not GT_DIR.exists():
        print(f"[ERROR] No existe el directorio GT: {GT_DIR}")
        return

    image_paths = get_image_paths(SUBSET_IDS)
    if not image_paths:
        print(f"[ERROR] No se encontraron imágenes de entrada en {INPUT_DIR}")
        return

    n_comb = int(np.prod([len(v) for v in PARAM_GRID.values()]))
    print(f"[Subset]   {len(image_paths)} imágenes: {[Path(p).name for p in image_paths]}")
    print(f"[GT]       {GT_DIR}")
    print(f"[Grid]     {n_comb} combinaciones")
    print(f"[Fijos]    MAX_RESOLUTION={FIXED_PARAMS['MAX_RESOLUTION']}, "
          f"GAUSSIAN_SIGMA={FIXED_PARAMS['GAUSSIAN_SIGMA']}, "
          f"percentiles={FIXED_PARAMS['PERCENTILE_LOW']}/{FIXED_PARAMS['PERCENTILE_HIGH']}")
    print("-" * 80)

    model, extractor = load_model(device)

    keys = list(PARAM_GRID.keys())
    combinations = list(itertools.product(*PARAM_GRID.values()))

    results = []
    base_tmp = Path(tempfile.mkdtemp(prefix="tune_dinov2_"))

    try:
        for idx, values in enumerate(combinations, 1):
            config = dict(FIXED_PARAMS)
            config.update(dict(zip(keys, values)))

            out_dir = base_tmp / f"comb_{idx:03d}"
            print(f"\n[{idx:>2}/{len(combinations)}] {format_config_short(config)}")

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
                else:
                    print(f"       [AVISO] No se pudieron evaluar resultados.")
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

    print("\n" + "=" * 100)
    print(" RESULTADOS ORDENADOS POR mIoU (umbral relativo: mean + k_std * std)")
    print("=" * 100)
    print(f"{'Rank':<5}{'T':<6}{'r':<5}{'k_std':<7}{'mIoU':<8}{'F1':<8}{"Fβ0.3":<8}{'MAE':<8}{'CorLoc':<9}{'Time':<8}")
    print("-" * 100)
    for rank, r in enumerate(results_sorted, 1):
        print(f"{rank:<5}"
              f"{r['config']['TEMPERATURE']:<6.1f}"
              f"{r['config']['GUIDED_FILTER_RADIUS']:<5}"
              f"{r['config']['K_STD']:<7.2f}"
              f"{r['miou']:<8.4f}"
              f"{r['max_fbeta_1.0']:<8.4f}"
              f"{r['max_fbeta_0.3']:<8.4f}"
              f"{r['mae']:<8.4f}"
              f"{r['corloc']:<9.4f}"
              f"{r['elapsed']:<8.1f}")
    print("=" * 100)

    # ─── Mejor configuración ─────────────────────────────────────────────────
    best = results_sorted[0]
    print("\nMejor configuración (por mIoU):")
    print(f"  TEMPERATURE          = {best['config']['TEMPERATURE']}")
    print(f"  GUIDED_FILTER_RADIUS = {best['config']['GUIDED_FILTER_RADIUS']}")
    print(f"  K_STD                = {best['config']['K_STD']}")
    print(f"  Métricas: mIoU={best['miou']:.4f}, F1={best['max_fbeta_1.0']:.4f}, "
          f"MAE={best['mae']:.4f}, CorLoc={best['corloc']:.4f}")


if __name__ == "__main__":
    main()
