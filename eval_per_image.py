#!/usr/bin/env python3
"""
eval_per_image.py

Calcula las métricas (MAE, F-beta 0.3, F1, mIoU, CorLoc) por imagen y por
método sobre el conjunto de evaluación (70 imágenes = 100 - 30 de tuning con
seed R=14), a partir de las máscaras ya generadas por la corrida canónica del
paper (outputs_all_methods_holdout, DINOv2 con k_std=0.7 y baselines con
k_std=0.5).

Salida:
  - results_per_image.csv  (columnas: image, category, method, mae, f03, f1,
                            miou, corloc)
  - Validación cruzada de las medias contra outputs_all_methods_holdout/report.txt;
    el script aborta si alguna media difiere más de TOL.

Uso:
    .venv/bin/python eval_per_image.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import eval_saliency

# ─── Configuración ────────────────────────────────────────────────────────────

GT_DIR = PROJECT_ROOT / "got_masks_v2"
INPUT_DIR = PROJECT_ROOT / "Dataset_tesis_numerado_ligero"
RUN_DIR = PROJECT_ROOT / "outputs_all_methods_holdout"

RANDOM_SEED = 14          # misma semilla que tune_dinov2.py (R=14)
N_TUNING = 30             # tamaño del subset de tuning
N_TOTAL = 100             # imágenes totales del dataset

METHODS = {
    # method_name: (directorio de máscaras, k_std de evaluación)
    "TokenCut": (RUN_DIR / "tokencut" / "masks", 0.5),
    "LOST":     (RUN_DIR / "lost" / "masks", 0.5),
    "DINOv2":   (RUN_DIR / "dinov2", 0.7),
}

OUT_CSV = PROJECT_ROOT / "results_per_image.csv"
TOL = 0.002  # tolerancia para la validación cruzada contra el reporte

# Medias esperadas (outputs_all_methods_holdout/report.txt), estructura:
# {metodo: {("C1".."C4" o "TOTAL"): (mae, f03, f1, miou, corloc)}}
EXPECTED = {
    "TokenCut": {
        "C1": (0.0904, 0.5806, 0.5987, 0.3280, 0.1579),
        "C2": (0.1477, 0.4046, 0.3992, 0.1673, 0.1000),
        "C3": (0.0257, 0.8895, 0.8677, 0.5226, 0.5714),
        "C4": (0.0896, 0.3883, 0.4227, 0.1713, 0.0000),
        "TOTAL": (0.0936, 0.5454, 0.5528, 0.2829, 0.1857),
    },
    "LOST": {
        "C1": (0.1761, 0.5541, 0.5640, 0.2662, 0.0526),
        "C2": (0.1570, 0.3915, 0.3886, 0.1697, 0.0000),
        "C3": (0.0845, 0.8833, 0.8503, 0.3944, 0.5000),
        "C4": (0.1861, 0.3646, 0.3802, 0.1602, 0.0000),
        "TOTAL": (0.1547, 0.5275, 0.5265, 0.2385, 0.1143),
    },
    "DINOv2": {
        "C1": (0.0610, 0.4531, 0.4610, 0.3126, 0.1579),
        "C2": (0.0850, 0.3237, 0.3666, 0.2249, 0.0500),
        "C3": (0.0275, 0.6138, 0.6337, 0.4768, 0.4286),
        "C4": (0.0466, 0.4721, 0.5099, 0.3537, 0.1176),
        "TOTAL": (0.0577, 0.4529, 0.4804, 0.3304, 0.1714),
    },
}

METRIC_COLS = ["mae", "f03", "f1", "miou", "corloc"]


# ─── Utilidades ───────────────────────────────────────────────────────────────

def tuning_ids():
    """Reproduce la selección de tune_dinov2.py: rng(14).choice(1..100, 30)."""
    all_ids = list(range(1, N_TOTAL + 1))
    rng = np.random.default_rng(RANDOM_SEED)
    chosen = rng.choice(all_ids, size=N_TUNING, replace=False)
    return set(int(x) for x in chosen)


def category_of(img_id):
    """Categoría secuencial del dataset: 1-25→C1, 26-50→C2, 51-75→C3, 76-100→C4."""
    return f"C{(img_id - 1) // 25 + 1}"


def gt_map_by_id():
    """GT indexado por ID numérico de imagen."""
    gt = {}
    for p in sorted(GT_DIR.glob("*.png")):
        try:
            gt[int(p.stem)] = p
        except ValueError:
            continue
    return gt


def pred_map_by_id(pred_dir):
    """Predicciones indexadas por ID (ignora _heatmap/_overlay, como eval_saliency)."""
    preds = {}
    for p in sorted(Path(pred_dir).glob("*.png")):
        stem = p.stem
        if stem.endswith("_heatmap") or stem.endswith("_overlay"):
            continue
        try:
            preds[int(stem)] = p
        except ValueError:
            continue
    return preds


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t_ids = tuning_ids()
    eval_ids = [i for i in range(1, N_TOTAL + 1) if i not in t_ids]
    print(f"[Split] Tuning ({len(t_ids)}): {sorted(t_ids)}")
    print(f"[Split] Evaluación ({len(eval_ids)}): {eval_ids}")

    gt = gt_map_by_id()
    rows = []
    for method, (pred_dir, k_std) in METHODS.items():
        preds = pred_map_by_id(pred_dir)
        missing = [i for i in eval_ids if i not in preds]
        if missing:
            print(f"[ERROR] {method}: faltan predicciones para IDs {missing}")
            sys.exit(1)
        for img_id in eval_ids:
            metrics = eval_saliency.evaluate_pair(
                str(preds[img_id]), str(gt[img_id]), k_std=k_std
            )
            if metrics is None:
                print(f"[ERROR] {method} ID {img_id}: falló evaluate_pair")
                sys.exit(1)
            rows.append({
                "image": f"{img_id:03d}",
                "category": category_of(img_id),
                "method": method,
                "mae": metrics["mae"],
                "f03": metrics["max_fbeta_0.3"],
                "f1": metrics["max_fbeta_1.0"],
                "miou": metrics["iou"],
                "corloc": metrics["corloc"],
            })
        print(f"[OK] {method}: {len(eval_ids)} imágenes evaluadas (k_std={k_std})")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[CSV] Guardado: {OUT_CSV} ({len(df)} filas)")

    # ─── Validación cruzada contra el reporte ────────────────────────────────
    print("\n[Validación] Comparando medias contra report.txt ...")
    failed = False
    for method in METHODS:
        sub = df[df["method"] == method]
        groups = dict(list(sub.groupby("category")))
        groups["TOTAL"] = sub
        for gname, gdf in groups.items():
            got = tuple(gdf[m].mean() for m in METRIC_COLS)
            exp = EXPECTED[method][gname]
            diffs = [abs(g - e) for g, e in zip(got, exp)]
            if max(diffs) > TOL:
                failed = True
                print(f"  [FALLO] {method}/{gname}: got={got} expected={exp}")
            else:
                print(f"  [OK] {method}/{gname} (max diff {max(diffs):.5f})")

    if failed:
        print("\n[ERROR] La validación cruzada FALLÓ: el CSV no reproduce el "
              "reporte canónico. Revisa el split o las máscaras antes de seguir.")
        sys.exit(1)
    print("\n[Validación] OK: todas las medias coinciden con el reporte canónico.")


if __name__ == "__main__":
    main()
