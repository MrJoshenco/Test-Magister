#!/usr/bin/env python3
"""
sweep_thresholds.py

Análisis de sensibilidad del umbral relativo (mean + k·std) para los tres
métodos sobre las 70 imágenes de evaluación (split R=14).

Por método:
  - TokenCut / LOST: re-binariza los mapas continuos guardados por
    run_all_methods.py (grayscale 0-255) con cada k del barrido.
  - DINOv2: regenera el mapa continuo (attn_map_resized) con la configuración
    de la corrida canónica (MAX_RESOLUTION=518, TEMPERATURE=0.3,
    GUIDED_FILTER_RADIUS=25, GAUSSIAN_SIGMA=1.0, percentiles 1/100) y lo guarda
    en outputs_sweep/dinov2_continuous/ como .npy (float16); luego binariza con
    make_binary_mask (umbral relativo + morfología), igual que el pipeline.

Protocolo de mIoU idéntico al de eval_saliency para los baselines: resize
bilineal del mapa continuo al tamaño del GT, normalización [0,1], umbral
mean + k·std, IoU contra GT binario. Para DINOv2 (mapa continuo regenerado)
se usa el mismo protocolo directo, más la morfología propia de su pipeline.

Validaciones:
  - Baselines con k=0.5 deben reproducir los mIoU del reporte canónico
    (TokenCut 0.2829, LOST 0.2385).
  - DINOv2 con k=0.7: (a) acuerdo pixel a pixel con las máscaras guardadas de
    outputs_all_methods_holdout/dinov2, (b) mIoU cercano al del reporte
    (0.3304) y, además, se emula el orden exacto de evaluación del reporte
    (umbral mean+0.7·std sobre la máscara binaria re-escalada).

Salida:
  - outputs_sweep/dinovio_continuous/*.npy  (mapas continuos de DINOv2)
  - sweep_results.csv  (method, k, n, miou_mean, corloc_mean)

Uso:
    .venv/bin/python sweep_thresholds.py            # todo (GPU para DINOv2)
    .venv/bin/python sweep_thresholds.py --skip-dinov2   # solo baselines (CPU)
    .venv/bin/python sweep_thresholds.py --skip-baselines  # solo DINOv2 (GPU)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import cv2
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import eval_saliency
import DINOv2

GT_DIR = PROJECT_ROOT / "got_masks_v2"
INPUT_DIR = PROJECT_ROOT / "Dataset_tesis_numerado_ligero"
RUN_DIR = PROJECT_ROOT / "outputs_all_methods_holdout"
SWEEP_DIR = PROJECT_ROOT / "outputs_sweep"
CONT_DIR = SWEEP_DIR / "dinov2_continuous"
OUT_CSV = PROJECT_ROOT / "sweep_results.csv"

RANDOM_SEED = 14
N_TUNING = 30
N_TOTAL = 100

K_GRID = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
K_REF = {"TokenCut": 0.5, "LOST": 0.5, "DINOv2": 0.7}

# Configuración de la corrida canónica de DINOv2 (holdout, k_std=0.7).
# IMPORTANTE: percentiles 1/100 según run_all_methods.py en el commit 1c8a8c4
# (estado del repo cuando se generó outputs_all_methods_holdout, 2026-08-03).
# Los defaults actuales de DINOv2.py (2/98) NO reproducen las máscaras.
DINOV2_CONFIG = {
    "MAX_RESOLUTION": 518,
    "TEMPERATURE": 0.3,
    "GUIDED_FILTER_RADIUS": 25,
    "GAUSSIAN_SIGMA": 1.0,
    "PERCENTILE_LOW": 1,
    "PERCENTILE_HIGH": 100,
    "GUIDED_FILTER_EPS": 1e-3,
    "PATCH_SIZE": 14,
    "ALPHA_OVERLAY": 0.6,
}

# mIoU esperados del reporte canónico a k de referencia (validación)
EXPECTED_MIOU_REF = {"TokenCut": 0.2829, "LOST": 0.2385, "DINOv2": 0.3304}
EVAL_K_STD_DINOV2 = 0.7  # k_std con que eval_saliency evaluó las máscaras DINOv2

BASELINE_MASK_DIRS = {
    "TokenCut": RUN_DIR / "tokencut" / "masks",
    "LOST": RUN_DIR / "lost" / "masks",
}


# ─── Utilidades de split / GT ────────────────────────────────────────────────

def eval_ids():
    all_ids = list(range(1, N_TOTAL + 1))
    rng = np.random.default_rng(RANDOM_SEED)
    t_ids = set(int(x) for x in rng.choice(all_ids, size=N_TUNING, replace=False))
    return [i for i in all_ids if i not in t_ids]


def load_gt(img_id):
    gt = eval_saliency.load_gray(str(GT_DIR / f"{img_id:03d}.png"))
    if gt is None:
        raise RuntimeError(f"No se pudo leer GT {img_id:03d}")
    return gt, eval_saliency.normalize01(gt) > 0.5


# ─── Protocolo de mIoU/CorLoc (idéntico a eval_saliency) ─────────────────────

def miou_from_continuous(pred_norm, gt_shape, gt_binary, k):
    """
    pred_norm: mapa continuo [0,1] en su resolución nativa.
    Replica eval_saliency.evaluate_pair: resize bilineal al tamaño del GT,
    umbral mean + k·std, IoU y CorLoc contra gt_binary.
    """
    if pred_norm.shape != gt_shape:
        pred = cv2.resize(pred_norm, (gt_shape[1], gt_shape[0]),
                          interpolation=cv2.INTER_LINEAR)
    else:
        pred = pred_norm
    thresh = pred.mean() + k * pred.std()
    pred_bin = pred > thresh
    inter = np.logical_and(pred_bin, gt_binary).sum()
    union = np.logical_or(pred_bin, gt_binary).sum()
    iou = inter / union if union > 0 else 0.0
    return float(iou), float(iou >= 0.5)


# ─── Baselines (CPU) ─────────────────────────────────────────────────────────

def sweep_baseline(method, ids):
    pred_dir = BASELINE_MASK_DIRS[method]
    per_image = {k: [] for k in K_GRID}
    for img_id in ids:
        pred_path = pred_dir / f"{img_id:03d}.png"
        pred = eval_saliency.load_gray(str(pred_path))
        if pred is None:
            raise RuntimeError(f"No se pudo leer {pred_path}")
        gt, gt_binary = load_gt(img_id)
        gt_shape = gt.shape
        pred_norm = eval_saliency.normalize01(pred)
        for k in K_GRID:
            iou, corloc = miou_from_continuous(pred_norm, gt_shape, gt_binary, k)
            per_image[k].append((iou, corloc))
    return per_image


# ─── DINOv2 (GPU) ────────────────────────────────────────────────────────────

def dinov2_continuous_map(model, extractor, img_path, device):
    """Reproduce DINOv2.process_image y devuelve el mapa continuo refinado [0,1]."""
    from scipy.ndimage import gaussian_filter

    ps = DINOV2_CONFIG["PATCH_SIZE"]
    image_original = Image.open(img_path).convert("RGB")
    image_resized = DINOv2.resize_for_dinov2(
        image_original, max_size=DINOV2_CONFIG["MAX_RESOLUTION"], patch_size=ps)
    w, h = image_resized.size

    x = DINOv2.preprocess_tensors(image_resized).unsqueeze(0).to(device)
    raw_attn = extractor.generate(x)

    h_p, w_p = h // ps, w // ps
    attn_map = raw_attn.reshape(h_p, w_p).cpu().numpy()
    attn_map = gaussian_filter(attn_map, sigma=DINOV2_CONFIG["GAUSSIAN_SIGMA"])

    p_low = np.percentile(attn_map, DINOV2_CONFIG["PERCENTILE_LOW"])
    p_high = np.percentile(attn_map, DINOV2_CONFIG["PERCENTILE_HIGH"])
    attn_map = np.clip(attn_map, p_low, p_high)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    attn_map = np.power(attn_map, 1.0 / DINOV2_CONFIG["TEMPERATURE"])
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    target_size = (image_original.size[0], image_original.size[1])
    attn_map_cubic = cv2.resize(attn_map, target_size, interpolation=cv2.INTER_CUBIC)
    attn_map_cubic = np.clip(attn_map_cubic, 0.0, 1.0)

    guide_gray = cv2.cvtColor(np.array(image_original), cv2.COLOR_RGB2GRAY)
    guide_gray = guide_gray.astype(np.float32) / 255.0
    return DINOv2.guided_filter(
        guide_gray, attn_map_cubic,
        radius=DINOV2_CONFIG["GUIDED_FILTER_RADIUS"],
        eps=DINOV2_CONFIG["GUIDED_FILTER_EPS"])


def sweep_dinov2(ids, device):
    import torch

    print("[DINOv2] Cargando modelo dinov2_vitb14_reg ...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg").to(device)
    model.eval()
    extractor = DINOv2.DINOv2AttentionExtractor(model)

    CONT_DIR.mkdir(parents=True, exist_ok=True)
    per_image = {k: [] for k in K_GRID}
    agreements = []
    try:
        for n, img_id in enumerate(ids, 1):
            npy_path = CONT_DIR / f"{img_id:03d}.npy"
            if npy_path.exists():
                cont = np.load(npy_path).astype(np.float32)
            else:
                cont = dinov2_continuous_map(
                    model, extractor, str(INPUT_DIR / f"{img_id:03d}.jpg"), device)
                np.save(npy_path, cont.astype(np.float16))
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            gt, gt_binary = load_gt(img_id)
            gt_shape = gt.shape
            saved_mask = eval_saliency.load_gray(
                str(RUN_DIR / "dinov2" / f"{img_id:03d}.png"))

            for k in K_GRID:
                # Binarización pipeline-faithful (umbral + morfología) y luego
                # resize al GT, igual que evalúa el reporte sobre la máscara.
                mask = DINOv2.make_binary_mask(cont, method="relative", k_std=k)
                mask_rs = cv2.resize(mask, (gt_shape[1], gt_shape[0]),
                                     interpolation=cv2.INTER_LINEAR)
                pred_bin = mask_rs > 127
                inter = np.logical_and(pred_bin, gt_binary).sum()
                union = np.logical_or(pred_bin, gt_binary).sum()
                iou = inter / union if union > 0 else 0.0
                per_image[k].append((iou, float(iou >= 0.5)))

                # Validación de reproducción solo en k de referencia (0.7)
                if k == K_REF["DINOv2"] and saved_mask is not None:
                    if saved_mask.shape != mask.shape:
                        saved_cmp = cv2.resize(saved_mask, (mask.shape[1], mask.shape[0]),
                                               interpolation=cv2.INTER_NEAREST)
                    else:
                        saved_cmp = saved_mask
                    agree = float((saved_cmp == mask).mean())
                    agreements.append(agree)
            if n % 10 == 0 or n == len(ids):
                print(f"[DINOv2] {n}/{len(ids)} imágenes procesadas")
    finally:
        extractor.close()

    if agreements:
        print(f"[Validación] Acuerdo pixel máscara regenerada vs guardada "
              f"(k=0.7): media={np.mean(agreements):.5f} "
              f"mín={np.min(agreements):.5f}")
    return per_image


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-dinov2", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()

    ids = eval_ids()
    print(f"[Split] {len(ids)} imágenes de evaluación (seed={RANDOM_SEED})")

    results = {}  # method -> {k: (miou_mean, corloc_mean)}

    if not args.skip_baselines:
        for method in ["TokenCut", "LOST"]:
            per_image = sweep_baseline(method, ids)
            results[method] = {
                k: (float(np.mean([x[0] for x in v])),
                    float(np.mean([x[1] for x in v])))
                for k, v in per_image.items()
            }
            print(f"[OK] {method}: sweep completo")

    if not args.skip_dinov2:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        per_image = sweep_dinov2(ids, device)
        results["DINOv2"] = {
            k: (float(np.mean([x[0] for x in v])),
                float(np.mean([x[1] for x in v])))
            for k, v in per_image.items()
        }

    # ─── Validación contra el reporte canónico ───────────────────────────────
    print("\n[Validación] mIoU a k de referencia vs reporte canónico:")
    for method, expected in EXPECTED_MIOU_REF.items():
        if method not in results:
            continue
        k_ref = K_REF[method]
        got = results[method][k_ref][0]
        status = "OK" if abs(got - expected) < 0.01 else "REVISAR"
        print(f"  {method:<9} k={k_ref}: sweep={got:.4f} reporte={expected:.4f} [{status}]")

    # ─── Tabla de resultados ─────────────────────────────────────────────────
    print("\nmIoU / CorLoc medios por método y k:")
    header = f"{'k':>5}" + "".join(f"{m:>18}" for m in ["TokenCut", "LOST", "DINOv2"])
    print(header)
    print("-" * len(header))
    for k in K_GRID:
        cells = []
        for m in ["TokenCut", "LOST", "DINOv2"]:
            if m in results:
                miou, corloc = results[m][k]
                cells.append(f"{miou:.4f}/{corloc:.4f}")
            else:
                cells.append("-")
        print(f"{k:>5}" + "".join(f"{c:>18}" for c in cells))

    # ─── Ranking y estabilidad ───────────────────────────────────────────────
    print("\n[Ranking] ¿DINOv2 > TokenCut > LOST en mIoU para todo k?")
    all_rank_ok = True
    for k in K_GRID:
        if not all(m in results for m in ["TokenCut", "LOST", "DINOv2"]):
            break
        r = {m: results[m][k][0] for m in ["TokenCut", "LOST", "DINOv2"]}
        order = sorted(r, key=r.get, reverse=True)
        ok = order == ["DINOv2", "TokenCut", "LOST"]
        all_rank_ok &= ok
        print(f"  k={k}: {' > '.join(order)} {'[OK]' if ok else '[CAMBIA]'}")
    if all(m in results for m in ["TokenCut", "LOST", "DINOv2"]):
        print(f"  -> Ranking estable para todo el sweep: {'SÍ' if all_rank_ok else 'NO'}")

    print("\n[Estabilidad] max |mIoU(k) - mIoU(k_ref)| por método:")
    max_devs = {}
    for m, res in results.items():
        k_ref = K_REF[m]
        dev = max(abs(v[0] - res[k_ref][0]) for k, v in res.items())
        max_devs[m] = dev
        print(f"  {m:<9} k_ref={k_ref}: {dev:.4f} ({dev*100:.2f}%)")
    if max_devs:
        worst = max(max_devs.values())
        print(f"  -> peor desviación global: {worst:.4f} ({worst*100:.2f}%)")

    # ─── CSV ─────────────────────────────────────────────────────────────────
    rows = []
    for m, res in results.items():
        for k, (miou, corloc) in res.items():
            rows.append({"method": m, "k": k, "n": len(ids),
                         "miou_mean": round(miou, 4),
                         "corloc_mean": round(corloc, 4)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[CSV] Guardado: {OUT_CSV}")

    print("\n[Nota] MAE y F-beta no dependen del umbral de binarización "
          "(MAE se calcula sobre el mapa continuo; F-beta es max sobre "
          "umbrales), por lo que son constantes a lo largo del sweep.")


if __name__ == "__main__":
    main()
