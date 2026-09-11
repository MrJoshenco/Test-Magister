#!/usr/bin/env python3
"""
latency_mp.py

Calcula los megapíxeles reales procesados por cada método sobre las 70
imágenes de evaluación (split R=14) y la latencia normalizada (s/MP), usando
los tiempos medios de la corrida canónica (outputs_all_methods_holdout/report.txt:
TokenCut 0.7339 s, LOST 0.0748 s, DINOv2 0.0739 s).

- Baselines (TokenCut/LOST): procesan a resolución nativa de entrada.
- DINOv2: replica resize_for_dinov2 (lado mayor ≤ 518, crop central a múltiplo
  de 14). No es 518×518 fijo: mantiene aspect ratio.

Salida: tabla por consola con MP medio, tiempo y s/MP por método.

Uso:
    .venv/bin/python latency_mp.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import DINOv2  # noqa: F401  (garantiza el mismo entorno; no se usa directamente)

INPUT_DIR = PROJECT_ROOT / "Dataset_tesis_numerado_ligero"

RANDOM_SEED = 14
N_TUNING = 30
N_TOTAL = 100

# Tiempos medios por imagen (outputs_all_methods_holdout/report.txt, IC 95%)
TIMES = {"TokenCut": 0.7339, "LOST": 0.0748, "DINOv2": 0.0739}

DINOV2_MAX_RESOLUTION = 518   # config de la corrida canónica
DINOV2_PATCH_SIZE = 14


def eval_ids():
    all_ids = list(range(1, N_TOTAL + 1))
    rng = np.random.default_rng(RANDOM_SEED)
    t_ids = set(int(x) for x in rng.choice(all_ids, size=N_TUNING, replace=False))
    return [i for i in all_ids if i not in t_ids]


def dinov2_internal_size(img):
    """Replica DINOv2.resize_for_dinov2 y devuelve (W, H) procesados."""
    w, h = img.size
    if max(w, h) > DINOV2_MAX_RESOLUTION:
        scale = DINOV2_MAX_RESOLUTION / max(w, h)
        w, h = int(w * scale), int(h * scale)
        img = img.resize((w, h), Image.Resampling.LANCZOS)
    w, h = img.size
    new_w = (w // DINOV2_PATCH_SIZE) * DINOV2_PATCH_SIZE
    new_h = (h // DINOV2_PATCH_SIZE) * DINOV2_PATCH_SIZE
    return new_w, new_h


def main():
    ids = eval_ids()
    mp_in, mp_dino = [], []
    for img_id in ids:
        with Image.open(INPUT_DIR / f"{img_id:03d}.jpg") as img:
            w, h = img.size
            mp_in.append(w * h / 1e6)
            iw, ih = dinov2_internal_size(img.copy())
            mp_dino.append(iw * ih / 1e6)

    mp_base = float(np.mean(mp_in))
    mp_ours = float(np.mean(mp_dino))

    print(f"Imágenes de evaluación: {len(ids)}")
    print(f"MP de entrada (baselines):   medio={mp_base:.4f}  "
          f"mín={min(mp_in):.4f}  máx={max(mp_in):.4f}")
    print(f"MP interno DINOv2 (≤518 lado mayor, crop ×14): "
          f"medio={mp_ours:.4f}  mín={min(mp_dino):.4f}  máx={max(mp_dino):.4f}")
    print(f"  -> DINOv2 procesa ~{mp_ours/mp_base:.2f}x menos píxeles que los "
          f"baselines (reducción del {(1 - mp_ours/mp_base)*100:.1f}%)\n")

    print(f"{'Método':<10} {'t (s/img)':>10} {'MP':>8} {'s/MP':>10}")
    print("-" * 42)
    results = {}
    for method in ["DINOv2", "LOST", "TokenCut"]:
        mp = mp_ours if method == "DINOv2" else mp_base
        smp = TIMES[method] / mp
        results[method] = (TIMES[method], mp, smp)
        print(f"{method:<10} {TIMES[method]:>10.4f} {mp:>8.4f} {smp:>10.4f}")

    best = min(results, key=lambda m: results[m][2])
    print("-" * 42)
    print(f"Método más eficiente por unidad de área (s/MP): {best} "
          f"({results[best][2]:.4f} s/MP)")
    print("\nNota: aunque DINOv2 reduce la resolución interna, su s/MP es "
          "mayor que el de LOST; la frase 'LOST is the most efficient per unit "
          "of processed area' se mantiene con los valores reales.")


if __name__ == "__main__":
    main()
