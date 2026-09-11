#!/usr/bin/env python3
"""
wilcoxon_tests.py

Tests estadísticos pareados (Wilcoxon signed-rank) de DINOv2 contra LOST y
TokenCut, para las 5 métricas, sobre las 70 imágenes de evaluación. Incluye
corrección de Holm sobre los 10 p-values.

Notas:
  - Las diferencias con empate exacto (d == 0) se descartan, como en el
    protocolo estándar de Wilcoxon.
  - Para MAE/mIoU el test es sobre "menor es mejor"; para F03/F1/CorLoc sobre
    "mayor es mejor". El signo de la diferencia no afecta al p-value bilateral,
    pero se reporta la mediana de las diferencias para interpretar la dirección.

Salida: tabla por consola (p crudo, p Holm, significativo al 0.05).

Uso:
    .venv/bin/python wilcoxon_tests.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).parent.resolve()
IN_CSV = PROJECT_ROOT / "results_per_image.csv"

METRICS = ["mae", "miou", "f03", "f1", "corloc"]
METRIC_LABEL = {"mae": "MAE", "miou": "mIoU", "f03": "F0.3", "f1": "F1",
                "corloc": "CorLoc"}
# MAE: menor es mejor. mIoU/F0.3/F1/CorLoc: mayor es mejor.
LOWER_IS_BETTER = {"mae"}
BASELINES = ["LOST", "TokenCut"]


def holm_correction(pvals, alpha=0.05):
    """Corrección de Holm. Devuelve (p_adj, reject) en el orden original."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    adj_sorted = np.empty(m)
    running_max = 0.0
    for i in range(m):
        val = (m - i) * p_sorted[i]
        running_max = max(running_max, val)
        adj_sorted[i] = min(running_max, 1.0)
    reject_sorted = adj_sorted < alpha
    p_adj = np.empty(m)
    reject = np.empty(m, dtype=bool)
    p_adj[order] = adj_sorted
    reject[order] = reject_sorted
    return p_adj, reject


def main():
    df = pd.read_csv(IN_CSV)
    piv = df.pivot_table(index="image", columns=["method"],
                         values=METRICS)

    pairs, pvals, infos = [], [], []
    for metric in METRICS:
        for base in BASELINES:
            # DINOv2 - base. Positivo = DINOv2 mayor (mejor para
            # higher-is-better, peor para MAE/mIoU).
            d = piv[metric]["DINOv2"] - piv[metric][base]
            d_nz = d[d != 0]
            n_nz = len(d_nz)
            n_ties = int((d == 0).sum())
            if n_nz < 6:
                print(f"[AVISO] {metric} DINOv2 vs {base}: solo {n_nz} "
                      f"diferencias no nulas; test poco potente.")
            stat, p = wilcoxon(d_nz)
            median_d = float(d_nz.median())
            better = None
            if metric in LOWER_IS_BETTER:
                better = "DINOv2 mejor" if median_d < 0 else (
                    "baseline mejor" if median_d > 0 else "empate")
            else:
                better = "DINOv2 mejor" if median_d > 0 else (
                    "baseline mejor" if median_d < 0 else "empate")
            pairs.append(f"DINOv2 vs {base} ({METRIC_LABEL[metric]})")
            pvals.append(p)
            infos.append((n_nz, n_ties, median_d, better))

    p_adj, reject = holm_correction(pvals)

    print("\nTests de Wilcoxon pareados (DINOv2 vs baselines), n=70 imágenes")
    print("Corrección de Holm sobre 10 tests, alpha=0.05")
    print("-" * 86)
    print(f"{'Comparación':<28} {'n≠0':>4} {'empates':>7} {'mediana Δ':>10} "
          f"{'p crudo':>9} {'p Holm':>9} {'sig.':>5}  dirección")
    print("-" * 86)
    for pair, p, pa, r, (n_nz, n_ties, med, better) in zip(
            pairs, pvals, p_adj, reject, infos):
        print(f"{pair:<28} {n_nz:>4} {n_ties:>7} {med:>10.4f} "
              f"{p:>9.4f} {pa:>9.4f} {'SÍ' if r else 'no':>5}  {better}")

    # Resumen para redacción de §V-A
    print("\nResumen para redacción:")
    sig = [pair for pair, r in zip(pairs, reject) if r]
    print(f"  Significativos tras Holm ({len(sig)}/10):")
    for s in sig:
        print(f"    - {s}")


if __name__ == "__main__":
    main()
