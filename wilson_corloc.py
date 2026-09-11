#!/usr/bin/env python3
"""
wilson_corloc.py

Calcula intervalos de confianza de Wilson (95%) para CorLoc por categoría y
método, a partir de results_per_image.csv (generado por eval_per_image.py).

Validaciones incluidas:
  - k/n (proporción de aciertos) se imprime junto al IC para cruzarlo con la
    columna CorLoc (p̂) de la Table IV del paper.
  - Casos p̂=0: Wilson da [0, z²/(n+z²)] con el n real de la categoría.
  - Ningún límite puede salir de [0, 1] (clip + chequeo).

Salida:
  - wilson_corloc_by_category.csv
  - Tabla en formato listo para pegar en LaTeX.

Uso:
    .venv/bin/python wilson_corloc.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

PROJECT_ROOT = Path(__file__).parent.resolve()
IN_CSV = PROJECT_ROOT / "results_per_image.csv"
OUT_CSV = PROJECT_ROOT / "wilson_corloc_by_category.csv"

CATEGORY_ORDER = ["C1", "C2", "C3", "C4"]
METHOD_ORDER = ["DINOv2", "TokenCut", "LOST"]


def main():
    df = pd.read_csv(IN_CSV)
    if not set(["category", "method", "corloc"]).issubset(df.columns):
        raise SystemExit(f"[ERROR] {IN_CSV} no tiene las columnas esperadas.")

    rows = []
    for cat in CATEGORY_ORDER:
        for method in METHOD_ORDER:
            g = df[(df["category"] == cat) & (df["method"] == method)]
            n = len(g)
            k = int(round(g["corloc"].sum()))
            if n == 0:
                continue
            lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
            lo, hi = float(np.clip(lo, 0.0, 1.0)), float(np.clip(hi, 0.0, 1.0))
            if lo < 0 or hi > 1:
                raise SystemExit(f"[ERROR] IC fuera de rango: {cat}/{method} [{lo}, {hi}]")
            rows.append({
                "category": cat, "method": method, "n": n, "k": k,
                "p_hat": round(k / n, 4),
                "wilson_lo": round(lo, 4), "wilson_hi": round(hi, 4),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"[CSV] Guardado: {OUT_CSV}\n")

    # Tabla lista para LaTeX (Table IV, columna CorLoc)
    print("CorLoc por categoría (p̂ = k/n, IC 95% Wilson):")
    print("-" * 72)
    for cat in CATEGORY_ORDER:
        print(f"{cat}:")
        for method in METHOD_ORDER:
            r = out[(out["category"] == cat) & (out["method"] == method)].iloc[0]
            print(f"  {method:<9} n={r['n']:>2}  k={r['k']:>2}  "
                  f"p̂={r['p_hat']:.4f}  [{r['wilson_lo']:.4f}, {r['wilson_hi']:.4f}]"
                  f"  -> LaTeX: $[{r['wilson_lo']:.4f},\\,{r['wilson_hi']:.4f}]$")
        print()

    # Casos especiales p̂=0
    zeros = out[out["k"] == 0]
    if len(zeros):
        print("Casos p̂=0 (límite superior = z²/(n+z²)):")
        for _, r in zeros.iterrows():
            print(f"  {r['category']}/{r['method']}: n={r['n']} -> "
                  f"[0.0000, {r['wilson_hi']:.4f}]")


if __name__ == "__main__":
    main()
