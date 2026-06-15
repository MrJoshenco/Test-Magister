"""
SIFT Explorer — Scale Invariant Feature Transform
Visualización extendida: keypoints + heatmap de densidad + histogramas.

Dependencias:
    pip install opencv-python matplotlib scipy
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.stats import gaussian_kde


# ──────────────────────────────────────────────
#  Diálogos Tk
# ──────────────────────────────────────────────

def seleccionar_imagen() -> str | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    ruta = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[
            ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
            ("Todos los archivos", "*.*"),
        ],
    )
    root.destroy()
    return ruta if ruta else None


def guardar_imagen(fig, ruta_original: str):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    nombre_base, _ = os.path.splitext(os.path.basename(ruta_original))
    ruta_out = filedialog.asksaveasfilename(
        title="Guardar visualización",
        initialfile=f"{nombre_base}_sift_analysis.png",
        initialdir=os.path.dirname(ruta_original),
        defaultextension=".png",
        filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Todos", "*.*")],
    )
    root.destroy()
    if ruta_out:
        fig.savefig(ruta_out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  ✓ Guardado en: {ruta_out}")
    else:
        print("  — Guardado cancelado.")


# ──────────────────────────────────────────────
#  Procesamiento SIFT
# ──────────────────────────────────────────────

def cargar_imagen(ruta: str):
    bgr = cv2.imread(ruta)
    if bgr is None:
        raise ValueError(f"No se pudo leer: {ruta}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return gray, rgb


def detectar_keypoints(gray: np.ndarray):
    sift = cv2.SIFT_create(nfeatures=0)
    kp, des = sift.detectAndCompute(gray, None)
    kp = sorted(kp, key=lambda k: k.response, reverse=True)
    return kp, des


def dibujar_keypoints_mpl(ax, rgb: np.ndarray, kp: list):
    """
    Dibuja keypoints sobre la imagen usando matplotlib:
    círculo con radio = escala, color por respuesta (colormap plasma),
    línea de orientación, transparencia.
    """
    ax.imshow(rgb)
    if not kp:
        return

    resps = np.array([k.response for k in kp])
    norm  = mcolors.Normalize(vmin=resps.min(), vmax=resps.max())
    cmap  = plt.cm.plasma

    for k in kp:
        cx, cy = k.pt
        radio  = k.size / 2.0
        angulo = np.deg2rad(k.angle)
        color  = cmap(norm(k.response))

        circ = plt.Circle((cx, cy), radio,
                           fill=False, color=color, linewidth=0.6, alpha=0.75)
        ax.add_patch(circ)

        dx = radio * np.cos(angulo)
        dy = radio * np.sin(angulo)
        ax.plot([cx, cx + dx], [cy, cy + dy],
                color=color, linewidth=0.6, alpha=0.75)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Respuesta", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

    ax.set_title(
        f"Keypoints SIFT  ({len(kp)} detectados)\n"
        "Color = respuesta · Radio = escala",
        color="white", fontsize=10, pad=6,
    )
    ax.axis("off")


# ──────────────────────────────────────────────
#  Heatmap de densidad KDE
# ──────────────────────────────────────────────

def _cmap_transparente(base_cmap_name: str) -> LinearSegmentedColormap:
    """Colormap donde alpha=0 en valores bajos y sube progresivamente."""
    base      = plt.cm.get_cmap(base_cmap_name)
    colors_arr = base(np.linspace(0, 1, 256))
    colors_arr[:, 3] = np.linspace(0, 1, 256) ** 0.6
    return LinearSegmentedColormap.from_list("_transp", colors_arr)


def panel_heatmap(ax, rgb: np.ndarray, kp: list):
    ax.imshow(rgb, alpha=0.55)
    if len(kp) < 3:
        ax.set_title("Heatmap (pocos puntos)", color="white", fontsize=10)
        ax.axis("off")
        return

    h, w = rgb.shape[:2]
    xs = np.array([k.pt[0] for k in kp])
    ys = np.array([k.pt[1] for k in kp])

    res = 300
    xi  = np.linspace(0, w, res)
    yi  = np.linspace(0, h, res)
    Xi, Yi = np.meshgrid(xi, yi)

    kde = gaussian_kde(np.vstack([xs, ys]))
    Zi  = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(res, res)

    cmap_t = _cmap_transparente("inferno")
    ax.imshow(Zi, extent=[0, w, h, 0], origin="upper",
              cmap=cmap_t, aspect="auto", interpolation="bilinear")

    ax.contour(Xi, Yi, Zi, levels=6,
               colors="white", linewidths=0.4, alpha=0.35)

    ax.set_title(
        "Heatmap de densidad KDE\n(concentración espacial de keypoints)",
        color="white", fontsize=10, pad=6,
    )
    ax.axis("off")


# ──────────────────────────────────────────────
#  Histogramas
# ──────────────────────────────────────────────

def panel_histogramas(ax_esc, ax_resp, kp: list):
    scales = np.array([k.size     for k in kp])
    resps  = np.array([k.response for k in kp])

    BG = "#1e1e2e"
    C1 = "#7ec8e3"   # azul claro → escalas
    C2 = "#f0a500"   # ámbar     → respuestas

    for ax in (ax_esc, ax_resp):
        ax.set_facecolor(BG)
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # ── Escala ──
    bins = min(40, max(10, len(kp) // 20))
    ax_esc.hist(scales, bins=bins, color=C1, alpha=0.85, edgecolor="none")
    ax_esc.axvline(scales.mean(), color="white", linewidth=1,
                   linestyle="--", label=f"media = {scales.mean():.1f} px")
    p25, p75 = np.percentile(scales, [25, 75])
    ax_esc.axvspan(p25, p75, alpha=0.15, color=C1)
    ax_esc.set_xlabel("Escala (diámetro, px)", color="white", fontsize=9)
    ax_esc.set_ylabel("Frecuencia",            color="white", fontsize=9)
    ax_esc.set_title("Distribución de escalas",color="white", fontsize=10, pad=4)
    ax_esc.legend(fontsize=8, labelcolor="white", framealpha=0, handlelength=1.2)

    # Texto con estadísticas clave
    stats_txt = (f"min {scales.min():.1f}  ·  "
                 f"med {np.median(scales):.1f}  ·  "
                 f"max {scales.max():.1f}")
    ax_esc.text(0.98, 0.97, stats_txt, transform=ax_esc.transAxes,
                ha="right", va="top", fontsize=7.5, color="#aaaaaa")

    # ── Respuesta ──
    ax_resp.hist(resps, bins=bins, color=C2, alpha=0.85, edgecolor="none")
    ax_resp.axvline(resps.mean(), color="white", linewidth=1,
                    linestyle="--", label=f"media = {resps.mean():.4f}")
    p25r, p75r = np.percentile(resps, [25, 75])
    ax_resp.axvspan(p25r, p75r, alpha=0.15, color=C2)
    ax_resp.set_xlabel("Respuesta (contraste local)", color="white", fontsize=9)
    ax_resp.set_ylabel("Frecuencia",                  color="white", fontsize=9)
    ax_resp.set_title("Distribución de respuestas",   color="white", fontsize=10, pad=4)
    ax_resp.legend(fontsize=8, labelcolor="white", framealpha=0, handlelength=1.2)

    stats_txt2 = (f"min {resps.min():.4f}  ·  "
                  f"med {np.median(resps):.4f}  ·  "
                  f"max {resps.max():.4f}")
    ax_resp.text(0.98, 0.97, stats_txt2, transform=ax_resp.transAxes,
                 ha="right", va="top", fontsize=7.5, color="#aaaaaa")


# ──────────────────────────────────────────────
#  Figura principal
# ──────────────────────────────────────────────

def mostrar_todo(rgb: np.ndarray, kp: list, ruta: str) -> str:
    """
    Layout 2×2:
      [0,0] Original        [0,1] Keypoints matplotlib (color por respuesta)
      [1,0] Heatmap KDE     [1,1] Histograma escala | Histograma respuesta
    """
    BG     = "#111118"
    nombre = os.path.basename(ruta)
    h, w   = rgb.shape[:2]

    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.suptitle(
        f"SIFT Explorer  ·  {nombre}  ·  {w}×{h} px  ·  {len(kp)} keypoints",
        color="white", fontsize=13, fontweight="bold", y=0.98,
    )

    outer = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.04, right=0.97, top=0.93, bottom=0.10,
        hspace=0.30, wspace=0.18,
    )

    ax_orig = fig.add_subplot(outer[0, 0])
    ax_kp   = fig.add_subplot(outer[0, 1])
    ax_heat = fig.add_subplot(outer[1, 0])

    inner   = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1, 1], wspace=0.35)
    ax_esc  = fig.add_subplot(inner[0, 0])
    ax_resp = fig.add_subplot(inner[0, 1])

    for ax in (ax_orig, ax_kp, ax_heat):
        ax.set_facecolor(BG)

    # Paneles
    ax_orig.imshow(rgb)
    ax_orig.set_title("Imagen original", color="white", fontsize=10, pad=6)
    ax_orig.axis("off")

    dibujar_keypoints_mpl(ax_kp, rgb, kp)
    panel_heatmap(ax_heat, rgb, kp)
    panel_histogramas(ax_esc, ax_resp, kp)

    # ── Botones ──
    resultado = {"accion": None}

    ax_bg = fig.add_axes([0.36, 0.01, 0.13, 0.042])
    ax_br = fig.add_axes([0.51, 0.01, 0.13, 0.042])
    ax_bs = fig.add_axes([0.66, 0.01, 0.10, 0.042])

    btn_g = Button(ax_bg, "💾  Guardar",      color="#1a3a2a", hovercolor="#2a5a40")
    btn_r = Button(ax_br, "🔄  Nueva imagen", color="#1a2a3a", hovercolor="#2a4a6a")
    btn_s = Button(ax_bs, "✕  Salir",         color="#3a1a1a", hovercolor="#6a2a2a")

    for b in (btn_g, btn_r, btn_s):
        b.label.set_color("white")
        b.label.set_fontsize(10)

    def on_g(e):
        resultado["accion"] = "guardar"
        guardar_imagen(fig, ruta)

    def on_r(e):
        resultado["accion"] = "repetir"
        plt.close(fig)

    def on_s(e):
        resultado["accion"] = "salir"
        plt.close(fig)

    btn_g.on_clicked(on_g)
    btn_r.on_clicked(on_r)
    btn_s.on_clicked(on_s)

    plt.show()
    return resultado["accion"]


# ──────────────────────────────────────────────
#  Main loop
# ──────────────────────────────────────────────

def main():
    print("=" * 54)
    print("  SIFT Explorer  ·  heatmap + histogramas")
    print("=" * 54)

    while True:
        print("\n→ Selecciona una imagen en el diálogo…")
        ruta = seleccionar_imagen()

        if not ruta:
            print("  Sin selección. Saliendo.")
            break

        print(f"  Imagen: {ruta}")
        try:
            gray, rgb = cargar_imagen(ruta)
        except ValueError as e:
            print(f"  Error: {e}")
            continue

        h, w = gray.shape
        print(f"  Dimensiones: {w} × {h} px")
        print("  Ejecutando SIFT…", end=" ", flush=True)
        kp, _ = detectar_keypoints(gray)
        print(f"{len(kp)} keypoints detectados.")
        print("  Calculando KDE (puede tardar unos segundos en imágenes grandes)…")

        accion = mostrar_todo(rgb, kp, ruta)

        if accion == "repetir":
            continue
        elif accion == "guardar":
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            continuar = messagebox.askyesno("Continuar", "¿Procesar otra imagen?")
            root.destroy()
            if not continuar:
                break
        else:
            break

    print("\nSIFT Explorer cerrado.")


if __name__ == "__main__":
    main()