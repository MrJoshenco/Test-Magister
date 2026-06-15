"""
Mapa de Saliencia - Script interactivo
Implementa múltiples algoritmos de saliencia visual y los visualiza.

Algoritmos disponibles:
  1. Espectral Residual (SR)     - Clasico de Hou & Zhang (2007)
  2. Fine-Grained Saliency (FGS) - Basado en gradientes de alta frecuencia
  3. GBVS-like (Graph-Based)     - Aproximacion basada en contraste local

Dependencias: pip install numpy opencv-python scipy matplotlib Pillow
"""

import os
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from tkinter import Tk, filedialog, messagebox


# ══════════════════════════════════════════════════════════
#  CONFIGURACION  ← edita estas rutas segun tu entorno
# ══════════════════════════════════════════════════════════

# Carpeta inicial al abrir el selector de imagen.
# None  → usa el directorio actual del sistema.
# Ejemplo: IMAGES_DIR = "/home/joshe/Pictures/Birds"
IMAGES_DIR = "/home/joshe/Pictures/Pajaritos wallpapers/Pajaritos"

# Carpeta inicial al guardar resultados.
# None  → usa el mismo directorio de la imagen seleccionada.
# Ejemplo: OUTPUT_DIR = "/home/joshe/Projects/Saliency/results"
OUTPUT_DIR = "/home/joshe/Projects/Test Magister/Salency Maps/results"

# Resolucion maxima de procesamiento (lado mayor en px).
# Sube para mayor detalle, baja para mayor velocidad.
MAX_DIM: int = 900

# ══════════════════════════════════════════════════════════


# ──────────────────────────────────────────────
# ALGORITMOS DE SALIENCIA
# ──────────────────────────────────────────────

def saliency_spectral_residual(img_bgr):
    """
    Spectral Residual (Hou & Zhang, 2007).
    Opera en el espacio frecuencial: el residuo espectral captura
    zonas estadisticamente inusuales (=salientes).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fft      = np.fft.fft2(gray)
    log_amp  = np.log(np.abs(fft) + 1e-8)
    phase    = np.angle(fft)
    smoothed = gaussian_filter(log_amp, sigma=3)
    residual = np.exp(log_amp - smoothed)
    recon    = np.fft.ifft2(residual * np.exp(1j * phase))
    sal      = gaussian_filter(np.abs(recon) ** 2, sigma=8)
    return _normalize(sal)


def saliency_fine_grained(img_bgr):
    """
    Fine-Grained Saliency basado en gradientes multi-escala.
    Resalta bordes y texturas inusuales.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sal  = np.zeros_like(gray)
    for sigma in [1, 2, 4, 8]:
        blur = gaussian_filter(gray, sigma=sigma)
        gx   = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        gy   = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        sal += _normalize(np.sqrt(gx**2 + gy**2))
    return _normalize(sal)


def saliency_gbvs_approx(img_bgr):
    """
    Aproximacion GBVS: contraste de color en L*a*b* respecto al
    promedio global, en multiples escalas espaciales.
    """
    lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    mean = lab.mean(axis=(0, 1))
    sal  = np.zeros(img_bgr.shape[:2], dtype=np.float32)
    for sigma in [4, 8, 16, 32]:
        blurred = gaussian_filter(lab, sigma=(sigma, sigma, 0))
        sal    += _normalize(np.linalg.norm(blurred - mean, axis=2))
    return _normalize(gaussian_filter(sal, sigma=10))


def _normalize(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


# ──────────────────────────────────────────────
# VISUALIZACION
# ──────────────────────────────────────────────

def _cv_cmap(name):
    return {
        "hot":     cv2.COLORMAP_HOT,
        "jet":     cv2.COLORMAP_JET,
        "inferno": cv2.COLORMAP_INFERNO,
    }.get(name, cv2.COLORMAP_HOT)


def overlay_heatmap(img_bgr, sal_norm, cmap_name="hot", alpha=0.55):
    """Mezcla la imagen original con el mapa de calor."""
    sal8  = (sal_norm * 255).astype(np.uint8)
    heat  = cv2.applyColorMap(sal8, _cv_cmap(cmap_name))
    img_r = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_r, 1 - alpha, heat[:, :, ::-1], alpha, 0)


def show_results(img_bgr, maps: dict, image_path: str):
    """
    Panel completo: imagen original, 3 mapas, overlays, fusion y ROI.
    Botones: [Guardar resultados]  [Nueva imagen]
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    names   = list(maps.keys())
    fname   = os.path.basename(image_path)

    fig = plt.figure(figsize=(16, 9), facecolor="#0f0f0f")
    fig.canvas.manager.set_window_title("Mapa de Saliencia")

    gs = gridspec.GridSpec(
        3, 4, figure=fig,
        hspace=0.35, wspace=0.08,
        left=0.03, right=0.97, top=0.90, bottom=0.12,
    )
    title_kw  = dict(color="white", fontsize=9, fontweight="bold", pad=4)
    cmap_name = "hot"

    # Fila 0, col 0 — imagen original
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(img_rgb)
    ax_orig.set_title(f"Original\n{fname}", **title_kw)
    ax_orig.axis("off")

    # Filas 0-1, cols 1-3 — mapas y overlays por algoritmo
    for col, name in enumerate(names, start=1):
        sal = maps[name]

        ax_s = fig.add_subplot(gs[0, col])
        ax_s.imshow(sal, cmap=cmap_name, vmin=0, vmax=1)
        ax_s.set_title(name, **title_kw)
        ax_s.axis("off")

        ax_o = fig.add_subplot(gs[1, col])
        ax_o.imshow(overlay_heatmap(img_bgr, sal, cmap_name))
        ax_o.set_title("Overlay", **title_kw)
        ax_o.axis("off")

    # Fila 1, col 0 — barra de color
    ax_cb = fig.add_subplot(gs[1, 0])
    sm = plt.cm.ScalarMappable(cmap=cmap_name)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax_cb, fraction=0.9, pad=0.05)
    cb.set_label("Saliencia", color="white", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=7)
    ax_cb.axis("off")

    # Fila 2 — fusion, overlay fusion, ROI binarizada
    fused  = _normalize(np.mean(list(maps.values()), axis=0))
    fused8 = (fused * 255).astype(np.uint8)
    _, mask = cv2.threshold(fused8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masked  = img_rgb.copy()
    masked[mask == 0] = (masked[mask == 0] * 0.25).astype(np.uint8)

    ax_f0 = fig.add_subplot(gs[2, 0])
    ax_f0.imshow(fused, cmap=cmap_name)
    ax_f0.set_title("Fusion (promedio)", **title_kw)
    ax_f0.axis("off")

    ax_f1 = fig.add_subplot(gs[2, 1:3])
    ax_f1.imshow(overlay_heatmap(img_bgr, fused, cmap_name, alpha=0.5))
    ax_f1.set_title("Overlay - Fusion", **title_kw)
    ax_f1.axis("off")

    ax_f2 = fig.add_subplot(gs[2, 3])
    ax_f2.imshow(masked)
    ax_f2.set_title("Region de Interes (ROI)", **title_kw)
    ax_f2.axis("off")

    # Titulo principal
    fig.text(0.5, 0.96, "Analisis de Saliencia Visual",
             ha="center", va="top", color="white",
             fontsize=14, fontweight="bold")

    # ── Botones (sin emojis para evitar warnings de glifos) ──
    state = {"action": None}

    ax_btn_save = fig.add_axes([0.30, 0.03, 0.18, 0.05])
    ax_btn_new  = fig.add_axes([0.52, 0.03, 0.18, 0.05])

    btn_save = Button(ax_btn_save, "[S]  Guardar resultados",
                      color="#1a6b3c", hovercolor="#27a85f")
    btn_new  = Button(ax_btn_new,  "[N]  Nueva imagen",
                      color="#1a3d6b", hovercolor="#2760a8")

    for btn in (btn_save, btn_new):
        btn.label.set_color("white")
        btn.label.set_fontsize(9)

    def _resolve_save_dir():
        """Devuelve la carpeta inicial para el dialogo de guardado."""
        if OUTPUT_DIR and os.path.isdir(OUTPUT_DIR):
            return OUTPUT_DIR
        # fallback: mismo directorio de la imagen
        return os.path.dirname(image_path)

    def on_save(_):
        save_dir  = _resolve_save_dir()
        stem      = os.path.splitext(fname)[0]
        init_file = f"saliencia_{stem}.png"

        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        out = filedialog.asksaveasfilename(
            title="Guardar mapa de saliencia",
            initialdir=save_dir,
            initialfile=init_file,
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
        )
        root.destroy()

        if out:
            fig.savefig(out, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"  Guardado en: {out}")
            root2 = Tk(); root2.withdraw()
            messagebox.showinfo("Guardado", f"Imagen guardada en:\n{out}")
            root2.destroy()

    def on_new(_):
        state["action"] = "new"
        plt.close(fig)

    btn_save.on_clicked(on_save)
    btn_new.on_clicked(on_new)

    plt.show()
    return state["action"]


# ──────────────────────────────────────────────
# FLUJO PRINCIPAL
# ──────────────────────────────────────────────

def _resolve_images_dir():
    """Devuelve la carpeta inicial para el selector de imagen."""
    if IMAGES_DIR and os.path.isdir(IMAGES_DIR):
        return IMAGES_DIR
    return os.path.expanduser("~")


def pick_image():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Selecciona una imagen",
        initialdir=_resolve_images_dir(),
        filetypes=[
            ("Imagenes", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
            ("JPEG",     "*.jpg *.jpeg"),
            ("PNG",      "*.png"),
            ("Todos",    "*.*"),
        ]
    )
    root.destroy()
    return path


def process_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")

    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img   = cv2.resize(img, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
        print(f"  Redimensionada a {img.shape[1]}x{img.shape[0]} px")

    print("  Calculando Spectral Residual...")
    sr   = saliency_spectral_residual(img)
    print("  Calculando Fine-Grained...")
    fgs  = saliency_fine_grained(img)
    print("  Calculando GBVS aproximado...")
    gbvs = saliency_gbvs_approx(img)

    return img, {
        "Spectral Residual": sr,
        "Fine-Grained":      fgs,
        "GBVS (approx.)":    gbvs,
    }


def main():
    print("=" * 52)
    print("  Analisis de Saliencia Visual")
    print("=" * 52)

    # Mostrar configuracion activa
    print(f"\n  Carpeta imagenes : {IMAGES_DIR or '(directorio home)'}")
    print(f"  Carpeta salida   : {OUTPUT_DIR or '(misma carpeta de la imagen)'}")

    while True:
        print("\nSelecciona una imagen desde el explorador...")
        path = pick_image()

        if not path:
            print("No se selecciono ninguna imagen. Saliendo.")
            break

        print(f"\nProcesando: {os.path.basename(path)}")
        try:
            img, maps = process_image(path)
        except Exception as e:
            root = Tk(); root.withdraw()
            messagebox.showerror("Error", str(e))
            root.destroy()
            continue

        print("  Mostrando resultados...")
        action = show_results(img, maps, path)

        if action != "new":
            print("Session finalizada.")
            break

        print("\nNueva imagen solicitada...")


if __name__ == "__main__":
    main()