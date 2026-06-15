"""
SIFT Explorer — Scale Invariant Feature Transform
Selecciona una imagen JPG, visualiza sus keypoints y guarda el resultado.

Dependencias:
    pip install opencv-python matplotlib
    (o: pip install opencv-contrib-python matplotlib  si quieres SIFT patentado legacy)
"""

import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ──────────────────────────────────────────────
#  Utilidades
# ──────────────────────────────────────────────

def seleccionar_imagen() -> str | None:
    """Abre un diálogo para elegir una imagen. Retorna la ruta o None."""
    root = tk.Tk()
    root.withdraw()          # Oculta la ventana principal de Tk
    root.attributes("-topmost", True)

    ruta = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[
            ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("Todos los archivos", "*.*"),
        ],
    )
    root.destroy()
    return ruta if ruta else None


def cargar_imagen(ruta: str):
    """Carga la imagen en BGR y devuelve (bgr, gray, rgb)."""
    bgr = cv2.imread(ruta)
    if bgr is None:
        raise ValueError(f"No se pudo leer la imagen: {ruta}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return bgr, gray, rgb


def detectar_keypoints(gray: np.ndarray, n_features: int = 0):
    """
    Detecta keypoints con SIFT.
    n_features=0 → sin límite superior.
    Devuelve (keypoints, descriptors).
    """
    sift = cv2.SIFT_create(nfeatures=n_features)
    kp, des = sift.detectAndCompute(gray, None)
    # Ordenar por respuesta (más relevante primero)
    kp = sorted(kp, key=lambda k: k.response, reverse=True)
    return kp, des


def dibujar_keypoints(rgb: np.ndarray, kp: list) -> np.ndarray:
    """
    Dibuja keypoints con círculo (escala) y línea de orientación
    sobre una copia de la imagen RGB.
    """
    img_out = rgb.copy()
    bgr_tmp = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

    # Dibuja con flags DRAW_RICH_KEYPOINTS (círculo + orientación)
    bgr_drawn = cv2.drawKeypoints(
        bgr_tmp,
        kp,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return cv2.cvtColor(bgr_drawn, cv2.COLOR_BGR2RGB)


def estadisticas(kp: list) -> dict:
    """Resumen numérico de los keypoints detectados."""
    if not kp:
        return {"total": 0}
    scales  = [k.size for k in kp]
    resps   = [k.response for k in kp]
    return {
        "total":       len(kp),
        "escala_min":  f"{min(scales):.1f}",
        "escala_max":  f"{max(scales):.1f}",
        "escala_prom": f"{np.mean(scales):.1f}",
        "resp_max":    f"{max(resps):.4f}",
        "resp_prom":   f"{np.mean(resps):.4f}",
    }


# ──────────────────────────────────────────────
#  Visualización
# ──────────────────────────────────────────────

def mostrar_resultados(rgb_orig: np.ndarray,
                       rgb_sift: np.ndarray,
                       kp: list,
                       ruta: str) -> bool:
    """
    Muestra la imagen original y la imagen con keypoints lado a lado.
    Retorna True si el usuario pide guardar.
    """
    stats = estadisticas(kp)
    nombre_archivo = os.path.basename(ruta)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a1a")

    # ── Panel izquierdo: original ──
    axes[0].imshow(rgb_orig)
    axes[0].set_title("Imagen original", color="white", fontsize=13, pad=10)
    axes[0].axis("off")

    # ── Panel derecho: SIFT ──
    axes[1].imshow(rgb_sift)
    axes[1].set_title(
        f"Keypoints SIFT detectados: {stats['total']}",
        color="#7ec8e3", fontsize=13, pad=10,
    )
    axes[1].axis("off")

    # ── Título general ──
    fig.suptitle(
        f"SIFT Explorer  ·  {nombre_archivo}",
        color="white", fontsize=15, fontweight="bold", y=1.01,
    )

    # ── Caja de estadísticas ──
    info = (
        f"Total keypoints : {stats['total']}\n"
        f"Escala  min / máx / prom : "
        f"{stats.get('escala_min','—')} / "
        f"{stats.get('escala_max','—')} / "
        f"{stats.get('escala_prom','—')} px\n"
        f"Respuesta  máx / prom : "
        f"{stats.get('resp_max','—')} / "
        f"{stats.get('resp_prom','—')}"
    )
    fig.text(
        0.5, -0.02, info,
        ha="center", va="top",
        fontsize=10, color="#cccccc",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#2a2a2a", edgecolor="#444"),
    )

    # ── Botones interactivos ──
    plt.subplots_adjust(bottom=0.12)
    ax_guardar  = fig.add_axes([0.35, 0.02, 0.13, 0.05])
    ax_repetir  = fig.add_axes([0.50, 0.02, 0.13, 0.05])
    ax_salir    = fig.add_axes([0.65, 0.02, 0.10, 0.05])

    from matplotlib.widgets import Button
    btn_guardar = Button(ax_guardar, "💾 Guardar", color="#264d40", hovercolor="#3a7060")
    btn_repetir = Button(ax_repetir, "🔄 Nueva imagen", color="#1e3a5f", hovercolor="#2e5a8f")
    btn_salir   = Button(ax_salir,   "✕ Salir",    color="#4d1a1a", hovercolor="#7a2a2a")

    for b in (btn_guardar, btn_repetir, btn_salir):
        b.label.set_color("white")
        b.label.set_fontsize(10)

    resultado = {"accion": None}

    def on_guardar(event):
        resultado["accion"] = "guardar"
        plt.close(fig)

    def on_repetir(event):
        resultado["accion"] = "repetir"
        plt.close(fig)

    def on_salir(event):
        resultado["accion"] = "salir"
        plt.close(fig)

    btn_guardar.on_clicked(on_guardar)
    btn_repetir.on_clicked(on_repetir)
    btn_salir.on_clicked(on_salir)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()

    return resultado["accion"], rgb_sift


def guardar_imagen(rgb_sift: np.ndarray, ruta_original: str):
    """Abre diálogo para guardar la imagen con keypoints."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    directorio = os.path.dirname(ruta_original)
    nombre_base, ext = os.path.splitext(os.path.basename(ruta_original))
    sugerido = os.path.join(directorio, f"{nombre_base}_sift.jpg")

    ruta_out = filedialog.asksaveasfilename(
        title="Guardar imagen con keypoints",
        initialfile=f"{nombre_base}_sift.jpg",
        initialdir=directorio,
        defaultextension=".jpg",
        filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("Todos", "*.*")],
    )
    root.destroy()

    if ruta_out:
        bgr_out = cv2.cvtColor(rgb_sift, cv2.COLOR_RGB2BGR)
        cv2.imwrite(ruta_out, bgr_out)
        print(f"  ✓ Imagen guardada en: {ruta_out}")
        return ruta_out
    else:
        print("  — Guardado cancelado.")
        return None


# ──────────────────────────────────────────────
#  Bucle principal
# ──────────────────────────────────────────────

def main():
    print("=" * 52)
    print("  SIFT Explorer — Scale Invariant Feature Transform")
    print("=" * 52)

    while True:
        # 1. Selección de imagen
        print("\n→ Selecciona una imagen en el diálogo…")
        ruta = seleccionar_imagen()

        if not ruta:
            print("  Sin selección. Saliendo.")
            break

        print(f"  Imagen: {ruta}")

        # 2. Carga
        try:
            bgr, gray, rgb = cargar_imagen(ruta)
        except ValueError as e:
            print(f"  Error: {e}")
            continue

        h, w = gray.shape
        print(f"  Dimensiones: {w} × {h} px")

        # 3. Detección SIFT
        print("  Ejecutando SIFT… ", end="", flush=True)
        kp, des = detectar_keypoints(gray)
        print(f"{len(kp)} keypoints encontrados.")

        # 4. Render
        rgb_sift = dibujar_keypoints(rgb, kp)

        # 5. Mostrar
        accion, rgb_sift_out = mostrar_resultados(rgb, rgb_sift, kp, ruta)

        # 6. Guardar si se solicitó
        if accion == "guardar":
            guardar_imagen(rgb_sift_out, ruta)
            # Después de guardar, preguntar si continuar
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            continuar = messagebox.askyesno(
                "Continuar", "¿Procesar otra imagen?", parent=root
            )
            root.destroy()
            if not continuar:
                break

        elif accion == "repetir":
            continue   # vuelve al inicio del while

        else:   # "salir" o cierre de ventana
            break

    print("\nSIFT Explorer cerrado. ¡Hasta pronto!")


if __name__ == "__main__":
    main()