import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import os
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN — edita solo esta sección
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = "/home/joshe/Projects/Test Magister/Inceptionv3/gradcam_outputs"  # carpeta de salida
ALPHA_OVERLAY = 0.5       # transparencia del heatmap (0.0 - 1.0)
SAVE_INDIVIDUAL = False   # guardar heatmap y overlay por separado además del grid
TOP_K_CLASSES = 3         # cuántas clases mostrar en el título

# ══════════════════════════════════════════════════════════════════════════════

# ─── Labels ImageNet (para nombres representativos) ──────────────────────────
def load_imagenet_labels():
    import urllib.request
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        with urllib.request.urlopen(url) as f:
            labels = [line.strip() for line in f.read().decode("utf-8").splitlines()]
        return labels
    except Exception:
        print("⚠ No se pudieron cargar labels de ImageNet, se usará el índice numérico.")
        return None

# ─── Preprocesamiento ────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─── Grad-CAM ────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x, class_idx=None):
        output = self.model(x)
        probs = torch.softmax(output, dim=1)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(299, 299), mode='bilinear', align_corners=False)
        cam = cam.squeeze().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Top-K probabilidades
        top_probs, top_indices = probs.topk(TOP_K_CLASSES, dim=1)
        top_probs = top_probs.squeeze().tolist()
        top_indices = top_indices.squeeze().tolist()

        return cam, class_idx, top_probs, top_indices

# ─── Overlay ─────────────────────────────────────────────────────────────────
def overlay_heatmap(image_pil, cam, alpha=ALPHA_OVERLAY):
    img = np.array(image_pil.resize((299, 299))).astype(np.uint8)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * img).astype(np.uint8)
    return overlay, heatmap

# ─── Nombre de archivo representativo ────────────────────────────────────────
def build_output_name(image_path, class_idx, labels, confidence):
    stem = Path(image_path).stem                          # nombre sin extensión
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = labels[class_idx].replace(" ", "_") if labels else f"class{class_idx}"
    conf_str = f"{confidence*100:.0f}pct"
    return f"{stem}__cls-{label}__conf-{conf_str}__{timestamp}"

# ─── Guardar resultados ───────────────────────────────────────────────────────
def save_results(image_path, image_pil, cam, overlay, heatmap_rgb,
                 class_idx, labels, top_probs, top_indices, output_dir):

    base_name = build_output_name(image_path, class_idx, labels, top_probs[0])
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    label_names = [labels[i].replace("_", " ") if labels else str(i) for i in top_indices]

    # ── Grid principal (3 paneles) ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0f0f0f")

    title_main = f"Top-1: {label_names[0]}  ({top_probs[0]*100:.1f}%)"
    fig.suptitle(title_main, color="white", fontsize=13, fontweight="bold", y=1.01)

    # Imagen original
    axes[0].imshow(image_pil.resize((299, 299)))
    axes[0].set_title("Original", color="#aaaaaa", fontsize=10)
    axes[0].axis("off")

    # Heatmap
    axes[1].imshow(cam, cmap="jet")
    subtitle = "  |  ".join([f"{n} {p*100:.0f}%" for n, p in zip(label_names, top_probs)])
    axes[1].set_title(f"Grad-CAM\n{subtitle}", color="#aaaaaa", fontsize=8)
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Superposición", color="#aaaaaa", fontsize=10)
    axes[2].axis("off")

    plt.tight_layout()
    grid_path = out_path / f"{base_name}__grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    if SAVE_INDIVIDUAL:
        # Heatmap solo
        hm_path = out_path / f"{base_name}__heatmap.png"
        plt.imsave(str(hm_path), cam, cmap="jet")

        # Overlay solo
        ov_path = out_path / f"{base_name}__overlay.png"
        Image.fromarray(overlay).save(str(ov_path))

    return grid_path

# ─── Selector de archivos ────────────────────────────────────────────────────
def select_images():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    paths = filedialog.askopenfilenames(
        title="Selecciona imágenes para Grad-CAM",
        filetypes=[
            ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
            ("Todos los archivos", "*.*"),
        ]
    )
    root.destroy()
    return list(paths)

# ─── Procesar una imagen ──────────────────────────────────────────────────────
def process_image(image_path, model, grad_cam, labels, output_dir):
    print(f"  → Procesando: {Path(image_path).name}")
    try:
        image = Image.open(image_path).convert("RGB")
        x = preprocess(image).unsqueeze(0)
        x.requires_grad_(True)

        cam, class_idx, top_probs, top_indices = grad_cam.generate(x)
        overlay, heatmap_rgb = overlay_heatmap(image, cam)

        saved_path = save_results(
            image_path, image, cam, overlay, heatmap_rgb,
            class_idx, labels, top_probs, top_indices, output_dir
        )

        label = labels[class_idx] if labels else str(class_idx)
        print(f"     ✓ Clase: {label} ({top_probs[0]*100:.1f}%) → {saved_path.name}")
        return True

    except Exception as e:
        print(f"     ✗ Error: {e}")
        return False

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Grad-CAM · InceptionV3")
    print("=" * 60)

    # Seleccionar imágenes
    print("\n[1/3] Abriendo selector de imágenes...")
    image_paths = select_images()

    if not image_paths:
        print("  ✗ No se seleccionaron imágenes. Saliendo.")
        return

    print(f"  ✓ {len(image_paths)} imagen(es) seleccionada(s)")

    # Cargar modelo
    print("\n[2/3] Cargando InceptionV3...")
    model = models.inception_v3(weights='IMAGENET1K_V1')
    model.eval()
    grad_cam = GradCAM(model, model.Mixed_7c)
    labels = load_imagenet_labels()
    print("  ✓ Modelo listo")

    # Procesar en cola
    print(f"\n[3/3] Procesando cola → salida en: {OUTPUT_DIR}")
    print("-" * 60)

    ok, fail = 0, 0
    for i, path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}]", end=" ")
        success = process_image(path, model, grad_cam, labels, OUTPUT_DIR)
        if success:
            ok += 1
        else:
            fail += 1

    print("-" * 60)
    print(f"\n  Completado: {ok} exitosas, {fail} fallidas")
    print(f"  Resultados en: {OUTPUT_DIR}\n")

if __name__ == "__main__":
    main()