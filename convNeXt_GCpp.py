import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ConvNeXt_Base_Weights
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = "./gradcam_outputs"  # Cambia a tu ruta de preferencia
ALPHA_OVERLAY = 0.5
SAVE_INDIVIDUAL = False
TOP_K_CLASSES = 5
MAX_RESOLUTION = 1024  # Límite dinámico para el lado más largo

# ══════════════════════════════════════════════════════════════════════════════

def load_imagenet_labels():
    import urllib.request
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        with urllib.request.urlopen(url) as f:
            labels = [line.strip() for line in f.read().decode("utf-8").splitlines()]
        return labels
    except Exception:
        print("⚠ No se pudieron cargar labels de ImageNet.")
        return None

# ─── Preprocesamiento Dinámico ───────────────────────────────────────────────
preprocess_tensors = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def resize_dynamic(image, max_size=MAX_RESOLUTION):
    """Mantiene el aspect ratio, limitando el lado más largo a max_size."""
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return image

# ─── Grad-CAM++ ──────────────────────────────────────────────────────────────
class GradCAMPlusPlus:
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

        # Extraer tensores
        gradients = self.gradients  # [1, C, H, W]
        activations = self.activations  # [1, C, H, W]

        # ─── Matemática Grad-CAM++ ──────────────────────────────────────────
        # alpha_num = (grad)^2
        # alpha_denom = 2*(grad)^2 + \sum(A * (grad)^3)
        
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)
        
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + activations.mul(grad_3).sum(dim=(2, 3), keepdim=True)
        # Evitar división por cero
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        
        alphas = alpha_num / alpha_denom
        
        # Pesos = \sum (alpha * ReLU(grad))
        weights = (alphas * torch.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        
        # CAM = ReLU(\sum (weights * activations))
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsampling al tamaño original de la entrada
        _, _, H_in, W_in = x.size()
        cam = F.interpolate(cam, size=(H_in, W_in), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalización min-max
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        top_probs, top_indices = probs.topk(TOP_K_CLASSES, dim=1)
        return cam, class_idx, top_probs.squeeze().tolist(), top_indices.squeeze().tolist()

# ─── Overlay ─────────────────────────────────────────────────────────────────
def overlay_heatmap(image_pil, cam, alpha=ALPHA_OVERLAY):
    img = np.array(image_pil).astype(np.uint8)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * img).astype(np.uint8)
    return overlay, heatmap

def build_output_name(image_path, class_idx, labels, confidence):
    stem = Path(image_path).stem
    timestamp = datetime.now().strftime("%H%M%S")
    label = labels[class_idx].replace(" ", "_") if labels else f"cls{class_idx}"
    return f"{stem}_{label}_{confidence*100:.0f}pct_{timestamp}"

# ─── Guardar resultados ───────────────────────────────────────────────────────
def save_results(image_path, image_pil, cam, overlay, class_idx, labels, top_probs, top_indices, output_dir):
    base_name = build_output_name(image_path, class_idx, labels, top_probs[0])
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    label_names = [labels[i].replace("_", " ") if labels else str(i) for i in top_indices]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0f0f0f")

    title_main = f"Top-1: {label_names[0]}  ({top_probs[0]*100:.1f}%)"
    fig.suptitle(title_main, color="white", fontsize=13, fontweight="bold", y=1.01)

    axes[0].imshow(image_pil)
    axes[0].set_title(f"Original ({image_pil.size[0]}x{image_pil.size[1]})", color="#aaaaaa", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    subtitle = " | ".join([f"{n} {p*100:.0f}%" for n, p in zip(label_names, top_probs)])
    axes[1].set_title(f"Grad-CAM++\n{subtitle}", color="#aaaaaa", fontsize=8)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Superposición", color="#aaaaaa", fontsize=10)
    axes[2].axis("off")

    plt.tight_layout()
    grid_path = out_path / f"{base_name}_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    if SAVE_INDIVIDUAL:
        plt.imsave(str(out_path / f"{base_name}_heatmap.png"), cam, cmap="jet")
        Image.fromarray(overlay).save(str(out_path / f"{base_name}_overlay.png"))

    return grid_path

# ─── Procesamiento ────────────────────────────────────────────────────────────
def process_image(image_path, model, grad_cam, labels, output_dir, device):
    print(f"  → Procesando: {Path(image_path).name}")
    try:
        # Carga y redimensionamiento dinámico
        image_original = Image.open(image_path).convert("RGB")
        image_resized = resize_dynamic(image_original)
        
        x = preprocess_tensors(image_resized).unsqueeze(0).to(device)
        x.requires_grad_(True)

        cam, class_idx, top_probs, top_indices = grad_cam.generate(x)
        overlay, _ = overlay_heatmap(image_resized, cam)

        saved_path = save_results(
            image_path, image_resized, cam, overlay,
            class_idx, labels, top_probs, top_indices, output_dir
        )

        label = labels[class_idx] if labels else str(class_idx)
        print(f"     ✓ Clase: {label} ({top_probs[0]*100:.1f}%) → OK")
        return True

    except Exception as e:
        print(f"     ✗ Error crítico: {e}")
        return False

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Grad-CAM++ · ConvNeXt Base")
    print("=" * 60)

    # Autodetectar GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Hardware] Usando: {device.type.upper()}")

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    image_paths = filedialog.askopenfilenames(title="Selecciona imágenes")
    root.destroy()

    if not image_paths:
        print("  ✗ Cancelado.")
        return

    print("\n[Cargando Modelo] ConvNeXt Base...")
    weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
    model = models.convnext_base(weights=weights).to(device)
    model.eval()

    # Target layer: La última capa del último bloque convolucional en ConvNeXt
    target_layer = model.features[-1][-1]
    
    grad_cam_plus = GradCAMPlusPlus(model, target_layer)
    labels = load_imagenet_labels()

    print(f"\n[Procesando] Cola de {len(image_paths)} imágenes...")
    print("-" * 60)

    for i, path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}]", end=" ")
        process_image(path, model, grad_cam_plus, labels, OUTPUT_DIR, device)

    print("-" * 60)
    print(f"  Resultados en: {OUTPUT_DIR}\n")

if __name__ == "__main__":
    main()