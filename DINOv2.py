import torch
import torch.nn.functional as F
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

OUTPUT_DIR = "./dinov2_outputs"
ALPHA_OVERLAY = 0.6
MAX_RESOLUTION = 518  # Múltiplo de 14 (tamaño del parche de DINOv2)
PATCH_SIZE = 14

# ══════════════════════════════════════════════════════════════════════════════

# ─── Preprocesamiento para ViT ───────────────────────────────────────────────
def resize_for_dinov2(image, max_size=MAX_RESOLUTION, patch_size=PATCH_SIZE):
    """
    Escala la imagen manteniendo el aspect ratio, limitando el lado más largo a max_size.
    Crucial: Recorta ligeramente para asegurar que las dimensiones sean múltiplos perfectos de 14.
    """
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    w, h = image.size
    # Ajustar al múltiplo de patch_size (14) más cercano hacia abajo
    new_w = (w // patch_size) * patch_size
    new_h = (h // patch_size) * patch_size
    
    # Recorte central
    left = (w - new_w) / 2
    top = (h - new_h) / 2
    right = (w + new_w) / 2
    bottom = (h + new_h) / 2
    
    return image.crop((left, top, right, bottom))

preprocess_tensors = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─── Extractor de Atención DINOv2 (VERSIÓN QKV) ──────────────────────────────
class DINOv2AttentionExtractor:
    def __init__(self, model):
        self.model = model
        self.hook_handle = None
        self.attention_weights = None
        
        # Interceptamos la proyección QKV del último bloque en lugar de attn_drop
        self.hook_handle = self.model.blocks[-1].attn.qkv.register_forward_hook(
            self._qkv_hook
        )

    def _qkv_hook(self, module, input, output):
        # El output de qkv tiene forma [Batch, N_tokens, 3 * embed_dim]
        B, N, C_times_3 = output.shape
        
        # Obtenemos la configuración de las cabezas (heads) desde el modelo
        num_heads = self.model.blocks[-1].attn.num_heads
        head_dim = (C_times_3 // 3) // num_heads
        
        # Desempaquetamos Q, K y V
        # Hacemos reshape a: [3, Batch, num_heads, N_tokens, head_dim]
        qkv = output.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]  # V (qkv[2]) no lo necesitamos para calcular la atención
        
        # Factor de escala estándar en Transformers
        scale = head_dim ** -0.5
        q = q * scale
        
        # Multiplicamos Q * K transpuesta para obtener los puntajes crudos
        attn = q @ k.transpose(-2, -1)  # [Batch, num_heads, N_tokens, N_tokens]
        
        # Aplicamos Softmax para obtener la distribución final de los mapas de atención
        attn = attn.softmax(dim=-1)
        
        self.attention_weights = attn.detach()

    def generate(self, x):
        with torch.no_grad():
            self.model(x)
            
        attn = self.attention_weights[0]  # [num_heads, N, N]
        cls_attn = attn[:, 0, 1:]  # [num_heads, num_patches]
        mean_attn = cls_attn.mean(dim=0)  # [num_patches]
        return mean_attn

    def close(self):
        if self.hook_handle:
            self.hook_handle.remove()

# ─── Utilidades Visuales ──────────────────────────────────────────────────────
def overlay_heatmap(image_pil, cam, alpha=ALPHA_OVERLAY):
    img = np.array(image_pil).astype(np.uint8)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * img).astype(np.uint8)
    return overlay

def resize_for_dinov2(image, patch_size=PATCH_SIZE):
    """Ajusta la imagen usando Padding para que sea múltiplo de 14."""
    w, h = image.size
    pad_w = (patch_size - w % patch_size) % patch_size
    pad_h = (patch_size - h % patch_size) % patch_size
    
    # Añadimos un borde negro (0,0,0) para completar el múltiplo
    new_image = Image.new("RGB", (w + pad_w, h + pad_h), (0, 0, 0))
    new_image.paste(image, (0, 0))
    return new_image

def process_image(image_path, model, extractor, output_dir, device):
    print(f"  → Analizando: {Path(image_path).name}")
    try:
        # Forzar conversión a RGB para evitar problemas con perfiles JPG CMYK/Grayscale
        image_original = Image.open(image_path).convert("RGB")
        image_resized = resize_for_dinov2(image_original)
        w, h = image_resized.size
        
        x = preprocess_tensors(image_resized).unsqueeze(0).to(device)

        # Extraer pesos
        raw_attn = extractor.generate(x)
        
        # El mapa de atención es (h/14) * (w/14)
        h_patches, w_patches = h // PATCH_SIZE, w // PATCH_SIZE
        attn_map = raw_attn.reshape(h_patches, w_patches).cpu().numpy()
        
        # 1. Normalización local: evitaremos que valores extremos sesguen todo
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        
        # 2. Upsampling a la resolución original (sin el padding)
        attn_map_resized = cv2.resize(attn_map, (image_original.size[0], image_original.size[1]), 
                                      interpolation=cv2.INTER_CUBIC)
        
        # 3. Aplicar Overlay sobre la imagen original (no la padded)
        overlay = overlay_heatmap(image_original, attn_map_resized)


        # 4. Guardar Grid
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%H%M%S")
        base_name = f"{Path(image_path).stem}_DINOv2_{timestamp}"

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.patch.set_facecolor("#0f0f0f")
        fig.suptitle(f"DINOv2 Self-Attention Map ({w}x{h})", color="white", fontsize=13, fontweight="bold", y=1.01)

        axes[0].imshow(image_resized)
        axes[0].set_title("Original", color="#aaaaaa", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(attn_map_resized, cmap="jet")
        axes[1].set_title("Attention (CLS Token)", color="#aaaaaa", fontsize=10)
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Segmentación Estructural", color="#aaaaaa", fontsize=10)
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(out_path / f"{base_name}_grid.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()

        print(f"     ✓ Procesado con {h_patches * w_patches} tokens espaciales → OK")
        return True

    except Exception as e:
        print(f"     ✗ Error crítico: {e}")
        return False

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Atención Semántica · Meta DINOv2 (ViT-Base)")
    print("=" * 60)

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

    print("\n[Cargando Modelo] Descargando/Cargando DINOv2 desde PyTorch Hub...")
    # dinov2_vitb14 es el balance perfecto entre precisión y velocidad
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
    model.eval()
    
    extractor = DINOv2AttentionExtractor(model)

    print(f"\n[Procesando] Cola de {len(image_paths)} imágenes...")
    print("-" * 60)

    for i, path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}]", end=" ")
        process_image(path, model, extractor, OUTPUT_DIR, device)

    extractor.close()
    print("-" * 60)
    print(f"  Resultados en: {OUTPUT_DIR}\n")

if __name__ == "__main__":
    main()