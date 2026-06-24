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
from scipy.ndimage import gaussian_filter

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = "./dinov2_outputs"
ALPHA_OVERLAY = 0.6
MAX_RESOLUTION = 518  # Múltiplo de 14 (tamaño del parche de DINOv2)
PATCH_SIZE = 14

# ══════════════════════════════════════════════════════════════════════════════

preprocess_tensors = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─── Extractor de Atención DINOv2 (VERSIÓN QKV) ──────────────────────────────
class DINOv2AttentionExtractor:
    def __init__(self, model):
        self.model = model          # ← PRIMERO, antes de usar self.model
        self.attention_list = []
        self.hook_handles   = []

        for block in self.model.blocks[-4:]:
            h = block.attn.qkv.register_forward_hook(self._qkv_hook)
            self.hook_handles.append(h)

    def _qkv_hook(self, module, input, output):
        B, N, C_times_3 = output.shape
        num_heads = self.model.blocks[-1].attn.num_heads
        head_dim  = (C_times_3 // 3) // num_heads

        qkv  = output.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]

        attn = (q * (head_dim ** -0.5)) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        self.attention_list.append(attn.detach())

    def generate(self, x):
        self.attention_list = []    # limpia antes de cada imagen

        with torch.no_grad():
            self.model(x)

        if len(self.attention_list) == 0:
            raise RuntimeError("Ningún hook capturó datos.")

        # Stack: [num_blocks, num_heads, N, N]
        all_attns = torch.stack(self.attention_list)

        # Con dinov2_vitb14_reg (4 register tokens después del CLS):
        NUM_REGISTERS = 4
        cls_attn = all_attns[:, 0, :, 0, 1 + NUM_REGISTERS:]

        # max sobre cabezas por bloque → mean sobre bloques
        mean_attn = cls_attn.max(dim=1).values.mean(dim=0)

        return mean_attn

    def close(self):
        for h in self.hook_handles:
            h.remove()
# ─── Utilidades Visuales ──────────────────────────────────────────────────────
def overlay_heatmap(image_pil, cam, alpha=ALPHA_OVERLAY):
    img = np.array(image_pil).astype(np.uint8)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * img).astype(np.uint8)
    return overlay

# ─── Preprocesamiento para ViT ────────────────────────────────────────────────
def resize_for_dinov2(image, max_size=MAX_RESOLUTION, patch_size=PATCH_SIZE):
    """
    Escala la imagen manteniendo aspect ratio, limitando el lado mayor a max_size.
    Luego hace center crop para que ambas dimensiones sean múltiplos de patch_size.
    """
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    w, h = image.size
    new_w = (w // patch_size) * patch_size
    new_h = (h // patch_size) * patch_size

    # Recorte central — coordenadas ENTERAS (bug corregido)
    left   = (w - new_w) // 2
    top    = (h - new_h) // 2
    right  = left + new_w
    bottom = top  + new_h

    return image.crop((left, top, right, bottom))

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
        attn_map = gaussian_filter(attn_map, sigma=1.0)
        
        # 1.
        # FIX: normalización robusta por percentil (evita que un outlier colapse todo) #####revisar percentiles 1 a 99 
        p_low  = np.percentile(attn_map, 2)
        p_high = np.percentile(attn_map, 98)
        attn_map = np.clip(attn_map, p_low, p_high)
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        
        # Después de normalizar, antes del upsample
        threshold = attn_map.mean() + 0.5 * attn_map.std()
        attn_map = np.where(attn_map >= threshold, attn_map, attn_map * 0.1)

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
    print("  Atención Semántica · Meta DINOv2 (ViT-Base/reg)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Hardware] Usando: {device.type.upper()}")

    # ── Selección de directorio ──────────────────────────────────
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    input_dir = filedialog.askdirectory(title="Selecciona el directorio del dataset")
    root.destroy()

    if not input_dir:
        print("  ✗ Cancelado.")
        return

    input_dir  = Path(input_dir)
    output_dir = ('output_' + input_dir.name)

    # ── Escanear imágenes ────────────────────────────────────────
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp",".ARW"}
    image_paths = sorted([
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not image_paths:
        print(f"  ✗ No se encontraron imágenes en: {input_dir}")
        return

    print(f"\n[Dataset] {input_dir.name}/ → {len(image_paths)} imágenes encontradas")

    # ── Modelo ───────────────────────────────────────────────────
    print("\n[Cargando Modelo] DINOv2 ViT-Base con registers...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg').to(device)
    model.eval()

    extractor = DINOv2AttentionExtractor(model)

    # ── Procesamiento ────────────────────────────────────────────
    print(f"\n[Procesando] Cola de {len(image_paths)} imágenes...")
    print(f"[Output]     {output_dir}")
    print("-" * 60)

    ok, fail = 0, 0
    for i, path in enumerate(image_paths, 1):
        print(f"[{i:>4}/{len(image_paths)}]", end=" ")
        if process_image(path, model, extractor, output_dir, device):
            ok += 1
        else:
            fail += 1

    extractor.close()
    print("-" * 60)
    print(f"  ✓ Completadas: {ok}  ✗ Fallidas: {fail}")
    print(f"  Resultados en: {output_dir}\n")

if __name__ == "__main__":
    main()