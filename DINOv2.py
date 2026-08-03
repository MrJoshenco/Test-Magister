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
MAX_RESOLUTION = 518  # Múltiplo de 14 (tamaño del parche de DINOv2). 
PATCH_SIZE = 14

# ─── Hiperparámetros ajustables del pipeline ─────────────────────────────────
# Estas constantes pueden ser modificadas por tune_dinov2.py para grid search.
GAUSSIAN_SIGMA = 1.0          # Suavizado gaussiano del mapa de atención
PERCENTILE_LOW = 2            # Percentil inferior para normalización robusta
PERCENTILE_HIGH = 98          # Percentil superior para normalización robusta
TEMPERATURE = 0.3             # Sharpening del mapa (<1 agudiza, >1 suaviza)
GUIDED_FILTER_RADIUS = 25     # Radio del guided filter (edge-aware smoothing)
GUIDED_FILTER_EPS = 1e-3      # Regularización del guided filter
K_STD = 0.7                   # Factor del umbral relativo: mean + k_std * std

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
def make_heatmap_bgr(cam):
    """cam: float [0,1] → heatmap color BGR (uint8), listo para cv2.imwrite."""
    return cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

def overlay_heatmap(image_pil, cam, alpha=ALPHA_OVERLAY):
    img = np.array(image_pil).astype(np.uint8)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * img).astype(np.uint8)
    return overlay

def guided_filter(guide_gray, src, radius=20, eps=1e-3):
    """
    Guided filter (He et al.) con guía en escala de grises, implementado con
    box filters puros (no requiere opencv-contrib).

    guide_gray : float32 [0,1], imagen original en escala de grises, tamaño full-res
    src        : float32 [0,1], mapa a refinar, YA escalado al mismo tamaño que guide_gray
    radius     : tamaño de ventana del box filter → controla cuánto "mira" alrededor
    eps        : regularización → más alto = más suave / menos detalle de textura

    Devuelve el mapa refinado, alineado a los bordes reales de la imagen.
    """
    I = guide_gray.astype(np.float32)
    p = src.astype(np.float32)
    ksize = (radius, radius)

    mean_I  = cv2.boxFilter(I, cv2.CV_32F, ksize)
    mean_p  = cv2.boxFilter(p, cv2.CV_32F, ksize)
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, ksize)
    cov_Ip  = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, ksize)
    var_I   = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

    q = mean_a * I + mean_b
    return np.clip(q, 0.0, 1.0)

def make_binary_mask(cam_guided_full_res, blur_ksize=0, method="relative", 
                      k_std=0.5, fixed_frac=0.35):
    """
    method:
      - "otsu"     : comportamiento original
      - "relative" : umbral = mean + k_std * std  (recomendado con temperature scaling)
      - "fixed"    : umbral = fixed_frac * max(mapa)
    """
    cam = cam_guided_full_res.astype(np.float32)
    if blur_ksize and blur_ksize > 1:
        cam = cv2.GaussianBlur(cam, (blur_ksize, blur_ksize), 0)

    if method == "otsu":
        cam_u8 = np.uint8(255 * cam)
        _, mask_full = cv2.threshold(cam_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif method == "relative":
        thresh_val = cam.mean() + k_std * cam.std()
        mask_full = np.where(cam > thresh_val, 255, 0).astype(np.uint8)

    elif method == "fixed":
        thresh_val = fixed_frac * cam.max()
        mask_full = np.where(cam > thresh_val, 255, 0).astype(np.uint8)

    else:
        raise ValueError(f"Método desconocido: {method}")

    # Limpieza morfológica (igual que antes)
    kernel = np.ones((3, 3), np.uint8)
    mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, kernel)
    mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_OPEN, kernel)

    return mask_full

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
        attn_map = gaussian_filter(attn_map, sigma=GAUSSIAN_SIGMA)
        
        # 1.
        # FIX: normalización robusta por percentil (evita que un outlier colapse todo) #####revisar percentiles 1 a 99 
        p_low  = np.percentile(attn_map, PERCENTILE_LOW)
        p_high = np.percentile(attn_map, PERCENTILE_HIGH)
        attn_map = np.clip(attn_map, p_low, p_high)
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        
        # Después de normalizar, antes del upsample
        # TEMPERATURE < 1 agudiza; experimentar entre 0.3-0.7
        attn_map = np.power(attn_map, 1.0 / TEMPERATURE)
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        # 2. Upsampling a la resolución original (sin el padding)
        target_size = (image_original.size[0], image_original.size[1])  # (W, H) para cv2
        attn_map_cubic = cv2.resize(attn_map, target_size,
                                     interpolation=cv2.INTER_CUBIC)
        attn_map_cubic = np.clip(attn_map_cubic, 0.0, 1.0)

        # 2.b Refinado edge-aware con guided filter: usa la imagen original en
        #     escala de grises como guía para que el mapa se pegue a los bordes
        #     reales del sujeto, en vez de quedar "borroso" o "en bloques".
        guide_gray = cv2.cvtColor(np.array(image_original), cv2.COLOR_RGB2GRAY)
        guide_gray = guide_gray.astype(np.float32) / 255.0
        attn_map_resized = guided_filter(guide_gray, attn_map_cubic, radius=GUIDED_FILTER_RADIUS, eps=GUIDED_FILTER_EPS)

        # 3. Aplicar Overlay sobre la imagen original (no la padded)
        overlay = overlay_heatmap(image_original, attn_map_resized)  # RGB

        # 4. Mapa de calor puro (color BGR, listo para imwrite)
        heatmap_bgr = make_heatmap_bgr(attn_map_resized)

        # 5. Máscara binaria (fondo vs. detección), umbralizada directamente
        #    sobre el mapa ya refinado a resolución completa → contornos finos
        mask_full = make_binary_mask(attn_map_resized, method="relative", k_std=K_STD)

        # 6. Guardar los 3 archivos por separado
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%H%M%S")
        base_name = f"{Path(image_path).stem}"

        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(out_path / f"{base_name}_heatmap.png"), heatmap_bgr)
        cv2.imwrite(str(out_path / f"{base_name}_overlay.png"), overlay_bgr)
        cv2.imwrite(str(out_path / f"{base_name}.png"), mask_full)

        print(f"     ✓ Procesado con {h_patches * w_patches} tokens espaciales → OK "
              f"(heatmap / overlay / mask)")
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

        # Liberar memoria de CUDA entre imágenes para evitar fragmentación y OOM
        if device.type == "cuda":
            torch.cuda.empty_cache()

    extractor.close()
    print("-" * 60)
    print(f"  ✓ Completadas: {ok}  ✗ Fallidas: {fail}")
    print(f"  Resultados en: {output_dir}\n")

if __name__ == "__main__":
    main()