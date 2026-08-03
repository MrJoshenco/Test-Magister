#!/usr/bin/env python3
"""
run_all_methods.py

Ejecuta tres métodos de descubrimiento de objetos / saliency sobre el mismo
dataset y los evalúa con eval_saliency contra el Ground Truth:

  1. TokenCut (versión estándar, ViT-S/16)
  2. LOST      (versión estándar, ViT-S/16)
  3. DINOv2    (con los parámetros obtenidos del tunning)

Para cada imagen se genera una predicción de saliency:
  - TokenCut: eigenvector (segundo autovector más pequeño) upsampleado a
    resolución de imagen.
  - LOST: mapa de inverse degree (scores) upsampleado a resolución de imagen.
  - DINOv2: máscara binaria generada con los parámetros de tunning.

Al finalizar se imprimen:
  - Tablas de métricas por categoría (25 imágenes) y resumen total para cada
    método (MAE, Max F-beta, mIoU, CorLoc).
  - Tabla aparte con media, desviación estándar e IC 95 % del tiempo de
    procesamiento por imagen para cada método.

Uso:
    .venv/bin/python3 run_all_methods.py
    python3 run_all_methods.py

Argumentos útiles:
    --input_dir      Dataset de imágenes (default: Dataset_tesis_numerado_ligero)
    --gt_dir         Ground truth (default: got_masks_v2)
    --output_dir     Raíz de salidas (default: outputs_all_methods)
    --batch_size     Tamaño de categoría para eval_saliency (default: 25)
    --confidence     Nivel de confianza para intervalos (default: 0.95)
"""
import argparse
import contextlib
import importlib.util
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# --------------------------------------------------------------------------- #
# Paths e imports modulares de TokenCut / LOST
#
# Los directorios TokenCut y LOST comparten nombres de módulos (networks,
# datasets, object_discovery, dino). Para evitar colisiones en sys.modules se
# importan con nombres únicos usando importlib.util y se limpia el caché entre
# cada grupo.
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).parent.resolve()


def _import_from_path(module_name, file_path, extra_paths=None):
    """Importa un módulo Python desde una ruta concreta con nombre único."""
    paths_to_restore = sys.path.copy()
    try:
        if extra_paths:
            for p in reversed(extra_paths):
                sys.path.insert(0, str(p))
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path = paths_to_restore


def _clear_generic_modules():
    """Elimina del caché los módulos genéricos para forzar re-importación."""
    for name in list(sys.modules.keys()):
        if name in ("networks", "datasets", "object_discovery", "dino",
                    "dino.vision_transformer", "dino.utils"):
            del sys.modules[name]


# TokenCut
tc_networks = _import_from_path(
    "tc_networks", PROJECT_ROOT / "TokenCut" / "networks.py", [PROJECT_ROOT / "TokenCut"]
)
tc_object_discovery = _import_from_path(
    "tc_object_discovery", PROJECT_ROOT / "TokenCut" / "object_discovery.py", [PROJECT_ROOT / "TokenCut"]
)
tc_datasets = _import_from_path(
    "tc_datasets", PROJECT_ROOT / "TokenCut" / "datasets.py", [PROJECT_ROOT / "TokenCut"]
)

_clear_generic_modules()

# LOST
lost_networks = _import_from_path(
    "lost_networks", PROJECT_ROOT / "LOST" / "networks.py", [PROJECT_ROOT / "LOST"]
)
lost_object_discovery = _import_from_path(
    "lost_object_discovery", PROJECT_ROOT / "LOST" / "object_discovery.py", [PROJECT_ROOT / "LOST"]
)
lost_datasets = _import_from_path(
    "lost_datasets", PROJECT_ROOT / "LOST" / "datasets.py", [PROJECT_ROOT / "LOST"]
)

_clear_generic_modules()

# DINOv2 y evaluador
import DINOv2
import eval_saliency


# --------------------------------------------------------------------------- #
# Utilidades generales
# --------------------------------------------------------------------------- #

def list_images(image_dir):
    """Lista y ordena las imágenes soportadas en un directorio."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    return sorted(
        p for p in Path(image_dir).iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


def patch_map_to_image(patch_map, dims, target_H, target_W):
    """
    Sube un mapa definido en el grid de parches (H_p, W_p) a la resolución
    original de la imagen (target_H, target_W) usando interpolación bilineal.
    Normaliza el resultado a uint8 [0, 255].
    """
    H_p, W_p = dims
    if patch_map.ndim == 1:
        patch_map = patch_map.reshape(H_p, W_p)

    tensor = torch.from_numpy(patch_map.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    upsampled = F.interpolate(
        tensor,
        size=(target_H, target_W),
        mode="bilinear",
        align_corners=False,
    )
    img = upsampled.squeeze().cpu().numpy()

    mn, mx = img.min(), img.max()
    if mx - mn > 1e-8:
        img = (img - mn) / (mx - mn)
    else:
        img = np.zeros_like(img)
    img = (img * 255).astype(np.uint8)
    return img


def bbox_to_mask(pred, H, W):
    """Convierte una bounding box [x1, y1, x2, y2] en máscara binaria 0/255 (no usado para eval)."""
    mask = np.zeros((H, W), dtype=np.uint8)
    x1, y1, x2, y2 = [int(round(float(v))) for v in pred]
    x1 = max(0, min(x1, W - 1))
    x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H - 1))
    y2 = max(0, min(y2, H))
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 255
    return mask


def timing_stats(times, confidence=0.95):
    """Media, std, SEM e IC para una lista de tiempos por imagen."""
    arr = np.asarray(times, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 0 else 0.0
    t_val = eval_saliency.t_critical(n - 1, confidence) if n > 1 else 0.0
    margin = t_val * sem
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_low": mean - margin,
        "ci_high": mean + margin,
    }


def format_timing_table(method_times, confidence=0.95):
    """Formatea la tabla de tiempos por método."""
    pct = int(round(confidence * 100))
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append(f" TIEMPOS DE PROCESAMIENTO POR IMAGEN (IC {pct}%)")
    lines.append("=" * 80)
    lines.append(f"{'Método':<20} {'N':>6} {'Media (s)':>12} {'± Std (s)':>12} {'IC':>24}")
    lines.append("-" * 80)
    for method, times in method_times.items():
        s = timing_stats(times, confidence)
        ci = f"[{s['ci_low']:.4f}, {s['ci_high']:.4f}]"
        lines.append(
            f"{method:<20} {s['n']:>6} {s['mean']:>12.4f} {s['std']:>12.4f} {ci:>24}"
        )
    lines.append("=" * 80)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# TokenCut
# --------------------------------------------------------------------------- #

def run_tokencut(image_paths, output_mask_dir, arch="vit_small", patch_size=16,
                 which_features="k", tau=0.2, eps=1e-5, no_binary_graph=False):
    """Ejecuta TokenCut sobre una lista de imágenes y devuelve tiempos."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TokenCut] Cargando modelo {arch}/{patch_size} en {device}...")
    model = tc_networks.get_model(arch, patch_size, device)

    output_mask_dir = Path(output_mask_dir)
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    times = []

    print(f"[TokenCut] Procesando {len(image_paths)} imágenes...")
    for img_path in image_paths:
        t0 = time.time()
        try:
            dataset = tc_datasets.ImageDataset(str(img_path))
            img = dataset.dataloader[0][0]
            init_image_size = img.shape

            size_im = (
                img.shape[0],
                int(np.ceil(img.shape[1] / patch_size) * patch_size),
                int(np.ceil(img.shape[2] / patch_size) * patch_size),
            )
            paded = torch.zeros(size_im)
            paded[:, : img.shape[1], : img.shape[2]] = img
            img = paded
            if device.type == "cuda":
                img = img.cuda(non_blocking=True)

            w_featmap = img.shape[-2] // patch_size
            h_featmap = img.shape[-1] // patch_size

            feat_out = {}
            def hook_fn_forward_qkv(module, input, output):
                feat_out["qkv"] = output
            model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

            attentions = model.get_last_selfattention(img[None, :, :, :])
            nb_im = attentions.shape[0]
            nh = attentions.shape[1]
            nb_tokens = attentions.shape[2]

            qkv = (
                feat_out["qkv"]
                .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
            k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
            q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
            v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

            if which_features == "k":
                feats = k
            elif which_features == "q":
                feats = q
            elif which_features == "v":
                feats = v
            else:
                raise ValueError("which_features debe ser k/q/v")

            scales = [patch_size, patch_size]
            pred, _, _, _, _, eigenvec = tc_object_discovery.ncut(
                feats,
                [w_featmap, h_featmap],
                scales,
                init_image_size,
                tau,
                eps,
                im_name=img_path.stem,
                no_binary_graph=no_binary_graph,
            )

            H, W = init_image_size[1], init_image_size[2]
            saliency = patch_map_to_image(eigenvec, [w_featmap, h_featmap], H, W)
            out_path = output_mask_dir / f"{img_path.stem}.png"
            cv2.imwrite(str(out_path), saliency)
        except Exception as e:
            print(f"  [TokenCut FAIL] {img_path.name}: {e}")
            continue
        finally:
            elapsed = time.time() - t0
            times.append(elapsed)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"[TokenCut] OK: {len(times)}/{len(image_paths)} imágenes")
    return times


# --------------------------------------------------------------------------- #
# LOST
# --------------------------------------------------------------------------- #

def run_lost(image_paths, output_mask_dir, arch="vit_small", patch_size=16,
             which_features="k", k_patches=100):
    """Ejecuta LOST sobre una lista de imágenes y devuelve tiempos."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LOST] Cargando modelo {arch}/{patch_size} en {device}...")
    model = lost_networks.get_model(arch, patch_size, resnet_dilate=2, device=device)

    output_mask_dir = Path(output_mask_dir)
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    times = []

    print(f"[LOST] Procesando {len(image_paths)} imágenes...")
    for img_path in image_paths:
        t0 = time.time()
        try:
            dataset = lost_datasets.ImageDataset(str(img_path))
            img = dataset.dataloader[0][0]
            init_image_size = img.shape

            size_im = (
                img.shape[0],
                int(np.ceil(img.shape[1] / patch_size) * patch_size),
                int(np.ceil(img.shape[2] / patch_size) * patch_size),
            )
            paded = torch.zeros(size_im)
            paded[:, : img.shape[1], : img.shape[2]] = img
            img = paded
            if device.type == "cuda":
                img = img.cuda(non_blocking=True)

            w_featmap = img.shape[-2] // patch_size
            h_featmap = img.shape[-1] // patch_size

            feat_out = {}
            def hook_fn_forward_qkv(module, input, output):
                feat_out["qkv"] = output
            model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

            attentions = model.get_last_selfattention(img[None, :, :, :])
            nb_im = attentions.shape[0]
            nh = attentions.shape[1]
            nb_tokens = attentions.shape[2]

            qkv = (
                feat_out["qkv"]
                .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
            k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
            q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
            v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

            if which_features == "k":
                feats = k[:, 1:, :]
            elif which_features == "q":
                feats = q[:, 1:, :]
            elif which_features == "v":
                feats = v[:, 1:, :]
            else:
                raise ValueError("which_features debe ser k/q/v")

            scales = [patch_size, patch_size]
            pred, A, scores, seed = lost_object_discovery.lost(
                feats,
                [w_featmap, h_featmap],
                scales,
                init_image_size,
                k_patches=k_patches,
            )

            H, W = init_image_size[1], init_image_size[2]
            # scores es el inverse degree centrality (negativo). Invertir para
            # obtener un mapa de saliencia donde mayor valor = más saliente.
            scores_np = -scores.cpu().numpy()  # grado positivo
            saliency_map = scores_np.max() - scores_np  # menor grado -> mayor saliencia
            saliency = patch_map_to_image(saliency_map, [w_featmap, h_featmap], H, W)
            out_path = output_mask_dir / f"{img_path.stem}.png"
            cv2.imwrite(str(out_path), saliency)
        except Exception as e:
            print(f"  [LOST FAIL] {img_path.name}: {e}")
            continue
        finally:
            elapsed = time.time() - t0
            times.append(elapsed)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"[LOST] OK: {len(times)}/{len(image_paths)} imágenes")
    return times


# --------------------------------------------------------------------------- #
# DINOv2
# --------------------------------------------------------------------------- #

def run_dinov2(image_paths, output_dir, params, device=None):
    """Ejecuta DINOv2 con parámetros de tunning y devuelve tiempos."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Aplicar parámetros de tunning a las constantes del módulo DINOv2
    for key, value in params.items():
        if hasattr(DINOv2, key):
            setattr(DINOv2, key, value)
        else:
            print(f"[DINOv2 AVISO] DINOv2 no tiene constante '{key}'")

    print(f"[DINOv2] Cargando modelo dinov2_vitb14_reg en {device}...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg").to(device)
    model.eval()
    extractor = DINOv2.DINOv2AttentionExtractor(model)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    times = []

    print(f"[DINOv2] Procesando {len(image_paths)} imágenes...")
    for img_path in image_paths:
        t0 = time.time()
        try:
            # Suprimir prints individuales de DINOv2.process_image para mantener log limpio
            with contextlib.redirect_stdout(io.StringIO()):
                DINOv2.process_image(str(img_path), model, extractor, str(output_dir), device)
        except Exception as e:
            print(f"  [DINOv2 FAIL] {img_path.name}: {e}")
            continue
        finally:
            elapsed = time.time() - t0
            times.append(elapsed)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    extractor.close()
    print(f"[DINOv2] OK: {len(times)}/{len(image_paths)} imágenes")
    return times


# --------------------------------------------------------------------------- #
# Evaluación
# --------------------------------------------------------------------------- #

def evaluate_method(pred_dir, gt_dir, method_name, batch_size=25, k_std=0.5, confidence=0.95):
    """Evalúa un directorio de predicciones contra GT e imprime tablas."""
    print(f"\n[{method_name}] Evaluando contra GT...")
    batch_results_list, all_results = eval_saliency.evaluate_dataset(
        gt_dir, pred_dir, batch_size=batch_size, k_std=k_std
    )
    output_text = eval_saliency.print_all_results(
        batch_results_list, all_results, k_std=k_std, confidence=confidence
    )
    return output_text


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Ejecuta TokenCut, LOST y DINOv2 sobre un dataset y evalúa."
    )
    parser.add_argument("--input_dir", type=str, default="Dataset_tesis_numerado_ligero",
                        help="Directorio con imágenes de entrada.")
    parser.add_argument("--gt_dir", type=str, default="got_masks_v2",
                        help="Directorio con máscaras Ground Truth.")
    parser.add_argument("--output_dir", type=str, default="outputs_all_methods",
                        help="Directorio raíz para salidas y reporte.")
    parser.add_argument("--batch_size", type=int, default=25,
                        help="Tamaño de categoría para eval_saliency.")
    parser.add_argument("--confidence", type=float, default=0.95,
                        help="Nivel de confianza para intervalos.")
    parser.add_argument("--skip", type=str, default="",
                        help="Métodos a omitir separados por coma (tokencut,lost,dinov2).")
    parser.add_argument("--exclude_ids", type=str, default="",
                        help="IDs numéricos de imágenes a excluir separados por coma (p. ej. 1,6,7).")

    # Parámetros de DINOv2 (defaults = configuración ganadora del tuning, rank 1 por mIoU)
    parser.add_argument("--dinov2_max_resolution", type=int, default=980)
    parser.add_argument("--dinov2_temperature", type=float, default=0.3)
    parser.add_argument("--dinov2_guided_filter_radius", type=int, default=15)
    parser.add_argument("--dinov2_k_std", type=float, default=0.2)
    parser.add_argument("--dinov2_gaussian_sigma", type=float, default=1.5)
    parser.add_argument("--dinov2_percentile_low", type=int, default=2)
    parser.add_argument("--dinov2_percentile_high", type=int, default=98)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skip = set(m.strip().lower() for m in args.skip.split(",") if m.strip())
    exclude_ids = set(int(x.strip()) for x in args.exclude_ids.split(",") if x.strip())

    image_paths = [p for p in list_images(input_dir) if int(p.stem) not in exclude_ids]
    if not image_paths:
        print(f"No se encontraron imágenes en {input_dir}")
        return

    excluded_info = f" (excluyendo {len(exclude_ids)} IDs de tuning)" if exclude_ids else ""
    print(f"=" * 80)
    print(" Ejecutando TokenCut + LOST + DINOv2")
    print(f" Dataset: {input_dir} ({len(image_paths)} imágenes){excluded_info}")
    print(f" GT:      {gt_dir}")
    print(f" Output:  {output_dir}")
    print(f"=" * 80)

    method_times = {}
    eval_texts = {}

    # Parámetros de tunning para DINOv2
    dinov2_params = {
        "ALPHA_OVERLAY": 0.6,
        "GAUSSIAN_SIGMA": args.dinov2_gaussian_sigma,
        "GUIDED_FILTER_EPS": 0.001,
        "GUIDED_FILTER_RADIUS": args.dinov2_guided_filter_radius,
        "K_STD": args.dinov2_k_std,
        "MAX_RESOLUTION": args.dinov2_max_resolution,
        "PATCH_SIZE": 14,
        "PERCENTILE_HIGH": args.dinov2_percentile_high,
        "PERCENTILE_LOW": args.dinov2_percentile_low,
        "TEMPERATURE": args.dinov2_temperature,
    }

    # 1. TokenCut
    if "tokencut" not in skip:
        tc_out = output_dir / "tokencut" / "masks"
        method_times["TokenCut"] = run_tokencut(image_paths, tc_out)
        eval_texts["TokenCut"] = evaluate_method(
            str(tc_out), str(gt_dir), "TokenCut",
            batch_size=args.batch_size, k_std=0.5, confidence=args.confidence
        )

    # 2. LOST
    if "lost" not in skip:
        lost_out = output_dir / "lost" / "masks"
        method_times["LOST"] = run_lost(image_paths, lost_out)
        eval_texts["LOST"] = evaluate_method(
            str(lost_out), str(gt_dir), "LOST",
            batch_size=args.batch_size, k_std=0.5, confidence=args.confidence
        )

    # 3. DINOv2
    if "dinov2" not in skip:
        dinov2_out = output_dir / "dinov2"
        method_times["DINOv2"] = run_dinov2(image_paths, dinov2_out, dinov2_params)
        eval_texts["DINOv2"] = evaluate_method(
            str(dinov2_out), str(gt_dir), "DINOv2",
            batch_size=args.batch_size, k_std=dinov2_params["K_STD"],
            confidence=args.confidence
        )

    # Tabla de tiempos
    timing_table = format_timing_table(method_times, confidence=args.confidence)
    print(timing_table)

    # Guardar reporte completo con métricas y tiempos
    report_path = output_dir / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Reporte generado por run_all_methods.py\n")
        f.write(f"Dataset: {input_dir}\n")
        f.write(f"GT: {gt_dir}\n")
        f.write(f"Imágenes: {len(image_paths)}\n")
        f.write(f"\n{'='*80}\n")
        f.write(" MÉTRICAS POR MÉTODO\n")
        f.write(f"{'='*80}\n")
        for method_name, text in eval_texts.items():
            f.write(f"\n--- {method_name} ---\n")
            f.write(text)
        f.write(f"\n\n{timing_table}\n")
    print(f"\n[INFO] Reporte guardado en: {report_path}")


if __name__ == "__main__":
    main()
