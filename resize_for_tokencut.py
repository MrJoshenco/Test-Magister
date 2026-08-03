#!/usr/bin/env python3
"""
Redimensiona las imágenes de un dataset para que sean compatibles con TokenCut.

TokenCut (DINO ViT) utiliza parches de 16 píxeles. Imágenes muy grandes generan
un número cuadrático de parches y pueden agotar la memoria de la GPU. Este script
escala las imágenes manteniendo la relación de aspecto, limitando el lado mayor a
un tamaño objetivo y redondeando las dimensiones a múltiplos del tamaño de parche.
"""
import argparse
import os
from pathlib import Path
from PIL import Image


def nearest_multiple(x, patch_size):
    """Redondea x al múltiplo más cercano de patch_size."""
    return max(patch_size, round(x / patch_size) * patch_size)


def resize_for_tokencut(image, target_max=1024, patch_size=16):
    """
    Escala una imagen para que su lado mayor sea como máximo target_max y ambas
    dimensiones sean múltiplos de patch_size. Mantiene la relación de aspecto con
    un redondeo mínimo a múltiplos del parche.
    """
    w, h = image.size
    if max(w, h) <= target_max and w % patch_size == 0 and h % patch_size == 0:
        return image.copy()

    scale = target_max / max(w, h)
    new_w = nearest_multiple(w * scale, patch_size)
    new_h = nearest_multiple(h * scale, patch_size)

    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def main():
    parser = argparse.ArgumentParser(
        description="Redimensiona imágenes para TokenCut/DINO ViT."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="Dataset_tesis_numerado_ligero",
        help="Directorio con las imágenes originales.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directorio de salida. Si no se indica, se sobrescriben las originales.",
    )
    parser.add_argument(
        "--target-max",
        type=int,
        default=1024,
        help="Tamaño máximo del lado mayor de la imagen (por defecto 1024).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="Tamaño de parche del modelo (por defecto 16 para DINO).",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="Calidad JPEG de salida (por defecto 95).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    if not image_paths:
        print(f"No se encontraron imágenes en {input_dir}")
        return

    print(f"Procesando {len(image_paths)} imágenes...")
    print(f"  Directorio entrada: {input_dir}")
    print(f"  Directorio salida:  {output_dir}")
    print(f"  Lado mayor máximo: {args.target_max}")
    print(f"  Parche: {args.patch_size}x{args.patch_size}")
    print()

    for i, p in enumerate(image_paths, 1):
        if i % 10 == 0 or i == 1 or i == len(image_paths):
            print(f"  [{i}/{len(image_paths)}] {p.name}")
        try:
            with Image.open(p) as img:
                img = img.convert("RGB")
                resized = resize_for_tokencut(img, args.target_max, args.patch_size)
                out_path = output_dir / p.name
                resized.save(out_path, quality=args.quality, optimize=True)
        except Exception as e:
            print(f"\nError procesando {p.name}: {e}")

    print("\nResumen de tamaños finales:")
    for p in sorted(output_dir.iterdir())[:10]:
        if p.suffix.lower() in extensions:
            with Image.open(p) as img:
                print(f"  {p.name}: {img.size}")


if __name__ == "__main__":
    main()
