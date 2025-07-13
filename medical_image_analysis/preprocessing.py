"""
Medical Image Analysis Pipeline - Preprocessing Module
This module handles image loading, normalization, and preprocessing operations.
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import random
from .config import TARGET_IMAGE_SIZE, MODALITY, MODALITY_PARAMS


def load_and_preprocess_image(image_path: str) -> Image.Image:
    """Loads a PNG image, converts to grayscale, normalizes intensity, and resizes."""
    print(f"1.1 Loading image: {image_path}")
    try:
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}. Creating a fictional image.")
        img = create_fictional_image(image_path)
        img = Image.open(image_path).convert('RGB')

    print("1.2 Converting to grayscale...")
    img_gray = img.convert('L')

    print("1.3 Normalizing intensity...")
    img_np = np.array(img_gray)
    params = MODALITY_PARAMS[MODALITY]
    norm_range = params["normalization_range"]
    min_val = np.min(img_np)
    max_val = np.max(img_np)

    if max_val > min_val:
        if MODALITY == "CT":
            img_normalized_np = ((img_np - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            img_normalized_np = ((img_np - min_val) / (max_val - min_val) *
                               (norm_range[1] - norm_range[0]) + norm_range[0]).astype(np.uint8)
    else:
        img_normalized_np = np.zeros_like(img_np, dtype=np.uint8)

    img_normalized = Image.fromarray(img_normalized_np)

    print(f"1.4 Resizing to {TARGET_IMAGE_SIZE}...")
    img_resized = img_normalized.resize(TARGET_IMAGE_SIZE, Image.LANCZOS)

    return img_resized


def create_fictional_image(output_path: str = "example_input_image.png") -> Image.Image:
    """Creates a simple fictional PNG image for testing, adjusted by modality."""
    print(f"Creating fictional image at {output_path}...")
    img = Image.new('RGB', (512, 512), color='white')
    d = ImageDraw.Draw(img)

    params = MODALITY_PARAMS[MODALITY]
    circle_radius = 150 if MODALITY == "CT" else 180
    circle_center = (256, 256)
    fill_color = (150, 150, 150) if MODALITY == "CT" else (180, 180, 180)

    d.ellipse((circle_center[0] - circle_radius, circle_center[1] - circle_radius,
               circle_center[0] + circle_radius, circle_center[1] + circle_radius),
              fill=fill_color, outline=(50, 50, 50), width=5)

    for _ in range(1000):
        x = random.randint(0, 511)
        y = random.randint(0, 511)
        color = random.randint(100, 200) if MODALITY == "CT" else random.randint(120, 220)
        d.point((x, y), fill=(color, color, color))

    img.save(output_path)
    print("Fictional image created.")
    return img


def validate_image(image_path: str) -> bool:
    """Validates if the image file exists and is readable."""
    try:
        if not os.path.exists(image_path):
            return False
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        print(f"Image validation failed: {e}")
        return False
