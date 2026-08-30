

import cv2
import numpy as np
from PIL import Image, ImageOps

MAX_IMAGE_DIMENSION = 1280


def preprocess_image(image: Image.Image) -> Image.Image:
    # Fix orientation and downscale large photos before detection.
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max(image.size) > MAX_IMAGE_DIMENSION:
        image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.LANCZOS)
    return image


def denoise(image: Image.Image) -> Image.Image:
    try:
        img = np.array(image.convert("RGB"))
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
        return Image.fromarray(denoised)
    except cv2.error as e:
        print(f"WARNING: denoising failed ({e}); using the original image instead.")
        return image


def enhance_image(image: Image.Image) -> Image.Image:
    # Preprocessing pipeline, run before detection.
    return denoise(image)
