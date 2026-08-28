

import cv2
import numpy as np
from PIL import Image


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
