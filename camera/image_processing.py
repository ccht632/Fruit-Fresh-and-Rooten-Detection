

import cv2
import numpy as np
from PIL import Image


def denoise(image: Image.Image) -> Image.Image:
    img = np.array(image)
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    return Image.fromarray(denoised)


def enhance_image(image: Image.Image) -> Image.Image:
    """Preprocessing pipeline, run before detection."""
    return denoise(image)
