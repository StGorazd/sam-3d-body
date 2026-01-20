import os

import cv2
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.webp',
    '.heic', '.HEIC', '.heif', '.HEIF'
]

from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()

def is_image(path):
    return any(path.endswith(extension) for extension in IMG_EXTENSIONS)


def load_image(fname):
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Image file not found: {fname}")
    # implement option to load .heic files as opencv images ussing pillow_heif
    try:
        pil_img = Image.open(fname)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except ImportError:
        print("Pillow-heif not installed. Cannot open .heic files. Please install it with 'pip install pillow-heif'")
        return None


def undistort_functor(K, distCoeffs):
    def undistort(img):
        return cv2.undistort(img, K, distCoeffs)

    return undistort
