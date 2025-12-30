import os
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*pkgutil.find_loader.*",
    category=DeprecationWarning,
    module="pytesseract",
)

import pytesseract
from PIL import Image

def ocr_image(path: str) -> str:
    return pytesseract.image_to_string(Image.open(path), config="--oem 3 --psm 6")
