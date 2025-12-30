import importlib
import warnings

from PIL import Image

_pytesseract = None


def _get_pytesseract():
    global _pytesseract
    if _pytesseract is None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*pkgutil.find_loader.*",
                category=DeprecationWarning,
                module="pytesseract",
            )
            _pytesseract = importlib.import_module("pytesseract")
    return _pytesseract


def ocr_image(path: str) -> str:
    pytesseract = _get_pytesseract()
    return pytesseract.image_to_string(Image.open(path), config="--oem 3 --psm 6")
