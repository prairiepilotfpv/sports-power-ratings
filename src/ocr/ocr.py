import importlib
import warnings
import shutil
from pathlib import Path

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
            # If tesseract isn't on PATH, try common install locations (Windows)
            try:
                if shutil.which("tesseract") is None:
                    common = [
                        Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
                        Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
                        Path("C:/ProgramData/chocolatey/bin/tesseract.exe"),
                    ]
                    for p in common:
                        if p.exists():
                            # set pytesseract executable path if supported
                            try:
                                # some pytesseract versions expose nested attribute
                                if hasattr(_pytesseract, "pytesseract"):
                                    _pytesseract.pytesseract.tesseract_cmd = str(p)
                                else:
                                    _pytesseract.tesseract_cmd = str(p)
                            except Exception:
                                # best-effort: ignore if attribute doesn't exist
                                pass
                            break
            except Exception:
                pass
    return _pytesseract


def ocr_image(path: str) -> str:
    pytesseract = _get_pytesseract()
    return pytesseract.image_to_string(Image.open(path), config="--oem 3 --psm 6")
