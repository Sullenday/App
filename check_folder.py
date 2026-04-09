import argparse
from pathlib import Path

import cv2

from app import container_ocr

ALLOWED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_images(folder: Path):
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in ALLOWED:
            yield path


def main():
    parser = argparse.ArgumentParser(description="Quick local check for container number OCR")
    parser.add_argument("folder", help="Folder with images")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")

    total = 0
    for path in iter_images(folder):
        img = cv2.imread(str(path))
        text, score, is_valid = container_ocr.ocr_container_best(img)
        total += 1
        print(f"{path.name}\t{text}\tscore={score:.4f}\tiso6346={is_valid}")

    print(f"Checked files: {total}")


if __name__ == "__main__":
    main()
