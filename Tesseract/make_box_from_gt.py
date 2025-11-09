# Generates a .box file that EXACTLY matches the .gt.txt text by
# evenly distributing character boxes across the image width.

import sys, os, glob
from PIL import Image

def write_box_for_line(img_path):
    base = os.path.splitext(img_path)[0]
    gt_path = base + ".gt.txt"
    box_path = base + ".box"

    if not os.path.exists(gt_path):
        return False, f"Missing GT: {gt_path}"

    with open(gt_path, "r", encoding="utf-8") as f:
        text = f.read().replace("\r", "").replace("\n", " ").strip()

    if text == "":
        return False, f"Empty GT: {gt_path}"

    with Image.open(img_path) as im:
        w, h = im.size

    # margins and vertical placement
    lm, rm = int(0.02 * w), int(0.02 * w)
    top, bottom = int(0.15 * h), int(0.85 * h)

    n = len(text)
    # avoid div-by-zero for single char
    step = (w - lm - rm) / max(n, 1)

    lines = []
    x = lm
    for ch in text:
        x2 = x + step
        # clamp & int
        l = max(0, int(x))
        r = min(w - 1, int(x2))
        t = max(0, top)
        b = min(h - 1, bottom)
        # format: <char> <l> <t> <r> <b> 0
        # Tesseract uses origin at bottom-left internally, but for LSTM training
        # box files with image coords (top-down) work fine for line training.
        lines.append(f"{ch} {l} {t} {r} {b} 0")
        x = x2

    with open(box_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return True, f"Wrote {box_path} ({n} chars, {w}x{h})"

def main(folder):
    imgs = sorted(glob.glob(os.path.join(folder, "*.png")))
    ok, bad = 0, 0
    for p in imgs:
        success, msg = write_box_for_line(p)
        if success:
            ok += 1
        else:
            bad += 1
        print(msg)
    print(f"Done. OK={ok}, Skipped={bad}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python make_box_from_gt.py <folder_with_png_gt>")
        sys.exit(1)
    main(sys.argv[1])
