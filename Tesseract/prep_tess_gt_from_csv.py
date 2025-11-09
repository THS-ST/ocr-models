import os, csv, shutil, argparse
from pathlib import Path
from PIL import Image
from augment import augment_pil, set_global_seed, BASE_AUG_SEED

def pick(row, *candidates, default=None):
    for k in candidates:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return default

def resolve_path(p: str) -> str:
    """Return an existing filesystem path for p (relative or absolute)."""
    if os.path.exists(p):
        return p
    # try relative to repo root
    alt = os.path.join(os.getcwd(), p)
    if os.path.exists(alt):
        return alt
    # try stripping leading "./"
    if p.startswith("./"):
        p2 = p[2:]
        if os.path.exists(p2):
            return p2
        alt2 = os.path.join(os.getcwd(), p2)
        if os.path.exists(alt2):
            return alt2
    return p  # will fail later, but we return for error reporting

def main():
    ap = argparse.ArgumentParser(description="Build Tesseract GT pairs (.png + .gt.txt) from CSV with deterministic aug.")
    ap.add_argument("--csv", required=True, help="Path to CSV (e.g., data/ground_truth_lines.csv)")
    ap.add_argument("--out_dir", default="tess_temp/train_epoch0")
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--subset", type=int, default=200, help="-1 = all rows")
    ap.add_argument("--start_index", type=int, default=0)
    args = ap.parse_args()

    set_global_seed(BASE_AUG_SEED)

    out = Path(args.out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            img_path = pick(row, "image_path", "path", "img_path", default=None)
            text = pick(row, "transcription", "text", "label", default="")
            if img_path is None:
                continue
            rows.append((img_path, text))

    if args.subset > -1:
        rows = rows[args.start_index : args.start_index + args.subset]

    if not rows:
        print("⚠️ No rows found from CSV. Check column names and filters.")
        return

    missing, written = 0, 0
    for i, (rel_path, text) in enumerate(rows):
        src = resolve_path(rel_path)
        try:
            with Image.open(src) as im:
                img = im.convert("L")
        except Exception as e:
            missing += 1
            if missing <= 5:
                print(f"❌ Missing/invalid image: {rel_path} (resolved: {src}) | {e}")
            continue

        img = augment_pil(img, idx=i, epoch=args.epoch)

        base = f"sample_{i:06d}"
        img.save(out / f"{base}.png")
        (out / f"{base}.gt.txt").write_text(text, encoding="utf-8")
        written += 1

        if written % 50 == 0:
            print(f"… wrote {written} pairs")

    print(f"✅ Done. Wrote {written} pairs to: {out}")
    if missing:
        print(f"⚠️ Skipped {missing} rows with missing/bad images.")

if __name__ == "__main__":
    main()
