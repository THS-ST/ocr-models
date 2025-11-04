# python create_lmdb_nested_csv.py "C:\Users\Shay\Documents\GitHub\ocr-models\output_lmdb" "C:\Users\Shay\Documents\GitHub\ocr-models\image_splits\training_set_splits" "C:\Users\Shay\Documents\GitHub\ocr-models\data\ground_truth_lines.csv"

import os
import sys
import lmdb
import cv2
import csv
import numpy as np
from tqdm import tqdm

def check_image_is_valid(image_bin):
    if image_bin is None:
        return False
    img = cv2.imdecode(np.frombuffer(image_bin, np.uint8), cv2.IMREAD_GRAYSCALE)
    return img is not None and img.size > 0

def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)

def list_all_images(root_dir, exts={'.png', '.jpg', '.jpeg'}):
    img_paths = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                img_paths.append(os.path.join(root, f))
    return img_paths

def create_lmdb(output_path, img_root, csv_path, csv_key="image_path", csv_label="transcription"):
    print(f"Reading truth file: {csv_path}")

    # Load CSV into dictionary
    truth = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row[csv_key].replace("\\", "/")  # normalize slashes
            truth[key] = row[csv_label]

    print(f"Loaded {len(truth)} label entries from CSV")

    print(f"Scanning for images under: {img_root}")
    all_imgs = list_all_images(img_root)

    print(f"Found {len(all_imgs)} total image files")

    env = lmdb.open(output_path, map_size=2 * 1024 * 1024 * 1024)
    cache = {}
    cnt = 1

    for img_path in tqdm(all_imgs, desc="Creating LMDB"):
        rel = os.path.relpath(img_path, img_root).replace("\\", "/")

        if rel not in truth:
            print(f"[WARN] No label found for: {rel}")
            continue

        label = truth[rel]
        with open(img_path, "rb") as f:
            img_bin = f.read()

        if not check_image_is_valid(img_bin):
            print(f"[SKIP] Invalid image: {img_path}")
            continue

        img_key = f"image-{cnt:09d}"
        label_key = f"label-{cnt:09d}"

        cache[img_key] = img_bin
        cache[label_key] = label.encode("utf-8")

        if cnt % 1000 == 0:
            write_cache(env, cache)
            cache.clear()

        cnt += 1

    cache["num-samples"] = str(cnt - 1).encode()
    write_cache(env, cache)
    env.close()

    print(f"\n✅ LMDB successfully created with {cnt - 1} samples at: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python create_lmdb_nested_csv.py <output_lmdb_path> <image_root_folder> <truth.csv>")
        sys.exit(1)

    output = sys.argv[1]
    image_root = sys.argv[2]
    csv_file = sys.argv[3]

    create_lmdb(output, image_root, csv_file)
