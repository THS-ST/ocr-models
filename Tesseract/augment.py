import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter

# Shared across the whole team
BASE_AUG_SEED = 1
EPOCH_MULTIPLIER = 10_000  # keeps per-epoch seeds far apart

def set_global_seed(seed: int = BASE_AUG_SEED):
    random.seed(seed); np.random.seed(seed)

def _per_sample_epoch_seed(idx: int, epoch: int, base_seed: int = BASE_AUG_SEED):
    return (idx * EPOCH_MULTIPLIER) + epoch + base_seed

def augment_pil(img: Image.Image, idx: int, epoch: int) -> Image.Image:
    """
    Deterministic, OCR-safe light augmentation based on (idx, epoch, BASE_AUG_SEED).
    """
    rng = random.Random(_per_sample_epoch_seed(idx, epoch))

    # 1) small rotate
    angle = rng.uniform(-2.0, 2.0)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)

    # 2) tiny random crop then resize back (simulate framing jitter)
    w, h = img.size
    max_crop = int(0.03 * min(w, h))
    l = rng.randint(0, max_crop)
    t = rng.randint(0, max_crop)
    r = rng.randint(0, max_crop)
    b = rng.randint(0, max_crop)
    img = img.crop((l, t, w - r, h - b)).resize((w, h), Image.BICUBIC)

    # 3) light contrast/brightness jitter
    img = ImageOps.autocontrast(img, cutoff=rng.uniform(0, 2))
    c = 1.0 + rng.uniform(-0.1, 0.1)
    b = rng.uniform(-10, 10)
    img = Image.eval(img, lambda px: max(0, min(255, int(px * c + b))))

    # 4) optional blur / tiny gaussian noise
    if rng.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))
    if rng.random() < 0.35:
        arr = np.array(img).astype(np.int16)
        noise = rng.gauss(0, 6)  # sigma ~ 6/255
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img
