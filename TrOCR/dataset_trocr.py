

import os
import sys
import random
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from ocr_augmentation import (
    estimate_blur,          # blur metric
    choose_aug_count,       # decides 1/3/5 transforms
    AUGMENTATION_POOL,      # full pool (Albumentations transforms)
    SAFE_AUG_POOL,          # safer/minimal pool
    set_seed,               # seeds Python & NumPy
    EPOCH_MULTIPLIER,       # stride for per-index seeds
    binarize_image          # Otsu binarization
)


try:
    import albumentations as A
except Exception:
    A = None


def _load_df(csv_path: str) -> pd.DataFrame:
    # Try: comma CSV with headers
    df = pd.read_csv(csv_path)
    if df.shape[1] >= 2:
        return df

    df = pd.read_csv(csv_path, sep="\t", header=None,
                     names=["image_path", "transcription"])
    return df


class TrOcrCsvDataset(Dataset):
    """
    CSV → (pixel_values, labels) for TrOCR.
    - Deterministic, on-the-fly aug for training using your ocr_augmentation logic.
    - No augmentation for validation/test.
    """

    def __init__(
        self,
        csv_path: str,
        processor,                          # Hugging Face TrOCRProcessor
        image_base_dir: str | None = None,
        image_col: str = "image_path",
        text_col: str = "transcription",    # default matches your CSV
        aug_seed: int = 1,                  # everyone uses 1
        include_epoch_in_seed: bool = True,  # match “image + epoch” policy
        apply_binarize: bool = True,        # keep your binarize step
        augment_on: bool = True             # False for val/test
    ):
        self.df = _load_df(csv_path)
        self.processor = processor
        self.image_base_dir = image_base_dir
        self.image_col = image_col
        self.text_col = text_col

        self.aug_seed = int(aug_seed)
        self.include_epoch_in_seed = bool(include_epoch_in_seed)
        self.apply_binarize = bool(apply_binarize)
        self.augment_on = bool(augment_on) and (A is not None)

        self._epoch = 0  # set by training loop each epoch

        # sanity checks
        if self.image_col not in self.df.columns:
            raise ValueError(
                f"Missing column '{self.image_col}' in {csv_path}")
        if self.text_col not in self.df.columns:
            raise ValueError(f"Missing column '{self.text_col}' in {csv_path}")

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Resolve image path (relative to repo root if needed)
        img_path = row[self.image_col]
        if self.image_base_dir and not os.path.isabs(img_path):
            img_path = os.path.join(self.image_base_dir, img_path)

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        # ===== train-time deterministic augmentation =====
        if self.augment_on:
            blur_score = estimate_blur(img_bgr)
            k = choose_aug_count(blur_score)

            # seed = aug_seed + idx * EPOCH_MULTIPLIER (+ epoch if enabled)
            seed = self.aug_seed + idx * EPOCH_MULTIPLIER
            if self.include_epoch_in_seed:
                seed += self._epoch
            set_seed(seed)  # seeds Python & NumPy for deterministic sampling

            chosen_pool = SAFE_AUG_POOL if k == 1 else AUGMENTATION_POOL
            k = min(k, len(chosen_pool))

            # If Albumentations is available, apply sampled transforms
            if A is not None and k > 0:
                pipeline = A.Compose(random.sample(chosen_pool, k=k))
                img_bgr = pipeline(image=img_bgr)["image"]

        # Optional binarization (matches your post-augment behavior)
        if self.apply_binarize:
            img_bgr = binarize_image(img_bgr)  # returns single-channel
            # keep 3-channel for HF processor
            if len(img_bgr.shape) == 2:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

        # BGR → RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        text = str(row[self.text_col])

        # Encode with TrOCR processor
        encoded = self.processor(
            images=img_rgb,
            text=text,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}

        # Ignore padding in labels
        if "labels" in item:
            labels = item["labels"]
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            item["labels"] = labels

        return item
