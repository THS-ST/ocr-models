import os
import csv
import random
import numpy as np
import cv2
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ==========================
# 1. CONFIG & UTILITIES
# ==========================
# Placeholder paths - adjust as needed
CSV_PATH = "data/ground_truth_lines.csv"
EPOCH_MULTIPLIER = 1000 # Used to ensure unique seeds across epochs

def set_seed(seed):
    """Sets seeds for Python, NumPy, and Albumentations."""
    random.seed(seed)
    np.random.seed(seed)

def estimate_blur(img):
    """Calculates Laplacian variance for blur score."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def choose_aug_count(blur_score):
    """Determines the number of random augmentations to apply."""
    if blur_score > 200:
        return 5    # clean / sharp → augment more
    elif blur_score > 80:
        return 3    # moderate quality → medium augmentation
    else:
        return 1    # low quality → minimal augmentation
    
def binarize_image(img):
    """Applies binarization (Otsu's method) to the image."""
    # Ensure image is in grayscale for thresholding
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
        
    # Apply Otsu's thresholding
    # The output is a single-channel (grayscale) binary image
    _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary_img

# ==========================
# 2. AUGMENTATION POOL
# ==========================
# Full pool of all possible augmentations (p=1.0 as the random selection controls probability)
AUGMENTATION_POOL = [
    A.HorizontalFlip(p=1.0),
    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
    A.MotionBlur(blur_limit=5, p=1.0),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
    A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
    A.Perspective(scale=(0.02, 0.05), p=1.0),
    A.Affine(translate_percent=(0.0625,0.0625), scale=(0.9,1.1), rotate=(-15,15), p=1.0)
]

# Define a safe pool to exclude harsh transforms for the lowest quality tier
SAFE_AUG_TYPES = (A.GaussianBlur, A.MotionBlur, A.GaussNoise)
SAFE_AUG_POOL = [aug for aug in AUGMENTATION_POOL if not isinstance(aug, SAFE_AUG_TYPES)]


# ==========================
# 3. CUSTOM PYTORCH DATASET
# ==========================
class DeterministicOCRDataset(Dataset):
    def __init__(self, csv_path, augmentation_seed, transform=None):
        self.transform = transform
        self.augmentation_seed = augmentation_seed
        self.data = []
        
        # Load all image paths and transcriptions
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.data = list(reader)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        img_path = row["image_path"]
        transcription = row["transcription"]
        
        # 1. Load Image (using cv2 for Albumentations compatibility)
        img = cv2.imread(img_path)
        if img is None:
            # Handle missing or unreadable images
            print(f"⚠️ Unable to read image: {img_path}")
            return self.__getitem__(random.randint(0, len(self) - 1)) # Load a random image instead

        # 2. DETERMINE AUGMENTATION COUNT (Blur-Aware Logic)
        blur_score = estimate_blur(img)
        num_aug = choose_aug_count(blur_score)
        
        # 3. DETERMINE THE UNIQUE SEED
        # The seed combines the image index and the current epoch for determinism
        # This ensures the same augmented image is created every time for this (index, epoch) pair
        SEED = (index * EPOCH_MULTIPLIER) + self.augmentation_seed
        set_seed(SEED)
        
        # 4. RANDOM SELECTION & COMPOSITION
        if num_aug == 1:
            # Use the safe pool for minimal augmentation
            selected_augs = random.sample(SAFE_AUG_POOL, k=min(num_aug, len(SAFE_AUG_POOL)))
        else:
            # Use the full pool for moderate/high augmentation
            selected_augs = random.sample(AUGMENTATION_POOL, k=min(num_aug, len(AUGMENTATION_POOL)))
            
        pipeline = A.Compose(selected_augs)
        
        # 5. Apply Augmentation
        augmented_img = pipeline(image=img)["image"]
        augmented_img = binarize_image(augmented_img)

        # 6. Apply Final Tensor Transformations (e.g., resizing, normalization)
        if self.transform:
            augmented_img = self.transform(augmented_img)
            
        # NOTE: You will need to implement target encoding for the 'transcription'
        # For simplicity, we just return the text here.
        return augmented_img, transcription

# ==========================
# 4. USAGE EXAMPLE
# ==========================
#Final PyTorch transforms (applied after Albumentations)
final_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 256)),  # Example resizing for OCR model input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# In your training loop (e.g., for TrOCR or EasyOCR)
# augmentation_seed = 1 
# train_dataset = DeterministicOCRDataset(
#     csv_path=CSV_PATH, 
#     augmentation_seed=augmentation_seed, 
#     transform=final_transforms
# )
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

# To train the models with identical augmented data, ensure both training
# scripts use this exact Dataset class implementation and are initialized with the 
# same `augmentation_seed` value before creating the DataLoader.