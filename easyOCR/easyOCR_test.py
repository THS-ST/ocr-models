import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from ocr_augmentation import DeterministicOCRDataset, final_transforms, CSV_PATH
import easyocr

# ------------------------
# 1. Config
# ------------------------
BATCH_SIZE = 16
NUM_WORKERS = 4
NUM_EPOCHS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # ------------------------
    # 2. Create Dataset + DataLoader
    # ------------------------
    augmentation_seed = 1  # Use same seed to reproduce augmentations
    train_dataset = DeterministicOCRDataset(
        csv_path=CSV_PATH,
        augmentation_seed=augmentation_seed,
        transform=final_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    # ------------------------
    # 3. Initialize EasyOCR model
    # ------------------------
    reader = easyocr.Reader(
        ['en'],
        gpu=(DEVICE=="cuda")
    )

    # ------------------------
    # 4. Training loop (simplified)
    # ------------------------
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch_idx, (images, labels) in enumerate(train_loader):
            images_np = [img.permute(1,2,0).cpu().numpy() for img in images]
            print(f"Batch {batch_idx+1}: {len(images_np)} images, {len(labels)} labels")

if __name__ == "__main__":
    main()
