import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor, crnn_vgg16_bn
from doctr.datasets import VOCABS
from jiwer import wer, cer
from pathlib import Path
import csv


# --- Configuration ---
RECO_MODEL_PATH = r"C:\Users\Shay\Documents\GitHub\ocr-models\doctr\sar_resnet31_20251108-132529.pt"
TEST_IMAGES_DIR = r"C:\Users\Shay\Documents\GitHub\ocr-models\data\image_splits\testing_set_full" 

VOCAB_KEY = "custom"  # Ensure this matches your training vocabulary
NUM_FILES = 5         # Number of files to evaluate

# --- 1. Load the Custom Recognition Model ---
reco_model = crnn_vgg16_bn(
    pretrained=False,
    pretrained_backbone=False,
    vocab=VOCABS[VOCAB_KEY]
)
print(f"Loading custom weights from: {RECO_MODEL_PATH}")
reco_model.load_state_dict(torch.load(RECO_MODEL_PATH, map_location="cpu"))
reco_model.eval()

# --- 2. Create the OCR Predictor Pipeline ---
predictor = ocr_predictor(
    det_arch="db_resnet50",  # default detector
    reco_arch=reco_model,    # pass your custom recognition model
    pretrained=True
)

# --- 3. Load the first N images in the directory ---
test_images_dir = Path(TEST_IMAGES_DIR)
image_files = sorted([f for f in test_images_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])[:NUM_FILES]

# --- 4. Ground truth labels ---
# You need a labels.txt file in the format: image_name|ground_truth_text
labels_path = test_images_dir / "labels.csv"
ground_truths = {}

with open(labels_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        img = row["filename"].strip()
        text = row["text"].replace("\n", " ").strip()
        ground_truths[img] = text


# --- 5. Run inference and compute WER/CER ---
all_wer = []
all_cer = []

for img_file in image_files:
    img_name = img_file.name
    gt_text = ground_truths.get(img_name, "")
    doc = DocumentFile.from_images([img_file])
    result = predictor(doc)
    
    # Extract predicted text
    pred_text = result.render().replace("\n", " ").strip()
    
    # Compute metrics
    sample_wer = wer(gt_text, pred_text)
    sample_cer = cer(gt_text, pred_text)
    all_wer.append(sample_wer)
    all_cer.append(sample_cer)
    
    # Display
    print(f"\n--- {img_name} ---")
    print(f"GT: {gt_text}")
    print(f"Pred: {pred_text}")
    print(f"WER: {sample_wer:.4f} | CER: {sample_cer:.4f}")

# --- 6. Summary ---
print("\n=== Summary for first {} files ===".format(len(image_files)))
print(f"Average WER: {sum(all_wer)/len(all_wer):.4f}")
print(f"Average CER: {sum(all_cer)/len(all_cer):.4f}")
