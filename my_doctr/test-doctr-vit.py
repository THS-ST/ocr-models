import torch
import pandas as pd
from pathlib import Path
from doctr.io import DocumentFile
from doctr.models import ocr_predictor, recognition
from doctr.datasets import VOCABS
from jiwer import wer, cer

# --- Config ---
RECO_MODEL_PATH = r"C:\Users\Shay\Documents\GitHub\ocr-models\doctr\vitstr_base_20251109-025318.pt"
TEST_IMAGES_DIR = r"C:\Users\Shay\Documents\GitHub\ocr-models\data\image_splits\testing_set_full" 
LABELS_CSV = Path(TEST_IMAGES_DIR) / "labels.csv"  # CSV with columns: image, text
VOCAB_KEY = "custom"

# --- Load ViTSTR Recognition Model ---
reco_model = recognition.vitstr_base(
    pretrained=False,
    pretrained_backbone=False,
    vocab=VOCABS[VOCAB_KEY]  # Set to None to load custom vocab via checkpoint
)

print(f"Loading custom weights from: {RECO_MODEL_PATH}")
reco_model.load_state_dict(torch.load(RECO_MODEL_PATH, map_location="cpu"))
reco_model.eval()

# --- Create OCR Predictor ---
predictor = ocr_predictor(
    det_arch="db_resnet50",
    reco_arch=reco_model,
    pretrained=True
)

# --- Load Test Labels ---
df_labels = pd.read_csv(LABELS_CSV)
df_labels = df_labels.head(5)  # Only first 5 files for quick check

total_wer, total_cer = 0, 0

for idx, row in df_labels.iterrows():
    img_path = TEST_IMAGES_DIR / row['filename']
    gt_text = row['text']

    # Load image
    doc = DocumentFile.from_images([img_path])
    result = predictor(doc)

    # Extract text
    pred_text = result.render()
    
    # Compute WER and CER
    file_wer = wer(gt_text, pred_text)
    file_cer = cer(gt_text, pred_text)

    total_wer += file_wer
    total_cer += file_cer

    print(f"\n--- {row['image']} ---")
    print("GT:", gt_text)
    print("Pred:", pred_text)
    print(f"WER: {file_wer:.4f} | CER: {file_cer:.4f}")

print("\n=== Summary for first 5 files ===")
print(f"Average WER: {total_wer/len(df_labels):.4f}")
print(f"Average CER: {total_cer/len(df_labels):.4f}")
