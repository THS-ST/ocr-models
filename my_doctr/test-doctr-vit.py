import torch
import pandas as pd
from pathlib import Path
from doctr.io import DocumentFile
from doctr.models import ocr_predictor, recognition
from doctr.datasets import VOCABS
from jiwer import wer, cer

# --- Config ---
RECO_MODEL_PATH = r"C:\Users\Shay\Documents\GitHub\ocr-models\doctr\vitstr_base_20251112-000553.pt"
TEST_IMAGES_DIR = r"C:\Users\Shay\Documents\GitHub\ocr-models\data\image_splits\testing_set_full" 
LABELS_CSV = Path(TEST_IMAGES_DIR) / "labels.csv"  # CSV: filename,text
VOCAB_KEY = "custom"

OUTPUT_CSV = "DocTR_evaluation-hail-mary.csv"  

# --- Load ViTSTR Recognition Model ---
reco_model = recognition.vitstr_base(
    pretrained=False,
    pretrained_backbone=False,
    vocab=VOCABS[VOCAB_KEY]
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

results = []   # ✅ where we'll store CSV rows

for idx, row in df_labels.iterrows():
    img_path = Path(TEST_IMAGES_DIR) / row['filename']
    gt_text = row['text']

    # Load image
    doc = DocumentFile.from_images([img_path])
    result = predictor(doc)

    # Extract text, sanitize newlines
    pred_text = result.render().replace("\n", " ").replace("\r", " ").replace("\t", " ")
    pred_text = " ".join(pred_text.split())

    # Metrics
    file_wer = wer(gt_text, pred_text)
    file_cer = cer(gt_text, pred_text)

    # Print
    print(f"\n--- {row['filename']} ---")
    print("GT:", gt_text)
    print("Pred:", pred_text)
    print(f"WER: {file_wer:.4f} | CER: {file_cer:.4f}")

    # ✅ Add to results list
    results.append({
        "image": row["filename"],
        "GT": gt_text,
        "pred": pred_text,
        "cer": file_cer,
        "wer": file_wer,
    })

# ✅ Save all results to CSV
pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved evaluation CSV → {OUTPUT_CSV}")
