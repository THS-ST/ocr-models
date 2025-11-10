import torch
import pandas as pd
from pathlib import Path
from doctr.io import DocumentFile
from doctr.models import ocr_predictor, recognition
from doctr.datasets import VOCABS
from jiwer import wer, cer


# --- Config ---
RECO_MODEL_PATH = r"C:\Users\Shay\Documents\GitHub\ocr-models\doctr\vitstr_base_20251109-032443.pt"
TEST_IMAGES_DIR = r"C:\Users\Shay\Documents\GitHub\ocr-models\data\image_splits\testing_set_full" 
LABELS_CSV = Path(TEST_IMAGES_DIR) / "labels.csv"  # CSV with columns: image, text
VOCAB_KEY = "custom"
SAVE_CSV_PATH = Path(TEST_IMAGES_DIR) / f"vitstr_base_evaluation.csv"


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
# Optional: limit for quick testing
# df_labels = df_labels.head(5)

results = []

for idx, row in df_labels.iterrows():
    img_path = Path(TEST_IMAGES_DIR) / row['filename']
    gt_text = row['text']

    # Load image and run prediction
    doc = DocumentFile.from_images([img_path])
    result = predictor(doc)
    pred_text = result.render()

    # Compute WER and CER
    file_wer = wer(gt_text, pred_text)
    file_cer = cer(gt_text, pred_text)
    exact_match = int(gt_text == pred_text)

    # Append result row
    results.append({
        "image": row['filename'],
        "ground_truth": gt_text,
        "prediction": pred_text,
        "CER": file_cer,
        "WER": file_wer,
        "exact": exact_match
    })

    print(f"\n--- {row['filename']} ---")
    print("GT:", gt_text)
    print("Pred:", pred_text)
    print(f"WER: {file_wer:.4f} | CER: {file_cer:.4f} | Exact: {exact_match}")

# --- Save results to CSV ---
df_results = pd.DataFrame(results)
df_results.to_csv(SAVE_CSV_PATH, index=False)
print(f"\nResults saved to: {SAVE_CSV_PATH}")

# --- Summary ---
avg_wer = df_results["WER"].mean()
avg_cer = df_results["CER"].mean()
accuracy = df_results["exact"].mean()

print("\n=== Summary ===")
print(f"Average WER: {avg_wer:.4f}")
print(f"Average CER: {avg_cer:.4f}")
print(f"Exact Match Accuracy: {accuracy:.4f}")
