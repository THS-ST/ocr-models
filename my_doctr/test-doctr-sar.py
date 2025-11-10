import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor, sar_resnet31
from doctr.datasets import VOCABS

# --- Paths ---
RECO_MODEL_PATH = r"C:\Users\Shay\Documents\GitHub\ocr-models\doctr\sar_resnet31_20251108-132529.pt"
TEST_IMAGE_PATH = "../data/image_splits/testing_set_full/data_3.png"
VOCAB_KEY = "custom"

# --- Load SAR Recognition Model ---
reco_model = sar_resnet31(
    pretrained=False,
    pretrained_backbone=False,
    vocab=VOCABS[VOCAB_KEY]
)

print(f"Loading custom weights from: {RECO_MODEL_PATH}")
state = torch.load(RECO_MODEL_PATH, map_location="cpu")
reco_model.load_state_dict(state)
reco_model.eval()

# --- Build Predictor ---
predictor = ocr_predictor(
    det_arch="db_resnet50",
    reco_arch=reco_model,  # passing custom SAR model
    pretrained=True       # VERY IMPORTANT – do NOT load pretrained reco model
)

# --- Run Inference ---
doc = DocumentFile.from_images([TEST_IMAGE_PATH])
result = predictor(doc)

print("\n--- Extracted Text ---")
print(result.render())
