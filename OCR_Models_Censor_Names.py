import os
import re
import cv2
import pytesseract
from pytesseract import Output

# --- CONFIG ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- REGEX PATTERNS ---
# Handles:
# - Patient Maria Santos
# - Name: Juan Cruz
# - Dr. Ana Lopez
# - Ana Lopez M.D. / MD
NAME_PATTERNS = [
    r'\bpatient[:\s]+[A-Z][a-z]+(?:\s[A-Z][a-z]+)*',   # Patient Maria Santos
    r'\bname[:\s]+[A-Z][a-z]+(?:\s[A-Z][a-z]+)*',      # Name: Juan Cruz
    r'\bdr\.?\s*[A-Z][a-z]+(?:\s[A-Z][a-z]+)*',        # Dr. Ana Lopez
    r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\s*M\.?D\.?\b',   # Ana Lopez M.D. or MD
]
CENSOR_PATTERNS = [re.compile(p, re.IGNORECASE) for p in NAME_PATTERNS]


def censor_image(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Cannot open {image_path}")
        return

    data = pytesseract.image_to_data(img, output_type=Output.DICT)
    num_boxes = len(data['text'])

    censored = img.copy()

    for i in range(num_boxes):
        # Combine text in the current line for better pattern matching
        line_text = " ".join(
            w for j, w in enumerate(data['text'])
            if data['block_num'][j] == data['block_num'][i]
            and data['par_num'][j] == data['par_num'][i]
            and data['line_num'][j] == data['line_num'][i]
        ).strip()

        # Check against all patterns
        if any(p.search(line_text) for p in CENSOR_PATTERNS):
            # Get full line bounding box
            indices = [
                j for j in range(num_boxes)
                if data['block_num'][j] == data['block_num'][i]
                and data['par_num'][j] == data['par_num'][i]
                and data['line_num'][j] == data['line_num'][i]
            ]
            x_min = min(data['left'][j] for j in indices)
            y_min = min(data['top'][j] for j in indices)
            x_max = max(data['left'][j] + data['width'][j] for j in indices)
            y_max = max(data['top'][j] + data['height'][j] for j in indices)

            # Draw black rectangle over detected sensitive line
            cv2.rectangle(censored, (x_min, y_min), (x_max, y_max), (0, 0, 0), -1)

    # Save censored output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, censored)
    print(f"✅ Censored saved: {output_path}")


def censor_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for file in os.listdir(input_folder):
        if file.lower().endswith(".png"):
            in_path = os.path.join(input_folder, file)
            out_path = os.path.join(output_folder, file)
            censor_image(in_path, out_path)

if __name__ == "__main__":
    censor_folder("data_png", "data_censored")
    print("🎉 Done! All sensitive names have been censored.")
