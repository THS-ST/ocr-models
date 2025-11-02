import os
from pdf2image import convert_from_path
from PIL import Image

# --- SETTINGS ---
input_folder = "synthetic_data_raw"       # folder containing PDFs and JPGs
output_folder = "synthetic_data_png" # all PNGs will go here

poppler_path = r"C:\Poppler\poppler-25.07.0\Library\bin"

os.makedirs(output_folder, exist_ok=True)

counter = 1  # global counter for unique filenames

for file in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file)
    if not os.path.isfile(file_path):
        continue

    filename, ext = os.path.splitext(file)
    ext = ext.lower()

    # --- Handle PDFs ---
    if ext == ".pdf":
        images = convert_from_path(file_path, dpi=200, poppler_path=poppler_path)
        for i, img in enumerate(images, start=1):
            out_path = os.path.join(output_folder, f"data_{counter}.png")
            img.save(out_path, "PNG")
            counter += 1
        print(f"{file} → {len(images)} page(s) converted")

    # --- Handle JPG / JPEG / JFIF ---
    elif ext in [".jpg", ".jpeg", ".jfif"]:
        try:
            img = Image.open(file_path).convert("RGB")
            out_path = os.path.join(output_folder, f"data_{counter}.png")
            img.save(out_path, "PNG")
            counter += 1
            print(f"🖼️ {file} → converted to PNG")
        except Exception as e:
            print(f"⚠️ Failed to convert {file}: {e}")

print(f"Done! {counter - 1} total PNGs saved in '{output_folder}'")
