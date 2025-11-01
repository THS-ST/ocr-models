import os
import shutil

# ----- CHANGE THESE -----
source_dir ="data_png"      # folder with your PNG files
target_dir = "data_finalized"    # folder to save renamed PNGs
# -------------------------

# Create the destination folder if needed
os.makedirs(target_dir, exist_ok=True)

# Get only PNG files
images = [f for f in os.listdir(source_dir) if f.lower().endswith(".png")]

# Sort alphabetically (or numerically if filenames contain numbers)
images.sort()

# Copy + rename
for index, filename in enumerate(images, start=1):
    old_path = os.path.join(source_dir, filename)
    new_name = f"img_{index:02d}.png"
    new_path = os.path.join(target_dir, new_name)

    shutil.copy2(old_path, new_path)

print(f"✅ Done! {len(images)} PNG images copied and renamed.")
print(f"👉 Saved to: {target_dir}")
