import os, csv, subprocess
from PIL import Image, ImageOps

# ==== CONFIG ====
MODEL = "eng_safemeds_e2_20k"
ROOT  = r"tess_temp\train_epoch2"            # traineddata filename (without .traineddata)
TESSDATA_DIR = r"models"
PSM = "13"                               # 13 for single-line crops
OUT_CSV = "eval_results.csv"

def levenshtein(a, b):
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    dp = list(range(lb+1))
    for i in range(1, la+1):
        prev, dp[0] = dp[0], i
        for j in range(1, lb+1):
            prev, dp[j] = dp[j], min(dp[j]+1, dp[j-1]+1, prev + (a[i-1]!=b[j-1]))
    return dp[lb]

def ocr(img_path):
    im = Image.open(img_path).convert("L")
    h = 56
    im = im.resize((int(im.width*h/im.height), h))
    im = ImageOps.autocontrast(im)
    tmp = img_path[:-4] + "_tmp.png"
    im.save(tmp)
    cmd = [
        "tesseract", tmp, "stdout",
        "--oem","1","--psm", PSM,
        "-l", MODEL, "--tessdata-dir", TESSDATA_DIR,
        "-c","preserve_interword_spaces=1",
        "-c","load_system_dawg=0","-c","load_freq_dawg=0",
        "-c",'tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#/:-()., '
    ]
    try:
        txt = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        txt = ""
    os.remove(tmp)
    return txt

pairs = []
for f in os.listdir(ROOT):
    if f.endswith(".gt.txt"):
        stem = f[:-7]  # remove ".gt.txt"
        img_path = os.path.join(ROOT, stem + ".png")
        gt_path  = os.path.join(ROOT, f)
        if os.path.exists(img_path):
            pairs.append((img_path, gt_path))


total_chars = total_words = err_chars = err_words = exact_hits = 0
rows = [["image","ground_truth","prediction","CER","WER","exact"]]

for img, gt_path in pairs:
    gt = open(gt_path, encoding="utf-8").read().strip()
    pr = ocr(img)
    ce = levenshtein(gt, pr)
    we = levenshtein(gt.split(), pr.split())
    cer = ce/len(gt)*100 if gt else 0
    wer = we/len(gt.split())*100 if gt.split() else 0
    rows.append([os.path.basename(img), gt, pr, f"{cer:.2f}", f"{wer:.2f}", int(gt==pr)])
    total_chars += len(gt)
    err_chars += ce
    total_words += len(gt.split())
    err_words += we
    exact_hits += int(gt == pr)

CER = err_chars / max(total_chars,1) * 100
WER = err_words / max(total_words,1) * 100
ACC = exact_hits / max(len(pairs),1) * 100

print(f"Samples: {len(pairs)}")
print(f"CER: {CER:.2f}%   WER: {WER:.2f}%   Exact line match: {ACC:.2f}%")

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(rows)
print(f"Saved detailed results to {OUT_CSV}")
