import os, math, argparse, pandas as pd
from PIL import Image
import torch
from jiwer import cer as jiwer_cer, wer as jiwer_wer
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def avg_token_conf(scores):
    """
    Average per-step max prob for the TOP beam.
    Works for num_beams >= 1. For greedy, it's identical.
    """
    if not scores:
        return float("nan")
    vals = []
    for step_logits in scores:
        # shape: (batch=1, vocab) if greedy; (beams, vocab) if beam search
        probs = step_logits.softmax(dim=-1).max(dim=-1).values  # (1,) or (beams,)
        # take the top beam's prob (index 0) if there are multiple
        if probs.numel() > 1:
            vals.append(probs[0].item())
        else:
            vals.append(float(probs.item()))
    return float(sum(vals) / len(vals)) if vals else float("nan")


def resolve_path(p: str, base_dir: str) -> str:
    p = str(p).strip()
    if not p:
        return p
    if os.path.isabs(p):
        return p
    # Common normalizations
    if p.startswith("./"):
        p = p[2:]
    if p.startswith("image_splits/"):
        p = os.path.join("data", p)
    if p.startswith("./data/"):
        p = p[2:]
    # Final resolution
    cand = p if os.path.isfile(p) else os.path.join(base_dir, p)
    # If still missing and doesn’t start with data/, try adding it
    if not os.path.isfile(cand) and not p.startswith("data/"):
        cand2 = os.path.join(base_dir, "data", p)
        if os.path.isfile(cand2):
            cand = cand2
    return cand

def main():
    ap = argparse.ArgumentParser("Evaluate a TrOCR checkpoint on a CSV list")
    ap.add_argument("--csv", required=True,
                    help="CSV with columns: image_path, gt")
    ap.add_argument("--ckpt", required=True,
                    help="Local checkpoint folder (contains model/processor)")
    ap.add_argument("--base_dir", default=".",
                    help="Prefix for resolving relative image paths")
    ap.add_argument("--out_csv", default=None,
                    help="Output CSV path (default: overwrite --csv)")
    ap.add_argument("--num_beams", type=int, default=8)
    args = ap.parse_args()

    out_csv = args.out_csv or args.csv

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
              else "cpu")
    print(f"[device] {device}")

    # Load processor & model from LOCAL checkpoint
    proc  = TrOCRProcessor.from_pretrained(args.ckpt, use_fast=False, local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(args.ckpt, local_files_only=True).to(device).eval()

    # Safety: make sure special tokens exist
    if getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = proc.tokenizer.bos_token_id
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = proc.tokenizer.pad_token_id

    # Read CSV
    df = pd.read_csv(args.csv)
    needed = {"image_path", "gt"}
    if not needed.issubset(df.columns):
        raise SystemExit(f"{args.csv} must contain columns: {needed}")

    # Preflight: check which images resolve
    paths = [resolve_path(p, args.base_dir) for p in df["image_path"]]
    resolvable = sum(os.path.isfile(p) for p in paths)
    print(f"[preflight] rows={len(df)} | resolvable images={resolvable} | missing={len(df)-resolvable}")

    # Prepare output columns (avoid dtype warnings)
    df["pred"] = ""
    df["confidence"] = pd.Series([float("nan")] * len(df), dtype="float64")
    df["wer"] = pd.Series([float("nan")] * len(df), dtype="float64")
    df["cer"] = pd.Series([float("nan")] * len(df), dtype="float64")

    kept_idx, preds, gts, cers, wers = [], [], [], [], []

    for i, (img_rel, gt_text, full) in enumerate(zip(df["image_path"], df["gt"], paths)):
        if not os.path.isfile(full):
            continue  # leave NaNs
        im = Image.open(full).convert("RGB")
        inp = proc(images=im, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inp,
                num_beams=args.num_beams,
                do_sample=False,
                max_new_tokens=64,
                no_repeat_ngram_size=4,
                repetition_penalty=1.5,
                length_penalty=0.8,
                output_scores=True,
                return_dict_in_generate=True,
                eos_token_id=proc.tokenizer.eos_token_id,
                pad_token_id=proc.tokenizer.pad_token_id,
            )

        pred = proc.batch_decode(out.sequences, skip_special_tokens=True)[0]
        conf = avg_token_conf(out.scores)

        # per-sample metrics
        gt = str(gt_text)
        s_cer = jiwer_cer([gt], [pred])
        s_wer = jiwer_wer([gt], [pred])

        df.at[i, "pred"] = pred
        df.at[i, "confidence"] = float(conf)
        df.at[i, "cer"] = float(s_cer)
        df.at[i, "wer"] = float(s_wer)

        kept_idx.append(i)
        preds.append(pred)
        gts.append(gt)
        cers.append(s_cer)
        wers.append(s_wer)

        if (len(kept_idx) % 50) == 0:
            print(f"[progress] {len(kept_idx)} samples evaluated")

    # Overall metrics
    if kept_idx:
        overall_cer = jiwer_cer(gts, preds)
        overall_wer = jiwer_wer(gts, preds)
        char_acc = (1.0 - overall_cer) * 100.0
    else:
        overall_cer = float("nan")
        overall_wer = float("nan")
        char_acc = float("nan")

    print(f"samples evaluated: {len(kept_idx)} / {len(df)}")
    print(f"Overall CER: {overall_cer:.4f} | WER: {overall_wer:.4f}")
    print(f"Overall character accuracy: {char_acc:.2f}%")

    df.to_csv(out_csv, index=False)
    print(f"Wrote results to: {out_csv}")

if __name__ == "__main__":
    main()
