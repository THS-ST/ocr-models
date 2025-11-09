
import os
import argparse
import torch
from jiwer import cer, wer
from PIL import Image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.cuda.amp import GradScaler
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    get_linear_schedule_with_warmup,
)

from dataset_trocr import TrOcrCsvDataset


def parse_args():
    ap = argparse.ArgumentParser(
        "Train TrOCR (deterministic on-the-fly augmentation)")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", default=None)
    ap.add_argument("--image_base_dir", default=".")
    ap.add_argument("--image_col", default="image_path")

    ap.add_argument("--text_col", default="transcription")
    ap.add_argument("--pretrained", default="microsoft/trocr-base-stage1")
    ap.add_argument("--out_dir", default="runs/trocr")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--fp16", action="store_true")

    # augmentation controls
    ap.add_argument("--aug_seed", type=int, default=1)
    ap.add_argument("--include_epoch_in_seed", action="store_true")
    ap.add_argument("--binarize", action="store_true")

    ap.add_argument("--max_train_steps", type=int, default=None)
    
    # parse_args():
    ap.add_argument("--no_train_aug", action="store_true")
    ap.add_argument("--no_binarize", action="store_true")  # optional, to override



    return ap.parse_args()


def collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def make_loader(csv, processor, args, train=True):
    ds = TrOcrCsvDataset(
        csv_path=csv,
        processor=processor,
        image_base_dir=args.image_base_dir,
        image_col=args.image_col,
        text_col=args.text_col,
        aug_seed=args.aug_seed,
        include_epoch_in_seed=args.include_epoch_in_seed,
        apply_binarize=(False if args.no_binarize else args.binarize),
        augment_on=(train and not args.no_train_aug),
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return ds, dl


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Device & AMP

    has_mps = getattr(torch.backends, "mps",
                      None) and torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        device = torch.device("cuda")
    elif has_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    use_amp = bool(args.fp16 and use_cuda)
    print(f"[device] {device} | amp={use_amp}")

    # Processor & Model
    processor = TrOCRProcessor.from_pretrained(args.pretrained, use_fast=False)
    processor.tokenizer.model_max_length = 256
    model = VisionEncoderDecoderModel.from_pretrained(args.pretrained)

    # ---- Ensure config has required IDs (stage1 lacks these) ----
    if getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
    # (Optional) keep config vocab_size in sync with decoder
    if hasattr(model.config, "vocab_size") and hasattr(model, "decoder") and hasattr(model.decoder, "config"):
        model.config.vocab_size = model.decoder.config.vocab_size

    for p in model.encoder.parameters():
        p.requires_grad = False
    freeze_encoder_epochs = 0
    
    #model.gradient_checkpointing_enable()
    model.to(device)

    # Datasets / Loaders
    train_ds, train_dl = make_loader(
        args.train_csv, processor, args, train=True)
    val_ds, val_dl = (None, None)
    if args.val_csv:
        val_ds, val_dl = make_loader(
            args.val_csv, processor, args, train=False)

    # Optimizer / Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(grouped, lr=args.lr)
    total_steps = args.max_train_steps or (len(train_dl) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )
    scaler = GradScaler(enabled=use_amp)

    # Training loop
    global_step = 0
    for epoch in range(args.epochs):
        
        if epoch == freeze_encoder_epochs:
            for p in model.encoder.parameters():
                p.requires_grad = True
        print("[info] unfroze encoder parameters")
        
        
        model.train()
        train_ds.set_epoch(epoch)

        running = 0.0
        for step, batch in enumerate(train_dl):
            if args.max_train_steps and global_step >= args.max_train_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast(device_type="cuda", enabled=use_amp):
                outputs = model(
                    pixel_values=batch["pixel_values"], labels=batch["labels"])
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            global_step += 1
            running += loss.item()

            if step % 50 == 0:
                avg = running / (step + 1)
                print(
                    f"epoch {epoch} step {step}/{len(train_dl)} loss {avg:.4f}")

        # Validation (if provided)
        if val_dl is not None:
            model.eval()
            total_loss, cnt = 0.0, 0
            preds, gts = [], []

            with torch.no_grad():
                for vb in val_dl:
                    vb = {k: v.to(device) for k, v in vb.items()}
                    out = model(
                        pixel_values=vb["pixel_values"], labels=vb["labels"])
                    total_loss += out.loss.item()
                    cnt += 1

                    ids = model.generate(
                        pixel_values=vb["pixel_values"],
                        num_beams=10, do_sample=False, max_new_tokens=64,
                        no_repeat_ngram_size=4, repetition_penalty=1.5,
                        length_penalty=0.8, early_stopping=True,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        pad_token_id=processor.tokenizer.pad_token_id,
                    )
                    preds += processor.batch_decode(ids, skip_special_tokens=True)
                    lbl = vb["labels"].clone()
                    lbl[lbl == -100] = processor.tokenizer.pad_token_id
                    gts  += processor.batch_decode(lbl, skip_special_tokens=True)

            val_loss = total_loss / max(1, cnt)
            print(f"[val] epoch {epoch} loss {val_loss:.4f}")

            if preds:
                print(
                    f"[val] CER {cer(gts, preds):.4f} | WER {wer(gts, preds):.4f}")

        # Save checkpoint each epoch
        save_dir = os.path.join(args.out_dir, f"epoch_{epoch+1}")
        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
        print(f"[save] {save_dir}")

    print("[done]")


if __name__ == "__main__":
    main()
