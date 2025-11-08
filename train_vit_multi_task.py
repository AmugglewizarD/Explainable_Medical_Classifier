# train_vit_multi_task.py (modified for resuming 4 more epochs safely)
import time, os
from pathlib import Path
import torch
import torch.nn as nn, torch.optim as optim
from torch import amp
from tqdm import tqdm

from config import DEVICE, EPOCHS, LR, BATCH_SIZE, CHECKPOINT_DIR, USE_AMP
from data.dataset_loader import get_dataloader
from models.multitask_vit import MultiTaskViT

def train_one(task, dl, model, opt, crit, scaler):
    model.train()
    total = 0.0
    count = 0
    pbar = tqdm(dl, desc=f"Train {task}", leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        opt.zero_grad()
        with amp.autocast(device_type="cuda", enabled=USE_AMP):
            preds = model(imgs, task)
            loss = crit(preds, labels)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total += loss.item()
        count += 1
        pbar.set_postfix_str(f"loss={total/count:.4f}")
    return total / max(1, count)

def create_model_and_optim(n_xray, n_skin, n_mri):
    model = MultiTaskViT(n_xray, n_skin, n_mri)
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR)
    return model, opt

def save_checkpoint(epoch, model, opt, scaler, suffix, out_dir=CHECKPOINT_DIR):
    ckpt = {
        "epoch": epoch,
        "model_state": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "opt_state": opt.state_dict(),
        "scaler_state": scaler.state_dict()
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"vit_epoch{epoch:02d}_{suffix}.pt"
    torch.save(ckpt, path)
    print(f"✅ Saved checkpoint: {path}")

def main():
    print("Device:", DEVICE)
    xray_dl = get_dataloader("XRAY", BATCH_SIZE)
    skin_dl = get_dataloader("SKIN", BATCH_SIZE)
    mri_dl  = get_dataloader("MRI", BATCH_SIZE)

    nx = len(xray_dl.dataset.classes)
    ns = len(skin_dl.dataset.classes)
    nm = len(mri_dl.dataset.classes)

    model, opt = create_model_and_optim(nx, ns, nm)
    crit_x = nn.BCEWithLogitsLoss()
    crit_ce = nn.CrossEntropyLoss()
    scaler = amp.GradScaler() if USE_AMP else None

    # ============================
    # 🔁 RESUME LOGIC MODIFIED HERE
    # ============================
    # We resume from the latest checkpoint (expected vit_epoch04_xray_final.pt)
    ckpts = sorted(Path(CHECKPOINT_DIR).glob("vit_epoch*.pt"))
    start_epoch = 5   # <-- Default start for continuation
    if ckpts:
        latest = ckpts[-1]
        print("🔁 Resuming from", latest)
        ckpt = torch.load(latest, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        if scaler and "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt.get("epoch", 4) + 1
        print(f"✅ Loaded checkpoint up to epoch {start_epoch - 1}. Resuming at epoch {start_epoch}.")
    else:
        print("⚠️ No checkpoint found, training from scratch.")

    # Run exactly 4 more epochs
    end_epoch = start_epoch + 3
    print(f"Training from epoch {start_epoch} to {end_epoch}...")

    for e in range(start_epoch, end_epoch + 1):
        t0 = time.time()

        ls = train_one("SKIN", skin_dl, model, opt, crit_ce, scaler if scaler else amp.GradScaler(enabled=False))
        save_checkpoint(e, model, opt, scaler, "skin_done")

        lm = train_one("MRI", mri_dl, model, opt, crit_ce, scaler if scaler else amp.GradScaler(enabled=False))
        save_checkpoint(e, model, opt, scaler, "mri_done")

        lx = train_one("XRAY", xray_dl, model, opt, crit_x, scaler if scaler else amp.GradScaler(enabled=False))
        save_checkpoint(e, model, opt, scaler, "xray_final")

        print(f"Epoch {e}: XRAY={lx:.4f}, SKIN={ls:.4f}, MRI={lm:.4f}  time={(time.time()-t0)/60:.2f}min")

if __name__ == "__main__":
    main()
