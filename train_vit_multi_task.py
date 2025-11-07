import torch, time
import torch.nn as nn, torch.optim as optim
from torch import amp
from tqdm import tqdm
from pathlib import Path
from data.dataset_loader import get_dataloader
from models.multitask_vit import MultiTaskViT

# ==============================
# Config
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 6
LR = 2e-4
BATCH_SIZE = 8
USE_AMP = True
CHECKPOINT_DIR = Path("/kaggle/working/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# Training Step
# ==============================
def train_one(task, dl, model, opt, crit, scaler):
    model.train()
    total = 0
    for imgs, labels in tqdm(dl, desc=f"Training {task}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        with amp.autocast("cuda", enabled=USE_AMP):
            preds = model(imgs, task)
            loss = crit(preds, labels)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total += loss.item()
    return total / len(dl)

# ==============================
# Main Loop
# ==============================
def main():
    print("Device:", DEVICE)

    # Load data
    xray_dl = get_dataloader("XRAY", BATCH_SIZE)
    skin_dl = get_dataloader("SKIN", BATCH_SIZE)
    mri_dl  = get_dataloader("MRI",  BATCH_SIZE)

    # Determine number of classes
    nx, ns, nm = len(xray_dl.dataset.classes), len(skin_dl.dataset.classes), len(mri_dl.dataset.classes)
    print(f"Classes — XRAY: {nx}, SKIN: {ns}, MRI: {nm}")

    # Initialize model
    model = MultiTaskViT(nx, ns, nm)

    # Enable multi-GPU if available
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print(f"🧩 Using DataParallel on {device_count} GPUs.")
        model = torch.nn.DataParallel(model)
    model = model.to(DEVICE)

    # Optimizer and loss functions
    opt = optim.AdamW(model.parameters(), lr=LR)
    crit_x = nn.BCEWithLogitsLoss()
    crit_c = nn.CrossEntropyLoss()
    scaler = amp.GradScaler("cuda")

    # Resume from checkpoint if available
    RESUME_PATH = CHECKPOINT_DIR / "vit_epoch03.pt"  # example
    start_epoch = 1
    if RESUME_PATH.exists():
        print(f"🔄 Resuming from checkpoint: {RESUME_PATH}")
        ckpt = torch.load(RESUME_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
    else:
        print("🚀 Starting fresh training run...")

    # ==============================
    # Training loop
    # ==============================
    t0 = time.time()
    for e in range(start_epoch, EPOCHS + 1):
        print(f"
