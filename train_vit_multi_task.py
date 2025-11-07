import torch, time
import torch.nn as nn, torch.optim as optim
from torch import amp
from tqdm import tqdm
from data.dataset_loader import get_dataloader
from models.multitask_vit import MultiTaskViT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 6
LR = 2e-4
BATCH_SIZE = 8
USE_AMP = True
CHECKPOINT_DIR = "checkpoints"

def train_one(task, dl, model, opt, crit, scaler):
    model.train(); total=0
    for imgs, labels in tqdm(dl, desc=f"Training {task}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        with amp.autocast("cuda", enabled=USE_AMP):
            preds = model(imgs, task)
            loss = crit(preds, labels)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        total += loss.item()
    return total / len(dl)

def main():
    print("Device:", DEVICE)
    xray_dl = get_dataloader("XRAY", BATCH_SIZE)
    skin_dl = get_dataloader("SKIN", BATCH_SIZE)
    mri_dl  = get_dataloader("MRI",  BATCH_SIZE)
    nx, ns, nm = len(xray_dl.dataset.classes), len(skin_dl.dataset.classes), len(mri_dl.dataset.classes)
    model = MultiTaskViT(nx, ns, nm).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR)
    crit_x = nn.BCEWithLogitsLoss(); crit_c = nn.CrossEntropyLoss()
    scaler = amp.GradScaler("cuda")

    t0=time.time()
    for e in range(1,EPOCHS+1):
        print(f"\n=== Epoch {e}/{EPOCHS} ===")
        lx = train_one("XRAY",xray_dl,model,opt,crit_x,scaler)
        ls = train_one("SKIN",skin_dl,model,opt,crit_c,scaler)
        lm = train_one("MRI", mri_dl, model,opt,crit_c,scaler)
        print(f"Losses: XRAY={lx:.4f}, SKIN={ls:.4f}, MRI={lm:.4f}")
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/vit_epoch{e}.pt")
    print("Done in %.2f h"%((time.time()-t0)/3600))

if __name__=="__main__":
    main()
