import os
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_dataloaders
from model import build_model


CONFIG = {
    "data_dir":             "data",
    "output_dir":           "../outputs",
    "batch_size":           32,
    "num_workers":          4,

    "phase1_epochs":        10,
    "phase1_lr":            1e-3,

    "phase2_epochs":        20,
    "phase2_lr":            1e-4,
    "unfreeze_block":       6,

    "early_stop_patience":  5,
    "pos_weight":           2.5
}


def get_criterion(device: str) -> nn.BCEWithLogitsLoss:
    pos_weight = torch.tensor([CONFIG["pos_weight"]]).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def run_epoch(model, loader, criterion, optimizer, device, is_train: bool):
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


class EarlyStopping:
    def __init__(self, patience: int, save_path: str):
        self.patience = patience
        self.save_path = save_path
        self.best_loss = float("inf")
        self.counter = 0
        self.stopped = False

    def step(self, val_loss: float, model) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            print(f"  Best model saved (val_loss={val_loss:.4f})")
            return False
        else:
            self.counter += 1
            print(f"  No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.stopped = True
                return True

        return False


def train_phase(
        phase: int,
        model,
        loaders: dict,
        criterion,
        optimizer,
        scheduler,
        early_stop: EarlyStopping,
        epochs: int,
        device: str,
        history: list
):
    print(f"\n{'='*60}")
    print(f"  PHASE {phase} TRAINING | {epochs} epochs")
    print(f"{'='*60}")

    train_loader, n_train = loaders["train"]
    val_loader, n_val = loaders["val"]

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, is_train=False)

        scheduler.step()

        elapsed = time.time() - t0

        print(
            f"  Epoch {epoch:02d}/{epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
            f"[{elapsed:.1f}s]"
        )

        history.append({
            "phase": phase,
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4)
        })

        if early_stop.step(val_loss, model):
            print(f"  Early stopping triggered at epoch {epoch}")
            break

    return history


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = str(out_dir / "butyag_best.pth")
    history_path = str(out_dir / "training_history.json")

    print("\nLoading datasets...")
    loaders = get_dataloaders(
        data_dir=CONFIG["data_dir"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"]
    )

    print("\nBuilding model...")
    model = build_model(device=device, freeze_backbone=True)
    criterion = get_criterion(device)
    history = []
    early_stop = EarlyStopping(CONFIG["early_stop_patience"], best_model_path)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=CONFIG["phase1_lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["phase1_epochs"])

    history = train_phase(
        phase=1,
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stop=early_stop,
        epochs=CONFIG["phase1_epochs"],
        device=device,
        history=history
    )

    print("\nUnfreezing backbone for fine-tuning...")
    model.unfreeze_backbone(from_block=CONFIG["unfreeze_block"])
    early_stop.counter = 0

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=CONFIG["phase2_lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["phase2_epochs"])

    history = train_phase(
        phase=2,
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stop=early_stop,
        epochs=CONFIG["phase2_epochs"],
        device=device,
        history=history
    )

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining history saved -> {history_path}")
    print(f"Best model saved -> {best_model_path}")


if __name__ == '__main__':
    main()