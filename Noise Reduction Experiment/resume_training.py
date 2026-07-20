"""
resume_training.py -- continue training from saved checkpoint
Architecture: v5

Usage:
    python resume_training.py --list
    python resume_training.py LIPN 25 --epochs 20
    python resume_training.py DnCNN 15 --epochs 10
"""
import os, sys, argparse
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from denoising_experiment import (
    LIPN_Denoise, DnCNN, IRCNN,
    DenoisingTrainDataset, DenoisingTestDataset,
    train_one_epoch, evaluate_model, DEVICE,
    BSD400_DIR, BSD68_DIR, MODEL_DIR,
    PATCH_SIZE, BATCH_SIZE, LR_INIT, LR_MIN
)

def get_model(name):
    if name == "LIPN":  return LIPN_Denoise()
    if name == "DnCNN": return DnCNN()
    if name == "IRCNN": return IRCNN()
    raise ValueError(f"Unknown model: {name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", help="LIPN, DnCNN, or IRCNN")
    parser.add_argument("sigma", nargs="?", type=int, help="noise level")
    parser.add_argument("--epochs", type=int, default=20, help="additional epochs to train")
    parser.add_argument("--list", action="store_true", help="list resumable checkpoints")
    args = parser.parse_args()

    if args.list:
        print("\nResumable checkpoints:")
        for f in sorted(os.listdir(MODEL_DIR)):
            if f.endswith("_best.pth"):
                print(f"  {f}")
        return

    if not args.model or not args.sigma:
        print("Usage: python resume_training.py MODEL SIGMA --epochs N")
        return

    ckpt = os.path.join(MODEL_DIR, f"{args.model}_s{args.sigma}_best.pth")
    if not os.path.exists(ckpt):
        print(f"Not found: {ckpt}")
        return

    print(f"\nResuming {args.model} at sigma={args.sigma} for {args.epochs} more epochs")
    print(f"Loading {ckpt}...")

    model = get_model(args.model).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    train_set = DenoisingTrainDataset(BSD400_DIR, patch_size=PATCH_SIZE, sigma=args.sigma)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_set = DenoisingTestDataset(BSD68_DIR, sigma=args.sigma)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=LR_INIT*0.5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=LR_MIN)

    best_psnr, _ = evaluate_model(model, test_loader, "resume-check")
    print(f"  Current PSNR: {best_psnr:.2f} dB")
    model_name = f"{args.model}_s{args.sigma}"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, epoch, model_name)
        scheduler.step()
        val_psnr, val_ssim = evaluate_model(model, test_loader, model_name)
        print(f"  [{model_name}] Epoch {epoch:3d} | Loss={train_loss:.5f} | PSNR={val_psnr:.2f} | SSIM={val_ssim:.4f}")
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), ckpt)
            print(f"  >> Best model updated (PSNR={best_psnr:.2f})")

    print(f"\nDone. Best PSNR: {best_psnr:.2f} dB. Model saved to {ckpt}")

if __name__ == "__main__":
    main()
