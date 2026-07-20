"""
LIPN Denoising Experiment (clean v6)
========================================
Compares LIPN vs DnCNN vs IRCNN vs BM3D on BSD68 at sigma=15/25/50.

Architecture:
  G_theta(z) = z + H(z), tail zero-inited (identity at start)
  LIPN: K=5 iterations, learned eta only, pure L1 loss
  DnCNN: 17-layer CNN, residual learning
  IRCNN: 7-layer dilated CNN, residual learning

Usage:
  python denoising_experiment.py --mode quick      # 30 epochs, all sigma
  python denoising_experiment.py --mode dryrun      # 1 epoch, sigma=25 only
  python denoising_experiment.py --mode eval --sigma 15 25 50
"""

import os, sys, argparse, time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import bm3d

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============================================================
# 0. CONFIG
# ============================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "denoising-datasets-main")
BSD400_DIR = os.path.join(DATA_DIR, "BSD400")
BSD68_DIR = os.path.join(DATA_DIR, "BSD68")
RESULTS_DIR = os.path.join(ROOT, "denoising_results")
MODEL_DIR = os.path.join(RESULTS_DIR, "models")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 64
BATCH_SIZE = 6
LR_INIT = 3e-5
LR_MIN = 1e-6
NUM_ITER = 5
INNER_FEATS = 64
NUM_BLOCKS = 4
NOISE_LEVELS = [15, 25, 50]


# Reproducibility
SEED = 13
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)

print(f"Device: {DEVICE}")


# ============================================================
# 1. DATA
# ============================================================

class TrainDataset(Dataset):
    """BSD400: random 64x64 patches + augment + online noise."""
    def __init__(self, data_dir, sigma, oversample=60):
        self.data_dir = data_dir
        self.sigma = sigma / 255.0       # noise std in [0,1]
        self.patch_size = PATCH_SIZE
        self.oversample = oversample
        self.files = sorted([f for f in os.listdir(data_dir)
                             if f.endswith(('.png','.jpg','.jpeg','.bmp'))])

    def __len__(self):
        return len(self.files) * self.oversample

    def __getitem__(self, idx):
        img = np.array(Image.open(os.path.join(self.data_dir,
                          self.files[idx % len(self.files)])).convert('L'),
                       dtype=np.float32) / 255.0
        h, w = img.shape
        if h < self.patch_size or w < self.patch_size:
            img = np.pad(img, ((0, max(0, self.patch_size - h)),
                               (0, max(0, self.patch_size - w))), mode='reflect')
            h, w = img.shape
        i = np.random.randint(0, h - self.patch_size + 1)
        j = np.random.randint(0, w - self.patch_size + 1)
        patch = img[i:i+self.patch_size, j:j+self.patch_size].copy()
        # Augment
        if np.random.rand() > 0.5: patch = np.fliplr(patch).copy()
        if np.random.rand() > 0.5: patch = np.flipud(patch).copy()
        k = np.random.randint(0, 4)
        if k > 0: patch = np.ascontiguousarray(np.rot90(patch, k))
        # Noise
        noise = np.random.randn(*patch.shape).astype(np.float32) * self.sigma
        noisy = np.clip(patch + noise, 0, 1)
        return (torch.from_numpy(noisy).unsqueeze(0),
                torch.from_numpy(patch).unsqueeze(0))


class TestDataset(Dataset):
    """BSD68: pre-computed noisy images."""
    def __init__(self, sigma):
        self.clean_d = os.path.join(BSD68_DIR, "original")
        self.noisy_d = os.path.join(BSD68_DIR, f"noise{sigma}")
        self.files = sorted([f for f in os.listdir(self.clean_d)
                             if f.endswith(('.png','.jpg'))])

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        fn = self.files[i]
        c = np.array(Image.open(os.path.join(self.clean_d, fn)).convert('L'), np.float32) / 255.
        n = np.array(Image.open(os.path.join(self.noisy_d, fn)).convert('L'), np.float32) / 255.
        h, w = c.shape
        ph, pw = (8-h%8)%8, (8-w%8)%8
        c = np.pad(c, ((0,ph),(0,pw)), 'reflect')
        n = np.pad(n, ((0,ph),(0,pw)), 'reflect')
        return (torch.from_numpy(np.ascontiguousarray(n)).unsqueeze(0),
                torch.from_numpy(np.ascontiguousarray(c)).unsqueeze(0), (h,w))


# ============================================================
# 2. MODELS
# ============================================================

class ResBlock(nn.Module):
    """Conv-BN-ReLU-Conv-BN with skip connection."""
    def __init__(self, feats):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(feats, feats, 3, padding=1, bias=False),
            nn.BatchNorm2d(feats), nn.ReLU(inplace=True),
            nn.Conv2d(feats, feats, 3, padding=1, bias=False),
            nn.BatchNorm2d(feats),
        )
    def forward(self, x): return x + self.body(x)


class Denoiser(nn.Module):
    """G_theta(z) = z + H(z). Tail zero-inited so G starts as identity."""
    def __init__(self, in_ch=1, feats=INNER_FEATS, blocks=NUM_BLOCKS):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(in_ch, feats, 3, padding=1), nn.ReLU(inplace=True))
        self.body = nn.Sequential(*[ResBlock(feats) for _ in range(blocks)])
        self.tail = nn.Conv2d(feats, in_ch, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, z):
        return z + self.tail(self.body(self.head(z)))


class LIPN_Denoise(nn.Module):
    """LIPN: z_{k+1} = G_theta(z_k - eta_k * F(z_k)), F(z) = z - y."""
    def __init__(self, K=NUM_ITER):
        super().__init__()
        self.K = K
        self.eta = nn.Parameter(torch.zeros(K))
        self.denoiser = Denoiser()

    def forward(self, y):
        z = y.clone()
        for k in range(self.K):
            etas = torch.sigmoid(self.eta) * 0.95 + 0.01
            z = z - etas[k] * (z - y)
            z = self.denoiser(z)
        return z


class DnCNN(nn.Module):
    """17-layer DnCNN, residual learning, no spectral norm."""
    def __init__(self, depth=17, feats=64):
        super().__init__()
        layers = [nn.Conv2d(1, feats, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(feats, feats, 3, padding=1, bias=False),
                       nn.BatchNorm2d(feats), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(feats, 1, 3, padding=1, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return x - self.net(x)


class IRCNN(nn.Module):
    """7-layer dilated CNN, residual learning."""
    def __init__(self, feats=64):
        super().__init__()
        layers = []
        for i, d in enumerate([1, 2, 3, 4, 3, 2, 1]):
            layers += [nn.Conv2d(1 if i == 0 else feats, feats, 3, padding=d, dilation=d, bias=False),
                       nn.BatchNorm2d(feats), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(feats, 1, 3, padding=1, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return x - self.net(x)


# ============================================================
# 3. METRICS
# ============================================================

def psnr_np(c, d):
    m = np.mean((c-d)**2)
    return 20*np.log10(1/np.sqrt(m)) if m > 0 else 100

def ssim_np(c, d):
    return structural_similarity(c, d, data_range=1.0)


# ============================================================
# 4. TRAINING
# ============================================================

def train_epoch(model, loader, opt, epoch, name):
    model.train()
    tot = 0.0
    pbar = tqdm(loader, desc=f"[{name}] E{epoch:3d}", ncols=90)
    for noisy, clean in pbar:
        noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
        opt.zero_grad()
        loss = F.l1_loss(model(noisy), clean)
        loss.backward()
        opt.step()
        tot += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.5f}")
    return tot / len(loader)


@torch.no_grad()
def eval_model(model, loader, name):
    model.eval()
    ps, ss = [], []
    for noisy, clean, sz in tqdm(loader, desc=f"[{name}] Eval", ncols=70, leave=False):
        out = model(noisy.to(DEVICE)).detach().cpu().squeeze().numpy()
        cn = clean.squeeze().numpy()
        h, w = sz[0].item(), sz[1].item()
        out, cn = out[:h,:w], cn[:h,:w]
        ps.append(psnr_np(cn, out))
        ss.append(ssim_np(cn, out))
    return float(np.mean(ps)), float(np.mean(ss))


def train_model(model, train_loader, test_loader, name, epochs):
    opt = optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=1e-4)
    sched = optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)
    best_psnr = 0.0
    hist = {'epoch': [], 'loss': [], 'psnr': [], 'ssim': []}
    for e in range(1, epochs+1):
        loss = train_epoch(model, train_loader, opt, e, name)
        sched.step()
        p, s = eval_model(model, test_loader, name)
        hist['epoch'].append(e); hist['loss'].append(loss)
        hist['psnr'].append(p); hist['ssim'].append(s)
        print(f"  [{name}] E{e:3d} | Loss={loss:.5f} | PSNR={p:.2f} | SSIM={s:.4f}")
        if p > best_psnr:
            best_psnr = p
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{name}_best.pth"))
            print(f"  >> Best saved (PSNR={best_psnr:.2f})")
    return hist


def eval_bm3d_runner(loader, sigma):
    sp = sigma / 255.
    ps, ss = [], []
    for noisy, clean, sz in tqdm(loader, desc="[BM3D] Eval", ncols=70, leave=False):
        nn = noisy.squeeze().numpy(); cn = clean.squeeze().numpy()
        out = bm3d.bm3d(nn, sp)
        h, w = sz[0].item(), sz[1].item()
        out, cn = out[:h,:w], cn[:h,:w]
        ps.append(psnr_np(cn, out)); ss.append(ssim_np(cn, out))
    return float(np.mean(ps)), float(np.mean(ss))


# ============================================================
# 5. VIZ
# ============================================================

def plot_curves(histories, sigma, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name, h in histories.items():
        ax[0].plot(h['epoch'], h['loss'], label=name)
        ax[1].plot(h['epoch'], h['psnr'], label=name)
        ax[2].plot(h['epoch'], h['ssim'], label=name)
    ax[0].set_title("Train Loss"); ax[1].set_title(f"PSNR (sigma={sigma})"); ax[2].set_title("SSIM")
    for a in ax: a.legend(); a.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()


def plot_examples(model, loader, name, sigma, path, n=4):
    model.eval()
    collected = []
    for noisy, clean, sz in loader:
        out = model(noisy.to(DEVICE)).detach().cpu().squeeze().numpy()
        cn = clean.squeeze().numpy(); nn = noisy.squeeze().numpy()
        h, w = sz[0].item(), sz[1].item()
        out, cn, nn = out[:h,:w], cn[:h,:w], nn[:h,:w]
        collected.append((nn, out, cn))
        if len(collected) >= n: break
    fig, axes = plt.subplots(n, 3, figsize=(9, 3*n))
    for i, (nn, out, cn) in enumerate(collected):
        for j, (img, t) in enumerate([(nn, f"Noisy {psnr_np(cn,nn):.1f}dB"),
                                       (out, f"{name} {psnr_np(cn,out):.1f}dB"),
                                       (cn, "Clean")]):
            ax = axes[i,j] if n > 1 else axes[j]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1); ax.set_title(t, fontsize=9); ax.axis('off')
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()


def plot_bar(results, path):
    models = list(next(iter(results.values())).keys())
    sigmas = list(results.keys())
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(sigmas)); w = 0.35 / len(models)
    cols = plt.cm.Set2(np.linspace(0, 1, len(models)))
    for j, m in enumerate(['psnr', 'ssim']):
        for i, mn in enumerate(models):
            vals = [results[s][mn][m] for s in sigmas]
            ax[j].bar(x + i*w, vals, w, label=mn, color=cols[i])
        ax[j].set_xticks(x + w*(len(models)-1)/2)
        ax[j].set_xticklabels([f"s={s}" for s in sigmas])
        ax[j].set_title("PSNR" if m == 'psnr' else "SSIM")
        ax[j].legend(fontsize=8); ax[j].grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()


# ============================================================
# 6. MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='quick', choices=['full','quick','train','eval','dryrun'])
    # p.add_argument('--sigma', type=int, nargs='+', default=[15,25,50])
    p.add_argument('--sigma', type=int, nargs='+', default=[25, 50])
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    args = p.parse_args()

    if args.mode == 'dryrun':
        args.epochs = 1; args.sigma = [25]
    if args.mode == 'quick':
        args.epochs = 3

    sigmas = args.sigma
    print(f"\n{'='*60}")
    print(f"LIPN Denoising | mode={args.mode} | sigma={sigmas} | epochs={args.epochs}")
    print(f"{'='*60}\n")

    results = {}
    for sigma in sigmas:
        print(f"\n{'#'*60}\n# sigma = {sigma}\n{'#'*60}")
        train_set = TrainDataset(BSD400_DIR, sigma)
        train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                   shuffle=True, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
        test_set = TestDataset(sigma)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
        print(f"Train: {len(train_set)} patches, Test: {len(test_set)} images")

        hist_dict = {}
        eval_dict = {}

        if args.mode in ('full','quick','train','dryrun'):
            # LIPN
            print("\n>>> LIPN")
            lipn = LIPN_Denoise().to(DEVICE)
            print(f"  Params: {sum(p.numel() for p in lipn.parameters()):,}")
            hist_dict['LIPN'] = train_model(lipn, train_loader, test_loader,
                                             f"LIPN_s{sigma}", args.epochs)
            lipn.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"LIPN_s{sigma}_best.pth"), weights_only=True))
            pl, sl = eval_model(lipn, test_loader, "LIPN-final")
            eval_dict['LIPN'] = {'psnr': pl, 'ssim': sl}
            print(f"  LIPN final: {pl:.2f} dB / {sl:.4f}")

            # DnCNN
            print("\n>>> DnCNN")
            dncnn = DnCNN().to(DEVICE)
            print(f"  Params: {sum(p.numel() for p in dncnn.parameters()):,}")
            hist_dict['DnCNN'] = train_model(dncnn, train_loader, test_loader,
                                              f"DnCNN_s{sigma}", args.epochs)
            dncnn.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"DnCNN_s{sigma}_best.pth"), weights_only=True))
            pd, sd = eval_model(dncnn, test_loader, "DnCNN-final")
            eval_dict['DnCNN'] = {'psnr': pd, 'ssim': sd}
            print(f"  DnCNN final: {pd:.2f} dB / {sd:.4f}")

            # IRCNN
            print("\n>>> IRCNN")
            ircnn = IRCNN().to(DEVICE)
            print(f"  Params: {sum(p.numel() for p in ircnn.parameters()):,}")
            hist_dict['IRCNN'] = train_model(ircnn, train_loader, test_loader,
                                              f"IRCNN_s{sigma}", args.epochs)
            ircnn.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"IRCNN_s{sigma}_best.pth"), weights_only=True))
            pi, si = eval_model(ircnn, test_loader, "IRCNN-final")
            eval_dict['IRCNN'] = {'psnr': pi, 'ssim': si}
            print(f"  IRCNN final: {pi:.2f} dB / {si:.4f}")

            # BM3D
            print("\n>>> BM3D")
            pb, sb = eval_bm3d_runner(test_loader, sigma)
            eval_dict['BM3D'] = {'psnr': pb, 'ssim': sb}
            print(f"  BM3D: {pb:.2f} dB / {sb:.4f}")

            results[sigma] = eval_dict
            plot_curves(hist_dict, sigma, os.path.join(FIG_DIR, f"curves_s{sigma}.png"))
            plot_examples(lipn, test_loader, "LIPN", sigma,
                          os.path.join(FIG_DIR, f"examples_LIPN_s{sigma}.png"))
            plot_examples(dncnn, test_loader, "DnCNN", sigma,
                          os.path.join(FIG_DIR, f"examples_DnCNN_s{sigma}.png"))

        elif args.mode == 'eval':
            print("\n>>> Eval only")
            for mn in ['LIPN','DnCNN','IRCNN']:
                ckpt = os.path.join(MODEL_DIR, f"{mn}_s{sigma}_best.pth")
                if os.path.exists(ckpt):
                    m = {'LIPN': LIPN_Denoise, 'DnCNN': DnCNN, 'IRCNN': IRCNN}[mn]().to(DEVICE)
                    m.load_state_dict(torch.load(ckpt, weights_only=True))
                    pv, sv = eval_model(m, test_loader, f"{mn}-eval")
                    eval_dict[mn] = {'psnr': pv, 'ssim': sv}
                    print(f"  {mn}: {pv:.2f} dB / {sv:.4f}")
                else:
                    print(f"  [SKIP] {ckpt} not found")
            pb, sb = eval_bm3d_runner(test_loader, sigma)
            eval_dict['BM3D'] = {'psnr': pb, 'ssim': sb}
            print(f"  BM3D: {pb:.2f} dB / {sb:.4f}")
            results[sigma] = eval_dict

    # Summary
    if results:
        print(f"\n{'='*60}\n{'FINAL RESULTS':^60}\n{'='*60}")
        hdr = f"{'Sigma':>8}"
        for mn in results[sigmas[0]].keys():
            hdr += f"  {mn:>12}{'':>7}"
        print(hdr)
        print("-" * len(hdr))
        for sigma in sigmas:
            row = f"{'s='+str(sigma):>8}"
            for mn in results[sigma].keys():
                row += f"  {results[sigma][mn]['psnr']:.2f} / {results[sigma][mn]['ssim']:.4f}"
            print(row)
        plot_bar(results, os.path.join(FIG_DIR, "results_bar.png"))
        with open(os.path.join(RESULTS_DIR, "results.txt"), 'w') as f:
            for sigma in sigmas:
                f.write(f"sigma={sigma}:\n")
                for mn, m in results[sigma].items():
                    f.write(f"  {mn}: PSNR={m['psnr']:.2f}, SSIM={m['ssim']:.4f}\n")

    print(f"\nDone. Output: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
