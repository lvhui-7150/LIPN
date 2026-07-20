# -*- coding: utf-8 -*-
"""
gen_analysis_figs.py (v6) -- analysis figures for current LIPN architecture

Architecture: G(z)=z+H(z), tail zero-inited, eta only (no alpha).

Generates:
  1. fixed_point_residual.png -- ||z_{k+1}-z_k||/||z_k|| decay
  2. learned_eta.png -- eta values across iterations and noise levels
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "..", "denoising_results", "models（3轮，好结果）")
FIG_DIR = os.path.join(ROOT, "..", "按模板", "fig")
DATA_DIR = os.path.join(ROOT, "denoising-datasets-main", "BSD68")
os.makedirs(FIG_DIR, exist_ok=True)
K = 5
SIGMAS = [15, 25, 50]


# ============================================================
# Models (v6 architecture)
# ============================================================

class ResBlock(nn.Module):
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
    def __init__(self, in_ch=1, feats=64, blocks=4):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(in_ch, feats, 3, padding=1), nn.ReLU(inplace=True))
        self.body = nn.Sequential(*[ResBlock(feats) for _ in range(blocks)])
        self.tail = nn.Conv2d(feats, in_ch, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)
    def forward(self, z):
        return z + self.tail(self.body(self.head(z)))

class LIPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.K = K
        self.eta = nn.Parameter(torch.zeros(K))
        self.denoiser = Denoiser()
    def forward(self, y, return_trace=False):
        z = y.clone()
        trace = [z.clone()] if return_trace else None
        for k in range(self.K):
            ek = (torch.sigmoid(self.eta[k]) * 0.95 + 0.01).item()
            z = z - ek * (z - y)
            z = self.denoiser(z)
            if return_trace:
                trace.append(z.clone())
        return (z, trace) if return_trace else z


# ============================================================
# Dataset
# ============================================================

class TestDS(torch.utils.data.Dataset):
    def __init__(self, sigma):
        self.cd = os.path.join(DATA_DIR, "original")
        self.nd = os.path.join(DATA_DIR, f"noise{sigma}")
        self.files = sorted([f for f in os.listdir(self.cd) if f.endswith(('.png','.jpg'))])
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        fn = self.files[i]
        c = np.array(Image.open(os.path.join(self.cd, fn)).convert('L'), np.float32) / 255.
        n = np.array(Image.open(os.path.join(self.nd, fn)).convert('L'), np.float32) / 255.
        h, w = c.shape
        ph, pw = (8-h%8)%8, (8-w%8)%8
        c = np.pad(c, ((0,ph),(0,pw)), 'reflect')
        n = np.pad(n, ((0,ph),(0,pw)), 'reflect')
        return (torch.from_numpy(np.ascontiguousarray(n)).unsqueeze(0),
                torch.from_numpy(np.ascontiguousarray(c)).unsqueeze(0), (h,w))


# ============================================================
# FIGURE 1: Fixed-point residual
# ============================================================

def fig_fixed_point():
    print("Figure 1: Fixed-point residual decay...")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ['#2196F3', '#4CAF50', '#F44336']
    markers = ['o', 's', '^']
    N = 4

    for ci, sigma in enumerate(SIGMAS):
        ckpt = os.path.join(MODEL_DIR, f"LIPN_s{sigma}_best.pth")
        if not os.path.exists(ckpt):
            print(f"  SKIP sigma={sigma}: {ckpt} not found")
            continue
        m = LIPN().to(DEVICE)
        m.load_state_dict(torch.load(ckpt, weights_only=True))
        m.eval()
        ds = TestDS(sigma)
        all_d = []
        cnt = 0
        for noisy, clean, sz in torch.utils.data.DataLoader(ds, batch_size=1):
            if cnt >= N: break
            with torch.no_grad():
                _, trace = m(noisy.to(DEVICE), return_trace=True)
            h, w = sz[0].item(), sz[1].item()
            dec = []
            for k in range(K):
                diff = (trace[k+1] - trace[k])[..., :h, :w]
                norm_k = (trace[k][..., :h, :w].norm(p=2) + 1e-8)
                dec.append((diff.norm(p=2) / norm_k).item())
            all_d.append(dec)
            cnt += 1
        all_d = np.array(all_d)
        m_vals, s_vals = all_d.mean(0), all_d.std(0)
        x = np.arange(1, K+1)
        ax.errorbar(x, m_vals, yerr=s_vals, color=colors[ci], marker=markers[ci],
                     ms=6, lw=1.8, capsize=3, label=f'sigma={sigma}')
        print(f"  sigma={sigma}: {[f'{v:.4f}' for v in m_vals]}")

    ax.set_xlabel("Iteration k", fontsize=12)
    ax.set_ylabel("||z_{k+1} - z_k|| / ||z_k||", fontsize=12)
    ax.set_title("Fixed-Point Residual Decay (LIPN)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(range(1, K+1))
    ax.set_yscale('log')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fixed_point_residual.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}\n")


# ============================================================
# FIGURE 2: Learned eta
# ============================================================

def fig_eta():
    print("Figure 2: Learned step sizes...")
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#2196F3', '#4CAF50', '#F44336']
    markers = ['o', 's', '^']
    x = np.arange(K)

    for ci, sigma in enumerate(SIGMAS):
        ckpt = os.path.join(MODEL_DIR, f"LIPN_s{sigma}_best.pth")
        if not os.path.exists(ckpt):
            print(f"  SKIP sigma={sigma}")
            continue
        m = LIPN()
        m.load_state_dict(torch.load(ckpt, weights_only=True))
        eta_raw = m.eta.detach().cpu().numpy()
        eta_vals = 1/(1+np.exp(-eta_raw)) * 0.95 + 0.01
        ax.plot(x, eta_vals, color=colors[ci], marker=markers[ci], ms=8, lw=2, label=f'sigma={sigma}')
        for j in range(K):
            ax.annotate(f"{eta_vals[j]:.2f}", (x[j], eta_vals[j]),
                        textcoords="offset points", xytext=(0, 12),
                        fontsize=8, ha='center', color=colors[ci])
        print(f"  sigma={sigma}: eta={[f'{v:.3f}' for v in eta_vals]}")

    ax.set_xlabel("Iteration k", fontsize=12)
    ax.set_ylabel("Step size eta_k", fontsize=12)
    ax.set_title("Learned Step Sizes (LIPN)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(range(K))
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "learned_eta.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}\n")


# ============================================================

if __name__ == "__main__":
    print(f"Device: {DEVICE}\n")
    fig_fixed_point()
    fig_eta()
    print("Done.")
