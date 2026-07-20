"""
inspect_model.py (v6) -- evaluate saved checkpoints
Usage:
    python inspect_model.py LIPN 25
    python inspect_model.py all 15 25 50
"""
import os, argparse
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity
import bm3d

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "..", "denoising_results", "models")
DATA_DIR = os.path.join(HERE, "denoising-datasets-main", "BSD68")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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
    def __init__(self, K=5):
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
    def __init__(self, feats=64):
        super().__init__()
        layers = []
        for i, d in enumerate([1, 2, 3, 4, 3, 2, 1]):
            layers += [nn.Conv2d(1 if i == 0 else feats, feats, 3, padding=d, dilation=d, bias=False),
                       nn.BatchNorm2d(feats), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(feats, 1, 3, padding=1, bias=False))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return x - self.net(x)


class DS(torch.utils.data.Dataset):
    def __init__(self, sigma):
        self.cd = os.path.join(DATA_DIR, "original")
        self.nd = os.path.join(DATA_DIR, f"noise{sigma}")
        self.files = sorted([f for f in os.listdir(self.cd) if f.endswith(('.png','.jpg'))])
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        c = np.array(Image.open(os.path.join(self.cd, self.files[i])).convert('L'), np.float32) / 255.
        n = np.array(Image.open(os.path.join(self.nd, self.files[i])).convert('L'), np.float32) / 255.
        h, w = c.shape
        ph, pw = (8-h%8)%8, (8-w%8)%8
        c = np.pad(c, ((0,ph),(0,pw)), 'reflect'); n = np.pad(n, ((0,ph),(0,pw)), 'reflect')
        return torch.from_numpy(np.ascontiguousarray(n)).unsqueeze(0), torch.from_numpy(np.ascontiguousarray(c)).unsqueeze(0), (h,w)

def psnr(c, d):
    m = np.mean((c-d)**2); return 20*np.log10(1/np.sqrt(m)) if m > 0 else 100

@torch.no_grad()
def evaluate(name, sigma):
    ckpt = os.path.join(MODEL_DIR, f"{name}_s{sigma}_best.pth")
    if not os.path.exists(ckpt):
        print(f"  [SKIP] {ckpt} not found"); return None
    m = {'LIPN':LIPN,'DnCNN':DnCNN,'IRCNN':IRCNN}[name]().to(DEVICE)
    m.load_state_dict(torch.load(ckpt, weights_only=True))
    m.eval()
    pv, sv = [], []
    for nz, cl, sz in tqdm(DataLoader(DS(sigma), batch_size=1), desc=f"  {name} s={sigma}", ncols=70, leave=False):
        o = m(nz.to(DEVICE)).cpu().squeeze().numpy()
        cn = cl.squeeze().numpy(); h, w = sz[0].item(), sz[1].item()
        pv.append(psnr(cn[:h,:w], o[:h,:w]))
        sv.append(structural_similarity(cn[:h,:w], o[:h,:w], data_range=1.0))
    return float(np.mean(pv)), float(np.mean(sv))

def eval_bm3d(sigma):
    sp = sigma / 255.
    pv, sv = [], []
    for nz, cl, sz in tqdm(DataLoader(DS(sigma), batch_size=1), desc=f"  BM3D s={sigma}", ncols=70, leave=False):
        o = bm3d.bm3d(nz.squeeze().numpy(), sp)
        cn = cl.squeeze().numpy(); h, w = sz[0].item(), sz[1].item()
        pv.append(psnr(cn[:h,:w], o[:h,:w]))
        sv.append(structural_similarity(cn[:h,:w], o[:h,:w], data_range=1.0))
    return float(np.mean(pv)), float(np.mean(sv))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("sigma", type=int, nargs="+", default=[25])
    a = p.parse_args()
    models = ["LIPN","DnCNN","IRCNN","BM3D"] if a.model.lower() == "all" else [a.model]
    print(f"\n{'='*55}\n{'Model':>8}  {'Sigma':>6}  {'PSNR':>8}  {'SSIM':>8}\n{'-'*55}")
    for mn in models:
        for s in a.sigma:
            if mn == "BM3D":
                pv, sv = eval_bm3d(s)
                print(f"{'BM3D':>8}  {'s='+str(s):>6}  {pv:>7.2f}  {sv:>7.4f}")
            else:
                r = evaluate(mn, s)
                if r is None: continue
                print(f"{mn:>8}  {'s='+str(s):>6}  {r[0]:>7.2f}  {r[1]:>7.4f}")
                if mn == "LIPN":
                    m = LIPN()
                    m.load_state_dict(torch.load(os.path.join(MODEL_DIR,f"LIPN_s{s}_best.pth"),weights_only=True))
                    et = (1/(1+np.exp(-m.eta.detach().cpu().numpy())))*0.95+0.01
                    print(f"{'':>8}  {'':>6}  {'eta':>8}  {'  '.join([f'{v:.3f}' for v in et])}")
    print('-'*55+'\n')

if __name__ == "__main__": main()
