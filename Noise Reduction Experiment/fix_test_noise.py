"""
fix_test_noise.py — Regenerate BSD68 test noise images with correct sigma
==========================================================================
The pre-computed BSD68 noise images had incorrect noise standard deviation
(~6 instead of 15 at sigma=15). This script regenerates them correctly.

Noise convention (matching DenoisingTrainDataset):
    noise_std (uint8) = sigma  →  noise_std ([0,1]) = sigma / 255

Usage: python fix_test_noise.py
"""
import os, shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
BSD68_DIR = os.path.join(ROOT, "denoising-datasets-main", "BSD68")
ORIGINAL_DIR = os.path.join(BSD68_DIR, "original")
SIGMAS = [15, 25, 50]
SEED = 42     # reproducibility

def main():
    files = sorted([f for f in os.listdir(ORIGINAL_DIR)
                    if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    print(f"Found {len(files)} original images in {ORIGINAL_DIR}\n")

    for sigma in SIGMAS:
        noisy_dir = os.path.join(BSD68_DIR, f"noise{sigma}")
        # Backup old noisy images
        backup_dir = noisy_dir + "_old"
        if os.path.exists(noisy_dir) and not os.path.exists(backup_dir):
            shutil.move(noisy_dir, backup_dir)
            print(f"Backed up old noise{sigma} -> noise{sigma}_old")

        os.makedirs(noisy_dir, exist_ok=True)
        rng = np.random.RandomState(SEED + sigma)

        for fname in tqdm(files, desc=f"sigma={sigma}"):
            clean = Image.open(os.path.join(ORIGINAL_DIR, fname)).convert('L')
            clean_arr = np.array(clean, dtype=np.float64)

            # Add Gaussian noise in uint8 space
            noise = rng.randn(*clean_arr.shape).astype(np.float64) * float(sigma)
            noisy_arr = np.clip(clean_arr + noise, 0, 255).astype(np.uint8)

            noisy_img = Image.fromarray(noisy_arr, mode='L')
            noisy_img.save(os.path.join(noisy_dir, fname))

        # Verify
        test = Image.open(os.path.join(noisy_dir, files[0])).convert('L')
        test_clean = Image.open(os.path.join(ORIGINAL_DIR, files[0])).convert('L')
        diff = np.array(test, dtype=np.float64) - np.array(test_clean, dtype=np.float64)
        print(f"  Verified: noise std={diff.std():.2f} (expected {float(sigma):.1f})\n")

    print("Done. Test noise images regenerated with correct sigma.")

if __name__ == "__main__":
    main()
