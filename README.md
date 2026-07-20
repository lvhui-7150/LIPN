# LIPN Experiments

Experimental code for the paper “A Complex-Valued Variational Inequality Method Based on Learned Inexact Proximal Operators.”

## Directory Structure

```
Experiments/
├── README.md               ← This file
├── Denoising Experiments/                ← Natural Image Denoising
│   ├── README.md
│   ├── denoising_experiment.py
│   ├── denoising-datasets-main/
│   └── ...
├── MRI Experiments/                 ← Complex-domain MRI reconstruction
│   ├── README.md
│   ├── LIPN.py
│   ├── IXI-dataset-master/
│   └── ...
├── denoising_results/       ← Denoising experiment outputs (models + figures)
└── By Template/                   ← LaTeX paper source files
    ├── main.tex
    ├── fig/
    └── ...
```

## Two Experiments

| Experiment | Data | Objective | Comparison Models |
|---|---|---|---|
| Denoising | BSD400 → BSD68 | Validate framework generality | BM3D, DnCNN, IRCNN |
| MRI | IXI brain MRI | Core application: Complex-domain VI | VarNet, MoDL, ComplexUNet, and 8 others |

## Environment

- Python 3.10+, PyTorch 2.x, CUDA 11.8+
- `pip install bm3d scikit-image matplotlib tqdm pillow`


Translated with DeepL.com (free version)
