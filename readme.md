<div align=center>
<img src="figs/logo.png" height="80%" width="80%"/>
</div>

# [TPAMI 2026] Band-kernel Stochastic Learning for Unsupervised Blind Hyperspectral Image Super-Resolution

[Zhixiong Yang](https://zhixiongyang21.github.io/), [Jingyuan Xia](https://www.xiajingyuan.com/), Shengxi Li, Lingyu Zheng, Shuanghui Zhang, Zhen Liu, Li Liu, Yaowen Fu, Deniz Gündüz, Yongxiang Liu

<p>
	<img src="https://img.shields.io/badge/TPAMI-2026-0A66C2?style=for-the-badge" alt="TPAMI 2026" />
	<img src="https://img.shields.io/badge/Task-Blind%20HSI%20SR-0F766E?style=for-the-badge" alt="Task" />
	<img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
	<img src="https://img.shields.io/badge/PyTorch-1.12.1%2Bcu113-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
</p>

<p>
	<a href="#quick-demo"><img src="https://img.shields.io/badge/Run-Demo-111827?style=flat-square" alt="Run Demo" /></a>
	<a href="#visual-results-on-noisy-super-resolution"><img src="https://img.shields.io/badge/View-Figures-111827?style=flat-square" alt="View Figures" /></a>
	<a href="#citation"><img src="https://img.shields.io/badge/Cite-TPAMI-111827?style=flat-square" alt="Citation" /></a>
</p>

[![arxiv](https://img.shields.io/badge/IEEE-paper-blue)](https://ieeexplore.ieee.org/document/11475666)

---

# Table of Contents

| Section | Description |
|---|---|
| 📌 [Overview](#overview) | Paper abstract and summary |
| 🖼️ [Visual Results](#visual-results-on-noisy-super-resolution) | Example outputs |
| ⚙️ [Environment Setup](#environment-setup) | Dependencies & install |
| 🗂️ [Project Structure](#project-structure) | Repo layout |
| 📁 [Datasets Preparation](#datasets-preparation) | Data & checkpoints |
| ▶️ [Quick Demo](#quick-demo) | Run examples & commands |
| 📚 [Citation](#citation) | How to cite |

# 📌 Overview
> **Abstract:** *Hyperspectral image super-resolution (HSI-SR) is fundamentally more difficult than RGB image SR, since its ultrahigh spectral dimensionality.Existing supervised methods rely on labeled training data to obtain data prior, which incurs prohibitive collection costs and limits generalization.Unsupervised methods individually preset the band and kernel with handcrafted priors, whereas this decoupling modeling artificially creates a complexity-performance trade-off in the selected band number.To address these issues, we propose BKX-HMM, a unified statistical framework for blind HSI-SR, which uniformly models the band selection, kernel estimation, and HSI restoration through the state transition of a hidden Markov model (HMM).BKX-HMM redefines the trade-off as a distributional fitting problem: each Markov transition progressively learns optimal parameters of a full-band distribution via limited spectral observations. Based on BKX-HMM, we propose BKSR, the first unsupervised blind HSI-SR method, which consists of three synergistic modules: Gibbs sampling-based band selection (GBS), test-time-training kernel estimation (TKE), and robust HSI restoration (RHR).These modules form a closed-loop optimization cycle: i) In GBS, the dynamic ergodicity of Gibbs sampling provides a global spectral view for kernel estimation and HSI restoration while maintaining local spectral computations;ii) In TKE, the GBS-sampled bands guide the kernel estimator update, achieving a learnable sampling-based mechanism, which refines kernel estimation to regularize RHR’s diffusion trajectory;iii) In RHR, a spectral hyper-Laplacian prior is integrated into the reverse process of an off-the-shelf diffusion model, which achieves non-i.i.d. noise robust HSI restoration, feedback reweights band and kernel importance for subsequent GBS and TKE iterations.Extensive experiments on both synthetic and real HSI datasets demonstrate our BKSR's superiority over baseline methods across diverse scenarios (e.g., unknown Gaussian/motion kernel, non-i.i.d. noise) while maintaining comparable computational costs to the classic band selection methods.*

<div align=center>
<img src="figs/framework.png" height="100%" width="100%"/>
</div>

# 🖼️ Visual Results On Noisy Super-Resolution

<div align=center>
<img src="figs/Salinas_nonblind.png" height="100%" width="100%"/>
</div>

<div align=center>
<img src="figs/WDC_Blind.png" height="100%" width="100%"/>
</div>

<div align=center>
<img src="figs/Chikusei_noniid.png" height="100%" width="100%"/>
</div>

# ⚙️ Environment Setup

Recommended software stack:

- Python 3.8+
- CUDA 11.3
- PyTorch 1.12.1+cu113

Install dependencies:

```bash
pip install -r requirements.txt
```

# 🗂️ Project Structure

```text
BKSR/
├─ main.py                         # main entry
├─ run_yaml_demo.py                # yaml quick demo 
├─ configs/                        # experiment config
├─ yaml/                           # pre-built demo configs
├─ data/                           # .mat datasets
├─ checkpoints/diffusion/          # pretrained weights
├─ guided_diffusion/               # model and sampling core
├─ utility/                        # metrics and utility functions
├─ result/                         # output files
└─ figs/                           # paper and experiment figures
```

# 📁 Datasets Preparation

## Dataset Paths

Please download the `prepared datasets` from [`Dropbox`](https://www.dropbox.com/scl/fo/26nqxrgmv4dkjvojkjyje/ADEfEJosLVEx5_AVxB2xLBE?rlkey=3iic8xfor6y93r3of5iuhf1er&st=tj534tdj&dl=0) or [`Baidu Netdisk`](https://pan.baidu.com/s/1MkvPBYlfJxRIIVKvnDb4zg?pwd=BKSR) first. Each dataset already contains the sliced Ground Truth cubes used by the demo, and the files should be saved under the corresponding `HR` directory.

```text
data/
├─ Chikusei/HR/Chikusei_crop.mat
├─ WDC/HR/WDC_crop.mat
├─ Salinas/HR/Salinas_crop.mat
└─ Indian/HR/Indian_crop.mat
```

## Pretrained Diffusion Weights

Download the checkpoint file
[`I190000_E97_gen.pth`](https://www.dropbox.com/sh/z6k5ixlhkpwgzt5/AAApBOGEUhHa4qZon0MxUfmua?dl=0)
provided by
[`ddpm-cd`](https://github.com/wgcban/ddpm-cd),
and place it at:

```text
checkpoints/diffusion/I190000_E97_gen.pth
```

# ▶️ Quick Demo

After tuning the hyperparameters, run main.py by default：

```bash
python main.py
```
You can store demo parameters in a YAML file and run the demo quickly. Example YAML is provided at `yaml/`.

Run with the helper script:

```bash
python run_yaml_demo.py yaml/Case1.yml
```
```bash
python run_yaml_demo.py yaml/Case2.yml
```
```bash
python run_yaml_demo.py yaml/Noniid.yml
```

The helper converts YAML keys to `--key value` CLI arguments and launches `main.py` in the current Python environment. Example content of `yaml/Case1.yml`:

```yaml
dataname: Salinas
eta1: 70
eta2: 4
k: 2
step: 50
task: sr
sf: 4
k_s: 9
min_var: 1.69
max_var: 1.69
non_iid: false
blind: false
```
Execution notes:

- `eta1` and `eta2` are internally rescaled in `main.py`.
- Default task is super-resolution (`-task sr`) with `x4` scale.
- Use `-gpu 0` (or your own GPU id) to select GPU.

# 📚 Citation

If this repository is useful for your research, please cite:

```text
@article{yang2026BKSR,
  author={Yang, Zhixiong and Xia, Jingyuan and Li, Shengxi and Zheng, Lingyu and Zhang, Shuanghui and Liu, Li and Fu, Yaowen and Liu, Yongxiang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Band-kernel Stochastic Learning for Unsupervised Blind Hyperspectral Image Super-Resolution}, 
  year={2026},
  volume={},
  number={},
  pages={1-18},
  keywords={Earth Observing System;Feedback;Circuits;Filters;Anisotropic;Circuits and systems;Filtering;Pixel;TV;Videos;Hyperspectral image super-resolution;unsupervised learning;Gibbs sampling;diffusion models},
  doi={10.1109/TPAMI.2026.3681688}}
```

# Acknowledge
This code is based on [`HIR-Diff`](https://github.com/LiPang/HIRDiff), and [`ddpm-cd`](https://github.com/wgcban/ddpm-cd). Thanks for their awesome work.

# Contact 
If you have any questions, please feel free to contact us at yzx21@nudt.edu.cn or lingyu25@nudt.edu.cn.
