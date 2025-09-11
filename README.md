# Encoder Proxy

This repository provides a deep learning model that **approximates a video encoder in all-intra mode**.  
This repo provides a full, reproducible pipeline from **dataset split → compression (x265) → frame extraction → CSV metadata → training** as well as cloud-friendly options and seeds for determinism.

**Project status:** Training is in progress. I am not publishing quantitative results yet. The repository is focused on clean setup and a smooth developer experience. Pretrained weights, validation plots, and evaluation scripts will be added once training stabilizes.

This work builds a **neural proxy** that:
- takes a **reference (original) frame** and a **CRF**,
- passes it through a **CRF-conditioned autoencoder** with a **quantization bottleneck**, and
- produces a **reconstruction that mimics the encoder’s all-intra output**.

While the pipeline supports multiple encoders, **x265** is the primary encoder used for training here.

# Dataset
This project trains and validates on **YouTube UGC (User-Generated Content) Dataset** which contains short real-world clips captured on phones, action cams, consumer gear, then uploaded with a wide range of native distortions such as blockiness, blur, banding, noise, and jerkiness. Training on this dataset has a wide variety of advantages such as:
- **Real-world distribution:** UGC covers the long tail of scenes, devices, and capture conditions
- **High content diversity:** Strong variation in motion, texture, lighting, scene dynamics, and camera pipelines improves **generalization** of the learned mapping from input frame to reconstructed frame

 I generated distorted frames by compressing each input sequence with **x265** in **all-intra** mode across a **wide CRF sweep: 19, 23, 27, 31, 35, 39, 43, 47, 51** to span low→high quality. To ensure all quality levels are well represented I apply binning and sample an approximately equal number of sequences per bin so training sees a balanced spread from visually pristine to very compressed, improving robustness of the learned behavior.

## Model architecture

- **Backbone:** residual conv/conv-transpose blocks with **GDN** (Generalized Divisive Normalization).
- **Conditioning:** **FiLM** layers modulate the latent using CRF (scalar input → small MLP → per-channel scale/shift).
- **Quantization bottleneck:** rounding with **Straight-Through Estimator** (STE) for gradients.
- **Output:** single-channel **Y (grayscale)** frame in `[0, 1]`.

**Objective**  
We minimize a weighted sum of:
- **L1 reconstruction loss**, and
- **MS-SSIM loss** (as a loss term: `1 − MS-SSIM`).

**PSNR** is also logged for visibility during training.

## Repository layout
```text
encoder-proxy-intra/
├─ encoder_proxy/
│  ├─ data/
│  │  └─ encoder_data_loader.py      # CSV-driven dataset (local or GCS), Y-channel crops
│  ├─ models/
│  │  └─ model.py                    # GDN, FiLM, residual (conv/deconv), STE rounding
│  └─ train/
│     └─ task.py                     # Training loop & CLI (seeding, early stop, MS-SSIM, etc.)
│
├─ scripts/
│  ├─ dataset_split.py               # Split UGC dataset into Train/Val/Test
│  ├─ compress.py                    # All-intra encode (x265) + per-frame QP/bits logs
│  ├─ extract_frames.py              # Extract Y (grayscale) frames for ref & distorted
│  ├─ generate_csv.py                # Assemble training CSV (+ optional bitrate balancing)
│  └─ infer.py                       # (COMING SOON) Visualize recon vs. distorted


