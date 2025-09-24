# Encoder Proxy

This repository provides a deep learning model that **approximates a video encoder in all-intra mode**.  
This repo provides a full, reproducible pipeline from **dataset split → compression (x265) → frame extraction → CSV metadata → training** as well as cloud-friendly options and seeds for determinism.

**Project status:** Training and evaluation are finalized. The evaluation results are available in its designated folder in this git repository.

This work builds a **neural proxy** that:
- takes a **reference (original) frame** and a **CRF**,
- passes it through a **CRF-conditioned autoencoder** with a **quantization bottleneck**, and
- produces a **reconstruction that mimics the encoder’s all-intra output**.

While the pipeline supports multiple encoders, **x265** is the primary encoder used for training here.

## Note: 
Found a bug or have an improvement? Contributions are welcome! 🙌
1. **Open an issue** with a minimal repro (dataset slice, command line, logs).
2. **Submit a Merge Request / Pull Request** referencing the issue.
3. Please follow this checklist:
   - Clear description of the bug/fix
   - Steps to reproduce (commands, args, env details)
   - Affected files

I will review merge requests and leave feedback or merge when ready. Thanks for helping improve the project! 🚀

# Dataset
This project trains and validates on **YouTube UGC (User-Generated Content) 720p Dataset** which contains short real-world clips captured on phones, action cams, consumer gear, then uploaded with a wide range of native distortions such as blockiness, blur, banding, noise, and jerkiness. Training on this dataset has a wide variety of advantages such as:
- **Real-world distribution:** UGC covers the long tail of scenes, devices, and capture conditions
- **High content diversity:** Strong variation in motion, texture, lighting, scene dynamics, and camera pipelines improves **generalization** of the learned mapping from input frame to reconstructed frame

 I generated distorted frames by compressing each raw YUV input sequence with **x265** in **all-intra** mode across a **wide CRF sweep: 19, 23, 27, 31, 35, 39, 43, 47, 51** to span low→high quality. To ensure all quality levels are well represented I apply binning and sample an approximately equal number of sequences per bin so training sees a balanced spread from visually pristine to very compressed, improving robustness of the learned behavior.

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

## Runtime environments

This codebase is **Vertex AI–first**:

- **Primary target**: Google Vertex AI (custom training jobs), GCS storage, and `gs://` paths.
- **Local runs**: It is supported, but you may need to turn off cloud-only features and/or install extra packages.

### Run on Google Vertex AI

```bash
PROJECT_ID="<YOUR_PROJECT>"
REGION="us-central1"
BUCKET_CODE="gs://<YOUR_CODE_BUCKET>"          # where your package .tar.gz lives
BUCKET_DATA="gs://<YOUR_DATA_BUCKET>"          # dataset/frames/CSV
BUCKET_OUT="<YOUR_OUTPUT_BUCKET>"               # for checkpoint uploads
PKG_URI="$BUCKET_CODE/encoder_proxy-0.1.tar.gz" # or your existing trainer-0.1.tar.gz
PY_IMAGE="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-0.py310:latest"

gcloud config set project "$PROJECT_ID"

gcloud ai custom-jobs create \
  --region="$REGION" \
  --display-name="encoder-proxy-full-training" \
  --python-package-uris="$PKG_URI" \
  --worker-pool-spec=machine-type=g2-standard-8,replica-count=1,accelerator-type=NVIDIA_L4,accelerator-count=1,executor-image-uri="$PY_IMAGE",python-module=encoder_proxy.train.task \
  --args="--csv_path=$BUCKET_DATA/dataset/encoder_proxy_training.csv" \
  --args="--frame_dir=$BUCKET_DATA/dataset/" \
  --args="--save_dir=/tmp/checkpoints" \
  --args="--gcs_bucket=$BUCKET_OUT" \
  --args="--save_best" \
  --args="--device=cuda"
```

## Repository layout
```text
encoder_proxy_intra/
├─ README.md
├─ LICENSE
├─ gitignore
├─ encoder_proxy/
│  ├─ data/
│  │  └─ encoder_data_loader.py      # CSV-driven dataset (local or GCS), Y-channel crops
│  ├─ models/
│  │  └─ model.py                    # GDN, FiLM, residual (conv/deconv), STE rounding
│  └─ train/
│     └─ train.py                    # Training loop & CLI (seeding, early stop, MS-SSIM, etc.)
│  └─ inference/
│     └─ infer_and_eval.py           # Inference and evaluation script
│     └─ README.md                   # Inference and evaluation results and analysis
├─ scripts/
│  ├─ dataset_split.py               # Split UGC dataset into Train/Val/Test
│  ├─ compress.py                    # All-intra encode (x265) + per-frame QP/bits logs
│  ├─ extract_frames.py              # Extract Y (grayscale) frames for ref & distorted
│  ├─ generate_csv.py                # Assemble training CSV (+ optional bitrate balancing)
```
