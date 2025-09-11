#!/usr/bin/env python3
import os
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from google.cloud import storage
from .encoder_data_loader import EncoderDataset
from .model import EncoderProxy
import torch.multiprocessing as mp
from pytorch_msssim import ms_ssim

# ------------------------------
# Adding random seed to guarantee reproducibility
# ------------------------------
import random
import numpy as np
seed = 42  # Or make it an argparse arg
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ------------------------------
# Fix for fork processes
# ------------------------------
mp.set_start_method('spawn', force=True)

# ------------------------------
# Helpers
# ------------------------------
def save_checkpoint(epoch, model, optimizer, val_loss, path, scheduler=None):
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'val_loss': val_loss,
    }

    if scheduler is not None:
        state['scheduler_state'] = scheduler.state_dict()

    torch.save(state, str(path))

    client = storage.Client()
    bucket = client.bucket('encoder_proxy_training_checkpoints')
    blob = bucket.blob(f'checkpoints/{os.path.basename(str(path))}')
    blob.upload_from_filename(str(path))
    print(f"Checkpoint Saved: {path}")

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

def ms_ssim_loss(pred, target):
    # compute on clamped copies to keep the metric well-behaved
    pred_ = pred.clamp(0, 1)
    target_ = target.clamp(0, 1)
    # ms_ssim returns similarity to MAXIMIZE; turn into a loss to MINIMIZE
    return 1.0 - ms_ssim(pred_, target_, data_range=1.0, size_average=True)

if __name__ == "__main__":
    # ------------------------------
    # CLI Argument Parsing
    # ------------------------------
    parser = argparse.ArgumentParser(description="Training Encoder Proxy")

    # Dataset paths
    parser.add_argument("--csv_path", type=str, required=True, help="Path to metadata CSV (training + validation).")
    parser.add_argument("--frame_dir", type=str, required=True, help="Root directory containing extracted frames.")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints.")

    # Model hyperparameters
    parser.add_argument("--crop_size", type=int, default=256, help="Crop size for input frames.")
    parser.add_argument("--bottleneck_channels", type=int, default=320, help="Channels in bottleneck.")
    parser.add_argument("--film_hidden_dim", type=int, default=32, help="Hidden dim for FiLM MLP.")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping based on validation loss.")
    parser.add_argument("--patience", type=int, default=10, help="Number of epochs to wait without improvement.")
    parser.add_argument("--use_plateau_scheduler", action="store_true", help="Modify the learning rate when learning plateaus.")
    parser.add_argument("--w_ms_ssim", type=float, default=0.12, help="Weight for MS-SSIM loss")
    parser.add_argument("--w_ms_ramp_epochs", type=int, default=10, help="Linearly ramp w_ms_ssim from 0 to target over these epochs.")

    # Checkpointing
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training.")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs.")
    parser.add_argument("--save_best", action="store_true", help="Save checkpoint with best validation loss.")

    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: 'cuda' or 'cpu'.")

    args = parser.parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # Setup
    # ------------------------------
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_dataset = EncoderDataset(args.csv_path, args.frame_dir, crop_size=args.crop_size,
                                fixed_crop=False, split="Training")
    val_dataset = EncoderDataset(args.csv_path, args.frame_dir, crop_size=args.crop_size,
                                fixed_crop=True, split="Validation")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)

    model = EncoderProxy(bottleneck_channels=args.bottleneck_channels,
                        film_hidden_dim=args.film_hidden_dim).to(device)

    mse_loss = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(),
                        lr=args.learning_rate,
                        weight_decay=args.weight_decay)
    
    if args.use_plateau_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6, verbose=True
        )

    start_epoch = 0
    best_val_loss = float("inf")
    best_ckpt_path = Path(args.save_dir) / "best_checkpoint.pth"
    patience_counter = 0

    if args.resume:
        resume_path = args.resume

        # If a directory was passed, try best_checkpoint.pth inside it
        if os.path.isdir(resume_path):
            resume_path = str(Path(resume_path) / "best_checkpoint.pth")
        
        # If it's a GCS URI, download it locally first
        if isinstance(resume_path, str) and resume_path.startswith("gs://"):
            if storage is None:
                raise RuntimeError("google-cloud-storage is required to resume from gs://")
            uri = resume_path[len("gs://"):]
            bucket_name, _, blob_name = uri.partition("/")
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            local_dir = Path(args.save_dir) if args.save_dir else Path("/tmp/checkpoints")
            local_dir.mkdir(parents=True, exist_ok=True)
            local_resume = str(local_dir / "resume_checkpoint.pth")
            blob.download_to_filename(local_resume)
            print(f"Downloaded resume checkpoint from {args.resume} to {local_resume}")
            resume_path = local_resume
        
        if os.path.isfile(resume_path):
            print(f"Resume Loading checkpoint {args.resume}")
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint.get('epoch', -1) + 1
            best_val_loss = checkpoint.get('val_loss', float("inf"))
            if args.use_plateau_scheduler and 'scheduler_state' in checkpoint:
                print("Loading the scheduler")
                scheduler.load_state_dict(checkpoint['scheduler_state'])
            print(f"Resumed at epoch {start_epoch}, best val loss {best_val_loss:.4f}")
        else:
            print(f"WARNING: --resume specified but file not found: {resume_path}")

    # ------------------------------
    # Training Loop
    # ------------------------------
    for epoch in range(start_epoch, args.epochs):

        model.train()
        train_loss = 0.0
        train_l1 = 0.0
        train_ms = 0.0
        train_psnr = 0.0

        # Training
        for batch in train_loader:
            x, crf, y_true = batch

            x, crf, y_true = x.to(device), crf.to(device), y_true.to(device)

            optimizer.zero_grad()

            y_hat = model(x, crf)

            # Reconstruction on distorted frame
            loss_l1 = torch.nn.functional.l1_loss(y_hat, y_true)
            mse = mse_loss(y_hat, y_true)
            psnr_db = -10.0* torch.log10(mse + 1e-9)

            #MS-SSIM loss
            loss_ms  = ms_ssim_loss(y_hat, y_true)
            
            # Final scalar loss
            target_ms_weight = args.w_ms_ssim
            ramp_e = max(1, args.w_ms_ramp_epochs)
            w_ms = target_ms_weight * min(1.0, (epoch + 1) / ramp_e)
            loss = loss_l1 + w_ms * loss_ms

            loss.backward()
            #GDN + deconvs can occasionally spike
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_l1 += loss_l1.item() * x.size(0)
            train_ms  += (1.0 - loss_ms.item()) * x.size(0)
            train_psnr += psnr_db.item() * x.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_l1 /= len(train_loader.dataset)
        train_ms /= len(train_loader.dataset)
        train_psnr /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss, l1_val, ms_val, psnr_val = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, crf, y_true = batch

                x, crf, y_true = x.to(device), crf.to(device), y_true.to(device)
                y_hat = model(x, crf)

                loss_l1 = torch.nn.functional.l1_loss(y_hat, y_true)
                mse = mse_loss(y_hat, y_true)
                psnr_db = -10.0* torch.log10(mse + 1e-9)
                loss_ms = ms_ssim_loss(y_hat, y_true)
                              
                total_loss = loss_l1 + args.w_ms_ssim * loss_ms

                # Accumulate
                val_loss += total_loss.item() * x.size(0)
                l1_val += loss_l1.item() * x.size(0)
                ms_val += (1.0 - loss_ms.item()) * x.size(0)
                psnr_val += psnr_db.item() * x.size(0)

        val_loss /= len(val_loader.dataset)
        l1_val /= len(val_loader.dataset)
        ms_val /= len(val_loader.dataset)
        psnr_val /= len(val_loader.dataset)

        if args.use_plateau_scheduler:
            scheduler.step(val_loss)
            print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")

        print(
        f"Epoch {epoch+1}/{args.epochs} | "
        f"Train l1={train_l1:.4f}, "
        f"Train MS-SSIM={train_ms:.4f}, "
        f"Train PSNR={train_psnr:.4f} dB, "
        f"Train total={train_loss:.4f} | "
        f"Validation l1={l1_val:.4f}, "
        f"Validation MS-SSIM={ms_val:.4f}, "
        f"Validation PSNR={psnr_val:.4f} dB, "
        f"Validation total={val_loss:.4f} "
        )

        # Early stopping
        if args.early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if args.save_best:
                    save_checkpoint(epoch, model, optimizer, val_loss, best_ckpt_path, scheduler if args.use_plateau_scheduler else None)
            else:
                patience_counter += 1
                print(f"No improvement in val_loss for {patience_counter} epoch(s).")
                if patience_counter >= args.patience:
                    print("Early stopping triggered.")
                    if best_ckpt_path.is_file():
                        print("Restoring best checkpoint...")
                        checkpoint = torch.load(best_ckpt_path, map_location=device)
                        model.load_state_dict(checkpoint['model_state'])
                    break

        # Checkpoints
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = Path(args.save_dir) / f"checkpoint_epoch{epoch+1}.pth"
            save_checkpoint(epoch, model, optimizer, val_loss, ckpt_path, scheduler if args.use_plateau_scheduler else None)