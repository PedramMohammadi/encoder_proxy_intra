#!/usr/bin/env python3

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from google.cloud import storage
from tqdm import tqdm
from pytorch_msssim import ms_ssim, ssim
from torch.utils.data._utils.collate import default_collate

# Import your model and dataset classes
from .encoder_data_loader import EncoderDataset
from .model import EncoderProxy

def _to_py(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(f"type {type(o).__name__} not JSON serializable")

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch) if batch else None

def load_encoder_model(checkpoint_path, device, bottleneck_channels=320, film_hidden_dim=32):
    """Load the trained encoder proxy model"""
    model = EncoderProxy(
        bottleneck_channels=bottleneck_channels,
        film_hidden_dim=film_hidden_dim
    ).to(device)
    
    # Download checkpoint if it's from GCS
    if checkpoint_path.startswith("gs://"):
        client = storage.Client()
        uri = checkpoint_path[len("gs://"):]
        bucket_name, _, blob_name = uri.partition("/")
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        local_checkpoint = "/tmp/encoder_checkpoint.pth"
        blob.download_to_filename(local_checkpoint)
        checkpoint_path = local_checkpoint
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print(f"Loaded encoder model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best validation loss: {checkpoint.get('val_loss', 'unknown')}")
    
    return model

def calculate_psnr(pred, target):
    """Calculate PSNR between predicted and target images"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return -10.0 * torch.log10(mse + 1e-10)

def calculate_metrics(pred, target):
    """Calculate comprehensive image quality metrics"""
    # Clamp values to [0, 1] for metric calculations
    pred_clamped = torch.clamp(pred, 0, 1)
    target_clamped = torch.clamp(target, 0, 1)
    
    # MSE and PSNR
    mse = torch.mean((pred_clamped - target_clamped) ** 2).item()
    psnr = -10.0 * torch.log10(torch.tensor(mse + 1e-10)).item()
    
    # L1 loss (MAE)
    l1_loss = torch.mean(torch.abs(pred_clamped - target_clamped)).item()
    
    # SSIM and MS-SSIM
    ssim_val = ssim(pred_clamped, target_clamped, data_range=1.0, size_average=True).item()
    ms_ssim_val = ms_ssim(pred_clamped, target_clamped, data_range=1.0, size_average=True).item()
    
    return {
        'MSE': mse,
        'PSNR': psnr,
        'L1': l1_loss,
        'SSIM': ssim_val,
        'MS_SSIM': ms_ssim_val
    }

def run_encoder_inference(model, test_loader, device):
    """Run inference on test set and collect results"""
    all_metrics = []
    crf_values = []
    sample_images = []
    
    print("Running encoder proxy inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            if batch is None:
                continue
            x, crf, y_true = batch
            x = x.to(device)
            crf = crf.to(device) 
            y_true = y_true.to(device)
            
            # Generate reconstructed frame
            y_pred = model(x, crf)
            
            # Calculate metrics for each sample in batch
            for i in range(x.size(0)):
                metrics = calculate_metrics(y_pred[i:i+1], y_true[i:i+1])
                metrics['CRF'] = (crf[i] * 51.0).item()  # Unnormalize CRF
                all_metrics.append(metrics)
                crf_values.append(metrics['CRF'])
                
                # Save some sample images for visualization (first 10 batches)
                if batch_idx < 10 and i < 2:
                    sample_images.append({
                        'original': x[i].cpu(),
                        'target': y_true[i].cpu(),
                        'predicted': y_pred[i].cpu(),
                        'crf': metrics['CRF'],
                        'psnr': metrics['PSNR'],
                        'ssim': metrics['SSIM']
                    })
    
    return all_metrics, crf_values, sample_images

def analyze_by_crf(all_metrics):
    """Analyze performance metrics grouped by CRF values"""
    # Group metrics by CRF ranges
    crf_ranges = [
        (0, 17, 'High Quality (CRF 0-17)'),
        (18, 28, 'Medium Quality (CRF 18-28)'),
        (29, 39, 'Low Quality (CRF 29-39)'),
        (40, 51, 'Very Low Quality (CRF 40-51)')
    ]
    
    crf_analysis = {}
    
    for min_crf, max_crf, name in crf_ranges:
        # Filter metrics for this CRF range
        range_metrics = [m for m in all_metrics if min_crf <= m['CRF'] <= max_crf]
        
        if range_metrics:
            crf_analysis[name] = {
                'count': len(range_metrics),
                'avg_psnr': np.mean([m['PSNR'] for m in range_metrics]),
                'avg_ssim': np.mean([m['SSIM'] for m in range_metrics]),
                'avg_ms_ssim': np.mean([m['MS_SSIM'] for m in range_metrics]),
                'avg_l1': np.mean([m['L1'] for m in range_metrics]),
                'std_psnr': np.std([m['PSNR'] for m in range_metrics]),
                'std_ssim': np.std([m['SSIM'] for m in range_metrics])
            }
    
    return crf_analysis

def create_visualizations(all_metrics, crf_values, sample_images, output_dir):
    """Create comprehensive visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metric arrays
    psnr_values = [m['PSNR'] for m in all_metrics]
    ssim_values = [m['SSIM'] for m in all_metrics]
    ms_ssim_values = [m['MS_SSIM'] for m in all_metrics]
    l1_values = [m['L1'] for m in all_metrics]
    
    # 1. PSNR vs CRF scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(crf_values, psnr_values, alpha=0.6, s=20)
    plt.xlabel('CRF Value', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('Reconstruction Quality vs Compression Level', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(crf_values, psnr_values, 1)
    p = np.poly1d(z)
    plt.plot(sorted(crf_values), p(sorted(crf_values)), "r--", alpha=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / 'psnr_vs_crf.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. SSIM vs CRF scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(crf_values, ssim_values, alpha=0.6, s=20, color='green')
    plt.xlabel('CRF Value', fontsize=12)
    plt.ylabel('SSIM', fontsize=12)
    plt.title('Structural Similarity vs Compression Level', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'ssim_vs_crf.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Multi-metric comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # PSNR histogram
    axes[0, 0].hist(psnr_values, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('PSNR (dB)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('PSNR Distribution')
    axes[0, 0].axvline(np.mean(psnr_values), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(psnr_values):.2f} dB')
    axes[0, 0].legend()
    
    # SSIM histogram
    axes[0, 1].hist(ssim_values, bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[0, 1].set_xlabel('SSIM')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('SSIM Distribution')
    axes[0, 1].axvline(np.mean(ssim_values), color='red', linestyle='--',
                       label=f'Mean: {np.mean(ssim_values):.3f}')
    axes[0, 1].legend()
    
    # MS-SSIM histogram
    axes[1, 0].hist(ms_ssim_values, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1, 0].set_xlabel('MS-SSIM')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('MS-SSIM Distribution')
    axes[1, 0].axvline(np.mean(ms_ssim_values), color='red', linestyle='--',
                       label=f'Mean: {np.mean(ms_ssim_values):.3f}')
    axes[1, 0].legend()
    
    # L1 Loss histogram
    axes[1, 1].hist(l1_values, bins=50, alpha=0.7, edgecolor='black', color='purple')
    axes[1, 1].set_xlabel('L1 Loss')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('L1 Loss Distribution')
    axes[1, 1].axvline(np.mean(l1_values), color='red', linestyle='--',
                       label=f'Mean: {np.mean(l1_values):.4f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Sample reconstruction comparisons
    if sample_images:
        n_samples = min(6, len(sample_images))
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, sample in enumerate(sample_images[:n_samples]):
            # Original
            axes[i, 0].imshow(sample['original'].squeeze(), cmap='gray')
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            
            # Target (encoder output)
            axes[i, 1].imshow(sample['target'].squeeze(), cmap='gray')
            axes[i, 1].set_title(f'Encoder Output\n(CRF {sample["crf"]:.0f})')
            axes[i, 1].axis('off')
            
            # Predicted
            axes[i, 2].imshow(sample['predicted'].squeeze(), cmap='gray')
            axes[i, 2].set_title(f'Predicted\nPSNR: {sample["psnr"]:.1f}dB\nSSIM: {sample["ssim"]:.3f}')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'sample_reconstructions.png', dpi=300, bbox_inches='tight')
        plt.close()

def print_encoder_results(all_metrics, crf_analysis):
    """Print comprehensive encoder proxy results"""
    print("\n" + "="*70)
    print("           ENCODER PROXY MODEL EVALUATION RESULTS")
    print("="*70)
    
    # Overall statistics
    psnr_values = [m['PSNR'] for m in all_metrics]
    ssim_values = [m['SSIM'] for m in all_metrics]
    ms_ssim_values = [m['MS_SSIM'] for m in all_metrics]
    l1_values = [m['L1'] for m in all_metrics]
    
    print(f"\nOVERALL PERFORMANCE METRICS (n={len(all_metrics)}):")
    print(f"  Mean PSNR:                     {np.mean(psnr_values):.3f} ± {np.std(psnr_values):.3f} dB")
    print(f"  Mean SSIM:                     {np.mean(ssim_values):.4f} ± {np.std(ssim_values):.4f}")
    print(f"  Mean MS-SSIM:                  {np.mean(ms_ssim_values):.4f} ± {np.std(ms_ssim_values):.4f}")
    print(f"  Mean L1 Loss:                  {np.mean(l1_values):.6f} ± {np.std(l1_values):.6f}")
    
    print(f"\nPERFORMANCE BY CRF RANGE:")
    for range_name, stats in crf_analysis.items():
        print(f"  {range_name:25} (n={stats['count']:4d}):")
        print(f"    PSNR:    {stats['avg_psnr']:6.2f} ± {stats['std_psnr']:5.2f} dB")
        print(f"    SSIM:    {stats['avg_ssim']:6.4f} ± {stats['std_ssim']:6.4f}")
        print(f"    MS-SSIM: {stats['avg_ms_ssim']:6.4f}")
        print(f"    L1 Loss: {stats['avg_l1']:6.6f}")
    
    print(f"\nMODEL PERFORMANCE ASSESSMENT:")
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    if avg_psnr > 35:
        psnr_assessment = "Excellent"
    elif avg_psnr > 30:
        psnr_assessment = "Good" 
    elif avg_psnr > 25:
        psnr_assessment = "Fair"
    else:
        psnr_assessment = "Poor"
        
    if avg_ssim > 0.95:
        ssim_assessment = "Excellent"
    elif avg_ssim > 0.90:
        ssim_assessment = "Good"
    elif avg_ssim > 0.80:
        ssim_assessment = "Fair"
    else:
        ssim_assessment = "Poor"
    
    print(f"  Reconstruction Quality:        {psnr_assessment} (based on PSNR)")
    print(f"  Perceptual Quality:            {ssim_assessment} (based on SSIM)")
    print(f"  CRF Range Coverage:            {min([m['CRF'] for m in all_metrics]):.0f} - {max([m['CRF'] for m in all_metrics]):.0f}")
    
    print("\n" + "="*70)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Encoder Proxy Model")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV file with test data")
    parser.add_argument("--frame_dir", type=str, required=True, help="Root directory with video frames")
    parser.add_argument("--checkpoint", type=str, default="gs://encoder_proxy_training_checkpoints/checkpoints/best_checkpoint.pth",  help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./encoder_evaluation_results", help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--crop_size", type=int, default=256, help="Crop size for input frames")
    parser.add_argument("--bottleneck_channels", type=int, default=320, help="Model bottleneck channels")
    parser.add_argument("--film_hidden_dim", type=int, default=32, help="FiLM hidden dimension")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Everything is local (prefetched with gsutil rsync to /data/dataset)
    def _gs_to_local(p: str, local_root="/data"):
        if not isinstance(p, str) or not p.startswith("gs://"):
            return p
        # gs://<bucket>/<suffix>  → /data/<suffix>
        bucket_and_suffix = p[5:]
        _, _, suffix = bucket_and_suffix.partition("/")
        return f"{local_root}/{suffix}".rstrip("/")

    # Redirect any gs:// paths to the local mirror
    args.frame_dir = _gs_to_local(args.frame_dir, "/data")
    args.csv_path  = _gs_to_local(args.csv_path,  "/data")

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = EncoderDataset(
        args.csv_path,
        args.frame_dir,
        crop_size=args.crop_size,
        fixed_crop=True,  # Consistent evaluation
        split="Testing"
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=collate_skip_none,
        persistent_workers=(args.num_workers > 0)
    )
    
    print(f"Test dataset size: {len(test_dataset)} samples")
    
    # Load model
    print("Loading encoder proxy model...")
    model = load_encoder_model(
        args.checkpoint, 
        args.device, 
        args.bottleneck_channels, 
        args.film_hidden_dim
    )
    
    # Run inference
    all_metrics, crf_values, sample_images = run_encoder_inference(model, test_loader, args.device)
    
    if not all_metrics:
        print("ERROR: No valid predictions generated. Check your test data.")
        return
    
    print(f"Generated {len(all_metrics)} predictions")
    
    # Analyze by CRF
    crf_analysis = analyze_by_crf(all_metrics)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(all_metrics, crf_values, sample_images, args.output_dir)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results as JSON
    results = {
        'overall_metrics': {
            'mean_psnr': float(np.mean([m['PSNR'] for m in all_metrics])),
            'mean_ssim': float(np.mean([m['SSIM'] for m in all_metrics])),
            'mean_ms_ssim': float(np.mean([m['MS_SSIM'] for m in all_metrics])),
            'mean_l1': float(np.mean([m['L1'] for m in all_metrics])),
            'std_psnr': float(np.std([m['PSNR'] for m in all_metrics])),
            'std_ssim': float(np.std([m['SSIM'] for m in all_metrics]))
        },
        'crf_analysis': crf_analysis,
        'test_samples': len(all_metrics),
        'model_config': {
            'bottleneck_channels': args.bottleneck_channels,
            'film_hidden_dim': args.film_hidden_dim,
            'crop_size': args.crop_size
        }
    }
    
    with open(output_dir / 'encoder_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=_to_py)
    
    # Save raw data
    np.save(output_dir / 'all_metrics.npy', all_metrics)
    
    # Print results
    print_encoder_results(all_metrics, crf_analysis)
    
    print(f"\nResults saved to: {args.output_dir}")
    print("Generated files:")
    print("  - encoder_evaluation_results.json (detailed metrics)")
    print("  - psnr_vs_crf.png (quality vs compression)")
    print("  - ssim_vs_crf.png (perceptual quality vs compression)")
    print("  - metrics_distribution.png (metric histograms)")
    print("  - sample_reconstructions.png (visual comparisons)")
    print("  - all_metrics.npy (raw evaluation data)")

if __name__ == "__main__":
    main()