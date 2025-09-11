#!/usr/bin/env python3

import os
import json
import argparse
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# ------------------------------
# CLI Argument Parsing
# ------------------------------
parser = argparse.ArgumentParser(description="Generate metadata CSV for encoder proxy training.")
parser.add_argument("--compressed_dir", required=True, help="Directory with compressed .mp4 files")
parser.add_argument("--frame_dir", required=True, help="Directory with extracted Y-channel frames")
parser.add_argument("--output_csv", required=True, help="Path to output CSV file")
parser.add_argument("--enable_balancing", action="store_true", help="Enable sampling across bitrates")
parser.add_argument("--clips_per_bin", type=int, default=40, help="Clips per bitrate bin")
parser.add_argument("--num_bins", type=int, default=10, help="Number of bitrate bins")
parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--ffmpeg_dir", type=str, default="", help="Directory containing ffmpeg and ffprobe binaries")
args = parser.parse_args()

# ------------------------------
# Setup
# ------------------------------
ffprobe_bin = os.path.join(args.ffmpeg_dir, "ffprobe") if args.ffmpeg_dir else "ffprobe"
ffmpeg_bin = os.path.join(args.ffmpeg_dir, "ffmpeg") if args.ffmpeg_dir else "ffmpeg"

random.seed(args.random_seed)
np.random.seed(args.random_seed)

# ------------------------------
# Metadata Extraction
# ------------------------------
def get_video_metadata(ffprobe_bin, video_path):
    try:
        cmd = [
            ffprobe_bin, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate:format=duration",
            "-of", "json", str(video_path)
        ]
        output = subprocess.check_output(cmd).decode()
        info = json.loads(output)
        stream = info["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
        fps = round(eval(stream["r_frame_rate"]))
        duration = float(info["format"].get("duration", 0))
        frame_count = int(duration * fps) if duration > 0 else 0
        return {"width": width, "height": height, "fps": fps, "duration": duration, "frame_count": frame_count}
    except Exception as e:
        print(f"{video_path}: {e}")
        return {"width": 0, "height": 0, "fps": 0, "duration": 0, "frame_count": 0}

def load_qp_and_bits_from_csv(qp_and_bits_csv_path):
    try:
        df = pd.read_csv(qp_and_bits_csv_path)

        # Normalize column names (handle variations)
        cols_lower = {c.lower(): c for c in df.columns}
        # frame index
        frame_col = (
            cols_lower.get("frame_num")
        )
        # qp
        qp_col = (
            cols_lower.get("qp")
        )
        # bits
        bits_col = (
            cols_lower.get("bits")
        )

        if not (frame_col and qp_col and bits_col):
            raise ValueError(
                f"Expected columns like frame_num, qp, bits; "
                f"got {list(df.columns)}"
            )

        # Build mapping: frame_idx -> (qp, bits)
        return {
            int(row[frame_col]): (float(row[qp_col]), float(row[bits_col]))
            for _, row in df[[frame_col, qp_col, bits_col]].iterrows()
        }

    except Exception as e:
        print(f"{qp_and_bits_csv_path}: {e}")
        return {}

# ------------------------------
# Bitrate Balancing
# ------------------------------
def apply_bitrate_balancing(df, clips_per_bin, num_bins, seed):
    print("Balancing bitrate distribution across clips...")
    df["clip_id"] = df["sequence_name"] + "_" + "_crf" + df["crf"].astype(str)
    avg_df = df.groupby("clip_id")["bitrate_kbps"].mean().reset_index()
    avg_df.rename(columns={"bitrate_kbps": "average_bitrate_kbps"}, inplace=True)

    # Dynamic bin edges based on bitrate range
    min_bitrate = avg_df["average_bitrate_kbps"].min()
    max_bitrate = avg_df["average_bitrate_kbps"].max()
    bins = np.linspace(min_bitrate, max_bitrate, num_bins + 1)
    avg_df["bin"] = pd.cut(avg_df["average_bitrate_kbps"], bins=bins, labels=False, include_lowest=True)
    avg_df["crf"] = avg_df["clip_id"].str.extract(r"_crf(\d+)$").astype(int)
    
    # Plot histogram BEFORE sampling
    plt.figure(figsize=(10, 5))
    plt.hist(avg_df["average_bitrate_kbps"], bins=bins, edgecolor='black', alpha=0.5, label="Before Sampling")

    # Sample from each bin
    sampled_dfs = []
    for b in range(num_bins):
        bin_group = avg_df[avg_df["bin"] == b]
        bin_size = len(bin_group)

        if bin_size == 0:
            continue
        elif bin_size <= clips_per_bin:
            print(f"\nBin {b} only has {bin_size} clips but {clips_per_bin} were requested. Including all.")
            sampled = bin_group
        else:
            sampled = bin_group.sample(n=clips_per_bin, random_state=seed)

        sampled_dfs.append(sampled)

    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    sampled_df["bin"] = pd.cut(sampled_df["average_bitrate_kbps"], bins=bins, labels=False, include_lowest=True)

    # Histogram AFTER sampling
    plt.hist(sampled_df["average_bitrate_kbps"], bins=bins, edgecolor='black', alpha=0.7, label="After Sampling")
    plt.title("Histogram of Average bitrate in Kbps per Clip (Before and After Sampling)")
    plt.xlabel("Bitrate (Kbps)")
    plt.ylabel("Number of Clips")
    plt.grid(True)
    plt.legend()
    plt.xticks(bins, rotation=45)
    plt.tight_layout()
    plt.savefig("bitrate_distribution.png")
    print("\nSaved histogram to bitrate_distribution.png")

    # Final filtering for full dataset
    df["clip_id"] = df["sequence_name"] + "_" + "_crf" + df["crf"].astype(str)
    filtered = df[df["clip_id"].isin(sampled_df["clip_id"])].copy()
    print(f"\nSelected {filtered['clip_id'].nunique()} clips across {num_bins} bins.")
    return filtered

# ------------------------------
# Metadata CSV Generation
# ------------------------------
compressed_dir = Path(args.compressed_dir)
frame_dir = Path(args.frame_dir)
data = []

#Currently, it's assumed the compressed sequences are stored as *.mp4. Can extend to other formats in the future.
for mp4_path in compressed_dir.rglob("*.mp4"):
    try:
        split = mp4_path.relative_to(compressed_dir).parts[0]
        stem = mp4_path.stem
        if "_crf" not in stem:
            print(f"Skipping unrecognized filename: {mp4_path.name}")
            continue

        video_name, crf_str = stem.rsplit("_crf", 1)
        crf = int(crf_str)

        ref_dir = frame_dir / "Original" / split / video_name
        dist_dir = frame_dir / "Compressed" / split / video_name / f"crf{crf}"

        if not ref_dir.exists() or not dist_dir.exists():
            print(f"Missing reference or distorted frame dir for {video_name}, CRF {crf}")
            continue

        ref_frames = sorted(list(ref_dir.glob("frame_*.png")))
        dist_frames = sorted(list(dist_dir.glob("frame_*.png")))
        
        if len(ref_frames) == 0 or len(dist_frames) == 0:
            print(f"No reference frames or mismatched for {video_name}, CRF {crf}")
            continue

        metadata = get_video_metadata(ffprobe_bin, mp4_path)
        
        if metadata["fps"] == 0 or metadata["duration"] == 0:
            print(f"Invalid metadata for {mp4_path}")
            continue

        compressed_size = os.path.getsize(mp4_path)
        frame_count = metadata["frame_count"]  # Use ffprobe frame count
        
        if frame_count != len(ref_frames):
            print(f"Frame count mismatch for {video_name}, CRF {crf}: ffprobe={frame_count}, ref_dir={len(ref_frames)}")
            frame_count = min(frame_count, len(ref_frames))  # Use smaller to avoid index errors

        # Load pre-extracted QP CSV
        qp_and_bits_csv_path = mp4_path.with_name(f"{stem}_qp_bits.csv")
        qp_and_bits_dict = load_qp_and_bits_from_csv(qp_and_bits_csv_path) if qp_and_bits_csv_path.exists() else {}
        if not qp_and_bits_dict:
            print(f"QP and bits CSV not found or empty for {mp4_path}; using -1 for all frames")

        min_len = min(len(ref_frames), len(dist_frames), frame_count if frame_count > 0 else len(ref_frames))

        if len(ref_frames) != len(dist_frames) or (frame_count > 0 and min_len != frame_count):
            print(f"[warn] Using min length for {video_name}, CRF {crf} "
                f"(ref={len(ref_frames)}, dist={len(dist_frames)}, ffprobe={frame_count} → min={min_len})")


        # Frame-level entries
        for frame_idx in range(min_len):
            ref_path  = ref_frames[frame_idx]
            dist_path = dist_frames[frame_idx]
            if frame_idx >= frame_count:
                break  # Avoid excess frames
            qp_value, per_frame_bits = qp_and_bits_dict.get(frame_idx, (-1, -1))
            bitrate_kbps = per_frame_bits * metadata["fps"] // 1000
            data.append({
                "split": split,
                "sequence_name": video_name,
                "frame_number": frame_idx,
                "crf": crf,
                "frame_qp": qp_value,
                "ref_frame_path": str(ref_path),
                "dist_frame_path": str(dist_path),
                "compressed_size_bytes": compressed_size,
                "bitrate_kbps": round(bitrate_kbps, 2),
                "per_frame_bits": round(per_frame_bits, 2),
                "width": metadata["width"],
                "height": metadata["height"],
                "fps": metadata["fps"]
            })

    except Exception as e:
        print(f"Skipping {mp4_path.name} due to: {e}")
        continue

df = pd.DataFrame(data)
df.sort_values(by=["split", "sequence_name", "crf", "frame_number"], inplace=True)
# Apply bitrate balancing
if args.enable_balancing:
    df_train = df[df["split"].str.lower() == "training"].copy()
    df_rest  = df[df["split"].str.lower() != "training"].copy()
    df_train = apply_bitrate_balancing(df_train, args.clips_per_bin, args.num_bins, args.random_seed)
    df = pd.concat([df_train, df_rest], ignore_index=True)
df.to_csv(args.output_csv, index=False)
print(f"Saved CSV with {len(df)} entries to: {args.output_csv}")