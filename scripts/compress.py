#!/usr/bin/env python3

import argparse
import os
import subprocess
from pathlib import Path
import json
import csv

# ------------------------------
# CLI Argument Parsing
# ------------------------------
parser = argparse.ArgumentParser(description="Compress video files using FFmpeg with CRF values.")
parser.add_argument("--input_dir", type=str, required=True, help="Base input directory containing Training, Validation, Testing subfolders")
parser.add_argument("--output_dir", type=str, required=True, help="Base output directory to store compressed files")
parser.add_argument("--crf", type=int, nargs="+", required=True, help="List of CRF values to use")
parser.add_argument("--input_formats", nargs="+", default=["yuv", "mp4", "mkv", "y4m"], help="Accepted input formats")
parser.add_argument("--width", type=int, help="Width of input (required for .yuv)")
parser.add_argument("--height", type=int, help="Height of input (required for .yuv)")
parser.add_argument("--fps", type=int, help="frame rate of input (required for .yuv)")
parser.add_argument("--ffmpeg_dir", type=str, default="", help="Directory containing ffmpeg binary")
args = parser.parse_args()

# ------------------------------
# Set ffmpeg binary path
# ------------------------------
ffmpeg_bin = os.path.join(args.ffmpeg_dir, "ffmpeg") if args.ffmpeg_dir else "ffmpeg"

# ------------------------------
# Encoder Map (Can extend to other encoders later on)
# ------------------------------
encoders = {
    "x265": "libx265",
}

# ------------------------------
# Helper: Extract metadata via ffprobe
# ------------------------------
def get_metadata_with_ffprobe(filepath):
    try:
        cmd = [
            os.path.join(args.ffmpeg_dir, "ffprobe") if args.ffmpeg_dir else "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "json", str(filepath)
        ]
        output = subprocess.check_output(cmd).decode()
        info = json.loads(output)['streams'][0]
        width = info['width']
        height = info['height']
        fps = eval(info['r_frame_rate'])  # e.g., "30/1"
        return str(width), str(height), str(int(fps))
    except Exception as e:
        print(f"Could not get metadata from {filepath}: {e}")
        return None, None, None

# ------------------------------
# Main Loop
# ------------------------------
splits = ["Training", "Validation", "Testing"]
for split in splits:
    input_split_dir = Path(args.input_dir) / split
    output_split_dir = Path(args.output_dir) / split
    output_split_dir.mkdir(parents=True, exist_ok=True)

    for file in input_split_dir.iterdir():
        ext = file.suffix.lower().lstrip('.')
        if ext not in args.input_formats:
            continue

        name = file.stem
        input_path = file

        # Get metadata
        if ext == "yuv":
            if not all([args.width, args.height, args.fps]):
                print(f"Skipping {file.name}: .yuv requires --width, --height, and --fps")
                continue
            w, h, fps = str(args.width), str(args.height), str(args.fps)
        else:
            w, h, fps = get_metadata_with_ffprobe(input_path)
            if not all([w, h, fps]):
                print(f"Skipping {file.name}: could not extract metadata")
                continue

        # Compress with each CRF
        for crf_val in args.crf:
            output_file = f"{name}_crf{crf_val}.mp4"
            output_path = output_split_dir / output_file

            cmd = [ffmpeg_bin, "-y"]

            if ext == "yuv":
                cmd += [
                    "-f", "rawvideo",
                    "-s:v", f"{w}x{h}",
                    "-r", str(fps),
                ]

            cmd += [
                "-i", str(input_path),
                "-pix_fmt", "yuv420p",
                "-c:v", encoders[args.encoder],
                "-crf", str(crf_val),
            ]

            log_file = output_split_dir / f"{name}_crf{crf_val}_x265_log.csv"
            cmd += [
                "-x265-params",
                f"keyint=1:min-keyint=1:scenecut=0:bframes=0:csv={log_file}:csv-log-level=2"
            ]

            cmd += [str(output_path)]

            print(f"Compressing {file.name} (split={split}, CRF={crf_val}) -> {output_file}")
            subprocess.run(cmd, check=True)

            # --- Extract QPs ---
            qp_csv = output_split_dir / f"{name}_crf{crf_val}_qp_bits.csv"

            with open(qp_csv, "w", newline="") as out_f:
                writer = csv.writer(out_f)
                writer.writerow(["frame_num", "qp", "bits"])

                # Skip header row, extract QP from column 2
                with open(log_file, "r") as f:
                    next(f)  # skip header
                    for i, line in enumerate(f):
                        cols = line.strip().split(",")
                        if len(cols) > 1:
                            qp = round(float(cols[3]))
                            bits = round(float(cols[4]))
                            writer.writerow([i, qp, bits])

            # --- Cleanup original logs ---
            os.remove(log_file)

print("All compression complete.")
