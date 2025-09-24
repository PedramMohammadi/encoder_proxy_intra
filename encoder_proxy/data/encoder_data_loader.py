#!/usr/bin/env python3

import os
import random
import posixpath
from urllib.parse import urlparse

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

try:
    import gcsfs
except ImportError:
    gcsfs = None


def _as_posix(p: str) -> str:
    # Convert backslashes to forward slashes and strip redundant prefixes
    return str(p).replace("\\", "/")

def _split_gs(url: str):
    """
    Parse a gs:// URL.
    Returns (is_gcs, bucket, path) where path has no leading slash.
    """
    u = urlparse(url)
    if u.scheme in ("gs", "gcs"):
        return True, u.netloc, u.path.lstrip("/")
    return False, "", url.lstrip("/")


class EncoderDataset(Dataset):
    def __init__(self, csv_file, root_dir, crop_size=256, fixed_crop=False, split=None):
        # Read CSV with consistent dtypes (same as your original)
        self.data = pd.read_csv(
            csv_file,
            low_memory=False,
            dtype={
                "height": "int32",
                "width": "int32",
                "crf": "int32",
                "per_frame_bits": "int64",
                "frame_qp": "int32",
                "ref_frame_path": "string",
                "dist_frame_path": "string",
                "split": "string",
            }
        )

        # Optional split filter (same behavior)
        if split:
            self.data = self.data[self.data["split"] == split].reset_index(drop=True)

        # Normalize path-like columns to POSIX separators
        for col in ["ref_frame_path", "dist_frame_path"]:
            if col in self.data.columns:
                self.data[col] = self.data[col].astype(str).map(_as_posix)

        # Normalize and parse root_dir
        self.root_dir = _as_posix(root_dir)
        self.is_gcs, self.bucket, self.base_path = _split_gs(self.root_dir)

        # Filesystem for GCS
        self.fs = None
        if self.is_gcs:
            assert gcsfs is not None, "Please install gcsfs: pip install gcsfs"
            self.fs = gcsfs.GCSFileSystem()

        self.crop_size = crop_size
        self.fixed_crop = fixed_crop
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def _resolve_path(self, rel: str) -> str:
        """
        Returns a fully-resolved path usable by the appropriate backend:
          - If rel is gs://..., return that (normalized)
          - If root_dir is gs://..., join with posix rules and return gs://bucket/key
          - Else return a local filesystem path
        """
        rel = _as_posix(rel)

        # If CSV already contains a full gs:// path, use it directly.
        already_gcs, b, p = _split_gs(rel)
        if already_gcs:
            return f"gs://{b}/{p}"

        if self.is_gcs:
            # Join using POSIX to avoid backslashes
            key = posixpath.join(self.base_path, rel) if self.base_path else rel
            return f"gs://{self.bucket}/{key}"
        else:
            # Local path join is fine on Windows
            return os.path.join(self.root_dir, rel)

    def load_and_crop(self, relative_path, i, j):
        full_path = self._resolve_path(relative_path)
        full_path = str(full_path)

        if full_path.startswith(("gs://", "gcs://")):
            # --- GCS path: ensure a per-process client, then read via gcsfs ---
            if getattr(self, "fs", None) is None or getattr(self, "_fs_pid", None) != os.getpid():
                assert gcsfs is not None, "Please install gcsfs: pip install gcsfs"
                try:
                    self.fs = gcsfs.GCSFileSystem(token="google_default")
                except Exception:
                    self.fs = gcsfs.GCSFileSystem()
                self._fs_pid = os.getpid()

            with self.fs.open(full_path, "rb") as f:
                image = Image.open(f).convert("L")
        else:
            # --- Local path: read directly with PIL ---
            image = Image.open(full_path).convert("L")

        # To tensor + crop
        image = self.to_tensor(image)   # (1, H, W)
        _, H, W = image.shape
        i = max(0, min(i, max(0, H - self.crop_size)))
        j = max(0, min(j, max(0, W - self.crop_size)))
        return image[:, i:i + self.crop_size, j:j + self.crop_size]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        H, W = int(row["height"]), int(row["width"])

        if self.fixed_crop:
            i = max(0, (H - self.crop_size) // 2)
            j = max(0, (W - self.crop_size) // 2)
        else:
            i = random.randint(0, max(0, H - self.crop_size))
            j = random.randint(0, max(0, W - self.crop_size))

        # Load original and distorted frames
        x = self.load_and_crop(row["ref_frame_path"], i, j)
        y_true = self.load_and_crop(row["dist_frame_path"], i, j)

        # CRF for conditioning
        crf = torch.tensor([row["crf"] / 51.0], dtype=torch.float32)

        # Bits-per-pixel target
        frame_pixels = row["width"] * row["height"]

        return (x, crf, y_true)
