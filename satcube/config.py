"""Configuration constants for satcube."""

from __future__ import annotations

import os
import pathlib
from typing import Final

# Chunk/Patch Sizes
CLOUD_CHUNK_SIZE: Final[int] = 512
CLOUD_OVERLAP: Final[int] = 32
SR_INPUT_CHUNK: Final[int] = 128
SR_INPUT_OVERLAP: Final[int] = 32

# Processing Thresholds
GAPFILL_STRICT_THRESHOLD: Final[float] = 0.15
GAPFILL_RELAXED_THRESHOLD: Final[float] = 0.40
DESPIKE_THRESHOLD: Final[float] = 0.15

# Alignment Parameters
ALIGN_OVERLAP_RATIO: Final[float] = 0.5
ALIGN_UPSAMPLE_FACTOR: Final[int] = 10

# Device Management
DEFAULT_DEVICE: Final[str] = "cpu"
DEFAULT_BATCH_SIZE: Final[int] = 4
DEFAULT_WORKERS: Final[int] = 4

# Model URLs
CLOUDSEN12_URL: Final[str] = (
    "https://huggingface.co/tacofoundation/CloudSEN12-models/"
    "resolve/main/SEN2CloudEnsemble/mlm.json"
)

SEN2SR_URLS: Final[dict[str, str]] = {
    "SEN2SRLite": (
        "https://huggingface.co/tacofoundation/sen2sr/"
        "resolve/main/SEN2SRLite/main/mlm.json"
    ),
    "SEN2SR": (
        "https://huggingface.co/tacofoundation/sen2sr/"
        "resolve/main/SEN2SR/main/mlm.json"
    ),
}

# Cache Directory
CACHE_DIR: Final[pathlib.Path] = pathlib.Path(
    os.getenv("SATCUBE_CACHE", "~/.satcube_cache")
).expanduser()

CACHE_DIR.mkdir(exist_ok=True, parents=True)
