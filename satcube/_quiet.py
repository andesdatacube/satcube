"""
_quiet.py - Silence all ML framework noise on import.

Silences:
  • TensorFlow C++ logs and oneDNN banners
  • PyTorch FutureWarnings and deprecation notices
  • Rasterio NotGeoreferencedWarning and other warnings
  • GLOG/ABSL logging (cuDNN/cuBLAS registration messages)
  • Satalign warnings (no matching points, translation too large)
  • Kornia/LightGlue deprecation warnings
  • HuggingFace token warnings
  • tqdm nested loop warnings

Works by:
  1. Setting environment variables BEFORE any imports
  2. Configuring warning filters
  3. Redirecting file descriptor 2 to /dev/null during TF import
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import os
import warnings

# =============================================================================
# 1. ENVIRONMENT VARIABLES (must be set BEFORE importing frameworks)
# =============================================================================

# TensorFlow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # 0=all, 1=info, 2=warn, 3=error
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # Disable oneDNN optimizations msg

# GLOG (used by TF internally)
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("FLAGS_minloglevel", "3")

# PyTorch CUDA
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# HuggingFace
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Hydra (if used)
os.environ.setdefault("HYDRA_FULL_ERROR", "0")

# GDAL/Rasterio
os.environ.setdefault("GDAL_PAM_ENABLED", "NO")  # Disable .aux.xml sidecar files
os.environ.setdefault("CPL_LOG", "/dev/null")  # Silence GDAL logs

# =============================================================================
# 2. ABSL LOGGING (TensorFlow's logging backend)
# =============================================================================

try:
    from absl import logging as absl_logging

    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.set_stderrthreshold("error")
except ImportError:
    pass

# =============================================================================
# 3. PYTHON LOGGING CONFIGURATION
# =============================================================================

# Silence common noisy loggers
for logger_name in [
    "tensorflow",
    "tensorflow.python",
    "h5py",
    "matplotlib",
    "PIL",
    "rasterio",
    "fiona",
    "pyproj",
    "shapely",
    "urllib3",
    "requests",
    "numba",
    "kornia",
    "lightglue",
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# =============================================================================
# 4. WARNING FILTERS
# =============================================================================

# --- TensorFlow / Keras ---
warnings.filterwarnings("ignore", category=FutureWarning, module=r"tensorflow(\.|$)")
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module=r"tensorflow(\.|$)"
)
warnings.filterwarnings("ignore", message=r".*tf\.keras.*", category=UserWarning)

# --- PyTorch ---
warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch(\.|$)")
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch(\.|$)")
warnings.filterwarnings(
    "ignore", message=r".*torch\.cuda\.amp.*", category=FutureWarning
)

# --- Kornia / LightGlue (satalign dependency) ---
warnings.filterwarnings(
    "ignore", message=r".*custom_fwd.*deprecated", category=FutureWarning
)
warnings.filterwarnings(
    "ignore", message=r".*custom_bwd.*deprecated", category=FutureWarning
)
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"kornia(\.|$)")
warnings.filterwarnings("ignore", category=UserWarning, module=r"kornia(\.|$)")
warnings.filterwarnings("ignore", category=UserWarning, module=r"lightglue(\.|$)")

# --- Rasterio / GDAL ---
warnings.filterwarnings("ignore", category=UserWarning, module=r"rasterio(\.|$)")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"rasterio(\.|$)")
warnings.filterwarnings("ignore", message=r".*NotGeoreferencedWarning.*")
warnings.filterwarnings("ignore", message=r".*The given matrix is not.*")
warnings.filterwarnings("ignore", message=r".*Dataset has no geotransform.*")

# --- Satalign ---

warnings.filterwarnings(
    "ignore",
    message=r".*Estimated translation is too large.*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=r".*No matching points found.*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=r".*translation.*too large.*",
    category=UserWarning,
)

# --- HuggingFace ---
warnings.filterwarnings("ignore", message=r"The secret HF_TOKEN", category=UserWarning)
warnings.filterwarnings(
    "ignore", message=r".*huggingface.*token.*", category=UserWarning
)

# --- NumPy ---
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"numpy(\.|$)")
warnings.filterwarnings("ignore", message=r".*invalid value encountered.*")
warnings.filterwarnings("ignore", message=r".*divide by zero.*")
warnings.filterwarnings("ignore", message=r".*overflow encountered.*")

# --- Pandas ---
warnings.filterwarnings("ignore", category=FutureWarning, module=r"pandas(\.|$)")

# --- Scipy ---
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"scipy(\.|$)")

# --- tqdm ---
try:
    from tqdm import TqdmWarning

    warnings.filterwarnings(
        "ignore",
        category=TqdmWarning,
        module=r"tqdm(\.|$)",
    )
except ImportError:
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"tqdm(\.|$)",
    )

# --- OpenCV ---
warnings.filterwarnings("ignore", category=UserWarning, module=r"cv2(\.|$)")

# --- Generic fallbacks ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r".*is deprecated.*")

# =============================================================================
# 5. FILE DESCRIPTOR REDIRECT (for C++ level logs)
# =============================================================================


class _MuteFD2(contextlib.AbstractContextManager):
    """
    Context manager to temporarily redirect stderr (fd 2) to /dev/null.

    This catches C++ level logs that Python's logging/warnings can't intercept,
    such as TensorFlow's CUDA initialization messages.
    """

    def __enter__(self):
        try:
            self._null = os.open(os.devnull, os.O_WRONLY)
            self._saved = os.dup(2)
            os.dup2(self._null, 2)
        except OSError:
            self._null = None
            self._saved = None
        return self

    def __exit__(self, *exc):
        if self._saved is not None:
            try:
                os.dup2(self._saved, 2)
                os.close(self._null)
                os.close(self._saved)
            except OSError:
                pass


# =============================================================================
# 6. QUIET IMPORTS (import noisy packages with stderr muted)
# =============================================================================


def _quiet_imports():
    """Import noisy frameworks with stderr temporarily muted."""
    with _MuteFD2():
        # TensorFlow (most verbose)
        with contextlib.suppress(ImportError):
            importlib.import_module("tensorflow")

        # PyTorch
        with contextlib.suppress(ImportError):
            importlib.import_module("torch")

        # Rasterio (GDAL backend can be noisy)
        with contextlib.suppress(ImportError):
            importlib.import_module("rasterio")


# Execute quiet imports
_quiet_imports()

# =============================================================================
# 7. POST-IMPORT LOGGING CLEANUP
# =============================================================================

# Re-apply after imports (some packages reset their loggers on import)
try:
    import tensorflow as tf

    tf.get_logger().setLevel(logging.ERROR)
    tf.autograph.set_verbosity(0)
except (ImportError, AttributeError):
    pass

try:
    # Disable torch autograd profiler warnings
    import torch

    torch.set_warn_always(False)
except (ImportError, AttributeError):
    pass
