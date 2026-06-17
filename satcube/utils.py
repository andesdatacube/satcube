from __future__ import annotations

import gc
import itertools
from typing import Any

import torch


def _reset_gpu() -> None:
    """Release CUDA memory and reset allocation statistics.

    Calling this on a system without a CUDA device is a no-op.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def define_iteration(dimension: tuple, chunk_size: int, overlap: int = 0):
    """
    Define the iteration strategy to walk through the image with an overlap.

    Args:
        dimension (tuple): Dimension of the S2 image.
        chunk_size (int): Size of the chunks.
        overlap (int): Size of the overlap between chunks.

    Returns:
        list: List of chunk coordinates.
    """
    dimy, dimx = dimension

    if chunk_size > max(dimx, dimy):
        return [(0, 0)]

    # Adjust step to create overlap
    y_step = chunk_size - overlap
    x_step = chunk_size - overlap

    # Generate initial chunk positions
    iterchunks = list(itertools.product(range(0, dimy, y_step), range(0, dimx, x_step)))

    # Fix chunks at the edges to stay within bounds
    iterchunks_fixed = fix_lastchunk(
        iterchunks=iterchunks, s2dim=dimension, chunk_size=chunk_size
    )

    return iterchunks_fixed


def fix_lastchunk(iterchunks, s2dim, chunk_size):
    """
    Fix the last chunk of the overlay to ensure it aligns with image boundaries.

    Args:
        iterchunks (list): List of chunks created by itertools.product.
        s2dim (tuple): Dimension of the S2 images.
        chunk_size (int): Size of the chunks.

    Returns:
        list: List of adjusted chunk coordinates.
    """
    itercontainer = []

    for index_i, index_j in iterchunks:
        # Adjust if the chunk extends beyond bounds
        if index_i + chunk_size > s2dim[0]:
            index_i = max(s2dim[0] - chunk_size, 0)
        if index_j + chunk_size > s2dim[1]:
            index_j = max(s2dim[1] - chunk_size, 0)

        itercontainer.append((index_i, index_j))

    return itercontainer


class DeviceManager:
    """Hold a compiled mlstac model and move it between devices on demand."""

    def __init__(self, experiment: Any, init_device: str = "cpu") -> None:
        """
        Parameters
        ----------
        experiment
            An mlstac experiment exposing ``compiled_model``.
        init_device
            Device where the model is first compiled, e.g. ``"cpu"`` or
            ``"cuda:0"``.
        """
        self._experiment: Any = experiment
        self.device: str | None = None
        self.model: torch.nn.Module | None = None
        self.switch(init_device)

    def switch(self, new_device: str) -> torch.nn.Module:
        """Return a model compiled for *new_device*, recompiling if needed.

        Parameters
        ----------
        new_device
            Target device identifier.

        Returns
        -------
        torch.nn.Module
            The model resident on *new_device*.

        Raises
        ------
        AssertionError
            If *new_device* requests CUDA but no GPU is available.
        """
        if new_device == self.device:
            return self.model  # type: ignore[return-value]

        if self.model is not None:
            del self.model
        gc.collect()

        if self.device == "cuda":
            _reset_gpu()

        if new_device == "cuda":
            assert torch.cuda.is_available(), "CUDA device not detected"

        print(f"→ Compiling model on {new_device} …")
        self.model = self._experiment.compiled_model(device=new_device, mode="max")
        self.device = new_device
        return self.model
