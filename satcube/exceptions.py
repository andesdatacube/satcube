"""Custom exceptions for satcube operations."""

from __future__ import annotations


class SatCubeError(Exception):
    """Base exception for all satcube operations."""

    pass


class AlignmentError(SatCubeError):
    """Image alignment failed."""

    pass


class CloudMaskingError(SatCubeError):
    """Cloud masking operation failed."""

    pass


class GapfillError(SatCubeError):
    """Gap filling operation failed."""

    pass


class SuperResolutionError(SatCubeError):
    """Super-resolution operation failed."""

    pass


class ModelLoadError(SatCubeError):
    """Failed to load ML model."""

    pass


class DeviceError(SatCubeError):
    """Device (CPU/GPU) management error."""

    pass
