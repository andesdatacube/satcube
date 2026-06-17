from satcube import _quiet  # noqa: F401

from importlib.metadata import PackageNotFoundError, version as _version

from satcube.download import metadata, metadata_polygon
from satcube.objects import SatCubeMetadata

from_directory = SatCubeMetadata.from_directory

__all__ = ["SatCubeMetadata", "from_directory", "metadata", "metadata_polygon"]

try:
    __version__ = _version("satcube")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"