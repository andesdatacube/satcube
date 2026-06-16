from satcube.download import metadata, metadata_polygon
from satcube.objects import SatCubeMetadata

from_directory = SatCubeMetadata.from_directory

__all__ = ["SatCubeMetadata", "from_directory", "metadata", "metadata_polygon"]

try:
    import importlib as _importlib
    _importlib.import_module("satcube._quiet")
except Exception:
    __version__ = "0.0.0-dev"
