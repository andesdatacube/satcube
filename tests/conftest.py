"""Shared pytest fixtures and EE-skip logic for satcube tests."""

from __future__ import annotations

import pytest

try:
    import ee

    ee.Initialize(project="ee-contrerasnetk")
    _EE_AVAILABLE = True
except Exception:
    _EE_AVAILABLE = False


def pytest_collection_modifyitems(config, items):
    """Skip tests marked needs_ee when Earth Engine isn't available."""
    if _EE_AVAILABLE:
        return
    skip_ee = pytest.mark.skip(reason="Earth Engine not available")
    for item in items:
        if "needs_ee" in item.keywords:
            item.add_marker(skip_ee)