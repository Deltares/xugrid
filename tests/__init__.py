# __init__.py for pytest-cov
import importlib

import pytest


def _importorskip(modname):
    try:
        importlib.import_module(modname)
        has = True
    except ImportError:
        has = False
    func = pytest.mark.skipif(not has, reason=f"requires {modname}")
    return has, func


has_meshkernel, requires_meshkernel = _importorskip("meshkernel")
