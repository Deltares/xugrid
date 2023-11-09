# __init__.py for pytest-cov
import pytest


def _importorskip(modname):
    try:
        import meshkernel

        # If the DLL/SO fails to load / be found, still skip.
        try:
            meshkernel.MeshKernel(is_geographic=False)
            has = True
        except OSError:
            has = False
    except ImportError:
        has = False
    func = pytest.mark.skipif(not has, reason=f"requires {modname}")
    return has, func


has_meshkernel, requires_meshkernel = _importorskip("meshkernel")
