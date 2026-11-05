"""Test the fallback that is used when numba is not installed."""

import subprocess
import sys
import textwrap
import warnings

import pytest

from xugrid.constants import NoOpNumba, NumbaPerformanceWarning


def double(x):
    return 2 * x


@pytest.fixture(autouse=True)
def not_warned_yet(monkeypatch):
    # The warning is given only once per process.
    monkeypatch.setattr(NoOpNumba, "_warned", False)


@pytest.mark.parametrize(
    "decorate",
    [
        lambda f: NoOpNumba.njit(f),
        lambda f: NoOpNumba.njit()(f),
        lambda f: NoOpNumba.njit(cache=True)(f),
        lambda f: NoOpNumba.njit(parallel=True, cache=True)(f),
        lambda f: NoOpNumba.njit(f, parallel=True, cache=True),
    ],
)
def test_warns_on_call_not_on_decoration(decorate):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        func = decorate(double)

    assert func.__name__ == "double"
    with pytest.warns(NumbaPerformanceWarning, match="numba is not installed"):
        assert func(2) == 4


def test_warns_once():
    first = NoOpNumba.njit(double)
    second = NoOpNumba.njit(cache=True)(double)
    with pytest.warns(NumbaPerformanceWarning):
        first(2)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        first(2)
        second(2)


@pytest.mark.parametrize(
    "decorate",
    [
        lambda f: NoOpNumba.njit(inline="always")(f),
        lambda f: NoOpNumba.njit(f, inline="always"),
    ],
)
def test_inlined_function_is_returned_as_is(decorate):
    # These only run as part of another decorated function, which warns.
    assert decorate(double) is double


def test_warns_on_use_not_on_import():
    code = """
    import sys
    import warnings

    sys.modules["numba"] = None  # Make `import numba` raise an ImportError.
    warnings.filterwarnings("error", message="numba is not installed")

    import numpy as np
    import xugrid
    from xugrid.constants import NumbaPerformanceWarning

    grid = xugrid.Ugrid1d(
        node_x=np.array([0.0, 1.0, 2.0]),
        node_y=np.array([0.0, 0.0, 0.0]),
        fill_value=-1,
        edge_node_connectivity=np.array([[0, 1], [1, 2]]),
    )
    try:
        grid.topological_sort_by_dfs()
    except NumbaPerformanceWarning:
        sys.exit(0)
    sys.exit("Expected a NumbaPerformanceWarning")
    """
    subprocess.run([sys.executable, "-c", textwrap.dedent(code)], check=True)
