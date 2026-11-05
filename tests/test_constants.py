import warnings

import pytest

from xugrid.constants import NoOpNumba, NumbaPerformanceWarning


def test_noop_numba_warns_on_call(monkeypatch):
    monkeypatch.setattr(NoOpNumba, "_warned", False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        func = NoOpNumba.njit(lambda x: x)

    with pytest.warns(NumbaPerformanceWarning):
        assert func(1) == 1
