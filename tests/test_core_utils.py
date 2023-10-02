"""
Copied from xarray.tests.test_utils.py

The reason is that the content of xarray.core.utils are all private methods.
Hence, Xarray provides no guarantees on breaking changes.

Xarray is licensed under Apache License 2.0:
https://github.com/pydata/xarray/blob/main/LICENSE
"""
import pytest

from xugrid.core.utils import either_dict_or_kwargs


def test_either_dict_or_kwargs():
    result = either_dict_or_kwargs(dict(a=1), None, "foo")
    expected = dict(a=1)
    assert result == expected

    result = either_dict_or_kwargs(None, dict(a=1), "foo")
    expected = dict(a=1)
    assert result == expected

    with pytest.raises(ValueError, match=r"foo"):
        result = either_dict_or_kwargs(dict(a=1), dict(a=1), "foo")
