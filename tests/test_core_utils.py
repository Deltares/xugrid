"""
Copied from xarray.tests.test_utils.py

The reason is that the content of xarray.core.utils are all private methods.
Hence, Xarray provides no guarantees on breaking changes.

Xarray is licensed under Apache License 2.0:
https://github.com/pydata/xarray/blob/main/LICENSE
"""
import numpy as np
import pytest

import xugrid
from xugrid.core.utils import either_dict_or_kwargs, unique_grids


def grid1d(dataset=None, indexes=None, crs=None, attrs=None):
    xy = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ]
    )
    grid = xugrid.Ugrid1d(
        node_x=xy[:, 0],
        node_y=xy[:, 1],
        fill_value=-1,
        edge_node_connectivity=np.array([[0, 1], [1, 2]]),
        dataset=dataset,
        indexes=indexes,
        crs=crs,
        attrs=attrs,
    )
    return grid


def test_either_dict_or_kwargs():
    result = either_dict_or_kwargs({"a": 1}, None, "foo")
    expected = {"a": 1}
    assert result == expected

    result = either_dict_or_kwargs(None, {"a": 1}, "foo")
    expected = {"a": 1}
    assert result == expected

    with pytest.raises(ValueError, match=r"foo"):
        result = either_dict_or_kwargs({"a": 1}, {"a": 1}, "foo")


def test_unique_grids():
    grid = grid1d()
    grid2 = grid1d()
    grid_different = grid1d()

    grid_different._attrs["something"] = "different"

    assert len(unique_grids([grid, grid2, grid_different])) == 2
    assert len(unique_grids([grid, grid2])) == 1
    assert len(unique_grids([grid, grid_different])) == 2
