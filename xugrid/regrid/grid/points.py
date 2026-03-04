"""Points: only suitable as a target, not as a source."""

from __future__ import annotations

import xarray as xr

from xugrid.constants import FloatArray
from xugrid.regrid.grid.basegrid import Grid


class Points2d(Grid):
    def __init__(self, xy: FloatArray, dim: str = "points", name_x="x", name_y="y"):
        self.xy = xy
        self._size = xy.shape[0]
        self._shape = (self.size,)
        self._dims = (dim,)
        self._name_x = name_x
        self._name_y = name_y

    @property
    def size(self) -> int:
        return self._size

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def dims(self) -> tuple:
        return self._dims

    @property
    def ndim(self) -> int:
        return 1

    @property
    def facet(self) -> str:
        return "node"

    @property
    def coords(self) -> dict:
        x, y = self.xy.T
        dim = self.dims[0]
        return {
            self._name_x: (dim, x),
            self._name_y: (dim, y),
        }

    @property
    def coordinates(self) -> FloatArray:
        return self.xy

    def to_dataset(self, name: str) -> xr.Dataset:
        dim = self.dims[0]
        return xr.Dataset(
            {
                f"{name}_{self.name_x}": (dim, self.xy[:, 0]),
                f"{name}_{self.name_y}": (dim, self.xy[:, 1]),
                f"{name}_type": xr.DataArray(0, attrs={"type": "Points2d"}),
            }
        )
