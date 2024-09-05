from typing import NamedTuple, Union

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

# import numpy.typing as npt

FloatDType = np.float64
IntDType = np.intp

# Requires numpy 1.21, not on conda yet...
# FloatArray = np.ndarray[FloatDType]
# IntArray = np.ndarray[IntDType]
# BoolArray = np.ndarray[np.bool_]

FloatArray = np.ndarray
IntArray = np.ndarray
BoolArray = np.ndarray
# Pygeos collections:
PointArray = np.ndarray
LineArray = np.ndarray
PolygonArray = np.ndarray
SparseMatrix = Union[coo_matrix, csr_matrix]

# Internally we always use a fill value of -1. This ensures we can always index
# with the fill value as well, since any array will have at least size 1.
FILL_VALUE = -1


class Point(NamedTuple):
    x: float
    y: float


class Vector(NamedTuple):
    x: float
    y: float


# Spatial coordinate epsilon for floating point comparison
# Assuming world coordinates in meters: 40 000 m along equator:
# 40 000 000 = 4e7 mm
# np.spacing(4e7) == 7.45E-9 ~= 1E-8
X_EPSILON = 1.0e-8
X_OFFSET = 1.0e-8
T_OFFSET = 1.0e-6


class MissingOptionalModule:
    """Presents a clear error for optional modules."""

    def __init__(self, name):
        self.name = name

    def __getattr__(self, name):
        raise ImportError(f"{self.name} is required for this functionality")
