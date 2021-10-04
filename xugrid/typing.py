from typing import NamedTuple

import numpy as np

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
