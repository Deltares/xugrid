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
