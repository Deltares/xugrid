"""
Provides a number of utilities for communicating with MeshKernel(Py)
"""
from enum import EnumMeta, IntEnum
from typing import Union

import meshkernel
import numpy as np
import shapely.geometry as sg


def either_string_or_enum(value: Union[str, IntEnum], enum_class: EnumMeta) -> IntEnum:
    """Convert to enum if needed, check value"""
    if isinstance(value, str):
        name = value.upper()
        enum_dict = dict(enum_class.__members__)
        try:
            value = enum_dict[name]
        except KeyError:
            valid_options = ", ".join(enum_dict.keys()).lower()
            raise ValueError(
                f"Invalid option: {value}. Valid options are: {valid_options}"
            )
    elif not isinstance(value, enum_class):
        raise TypeError(
            f"Option should be one of {enum_class}, received: {type(value)}"
        )
    return value


def to_geometry_list(polygon: sg.Polygon) -> meshkernel.GeometryList:
    if not isinstance(polygon, sg.Polygon):
        raise TypeError(
            "polygon must be a shapely.Polygon, received instead: " f"{type(polygon)}"
        )
    x, y = polygon.exterior.xy
    return meshkernel.GeometryList(np.array(x), np.array(y))
