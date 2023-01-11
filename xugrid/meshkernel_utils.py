"""
Provides a number of utilities for communicating with MeshKernel(Py)
"""
from enum import EnumMeta, IntEnum
from typing import Union

import numpy as np


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


def to_geometry_list(
    polygon: Union["shapely.geometry.Polygon", "pygeos.Geometry"]  # type: ignore # noqa
) -> "meshkernel.GeometryList":  # type: ignore # noqa
    import meshkernel
    import pygeos

    from xugrid.conversion import _to_pygeos

    polygon = _to_pygeos([polygon])[0]
    xy = pygeos.get_coordinates(pygeos.get_exterior_ring(polygon))
    return meshkernel.GeometryList(np.array(xy[:, 0]), np.array(xy[:, 1]))
