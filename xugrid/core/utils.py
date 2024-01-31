"""
Copied from xarray.core.utils.py

The reason is that the content of xarray.core.utils are all private methods.
Hence, Xarray provides no guarantees on breaking changes.

Xarray is licensed under Apache License 2.0:
https://github.com/pydata/xarray/blob/main/LICENSE
"""
from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any, TypeVar, cast

from xugrid.ugrid.ugridbase import UgridType

T = TypeVar("T")


def is_dict_like(value: Any) -> Any:
    return hasattr(value, "keys") and hasattr(value, "__getitem__")


def either_dict_or_kwargs(
    pos_kwargs: Mapping[Any, T] | None,
    kw_kwargs: Mapping[str, T],
    func_name: str,
) -> Mapping[Hashable, T]:
    if pos_kwargs is None or pos_kwargs == {}:
        # Need an explicit cast to appease mypy due to invariance; see
        # https://github.com/python/mypy/issues/6228
        return cast(Mapping[Hashable, T], kw_kwargs)

    if not is_dict_like(pos_kwargs):
        raise ValueError(f"the first argument to .{func_name} must be a dictionary")
    if kw_kwargs:
        raise ValueError(
            f"cannot specify both keyword and positional arguments to .{func_name}"
        )
    return pos_kwargs


# EDIT: copied and simplified.
class UncachedAccessor:
    """Acts like a property, but on both classes and class instances

    This class is necessary because some tools (e.g. pydoc and sphinx)
    inspect classes for which property returns itself and not the
    accessor.
    """

    def __init__(self, accessor: type) -> None:
        self._accessor = accessor

    def __get__(self, obj: None | object, cls) -> Any:
        if obj is None:
            return self._accessor

        return self._accessor(obj)  # type: ignore  # assume it is a valid accessor!


def partition(grids: list[UgridType]):
    parts = []
    for item in grids:
        for part in parts:
            if item.equals(part[0]):
                part.append(item)
                break
        else:
            parts.append([item])
    return parts


def unique_grids(grids: list[UgridType]):
    """
    Find uniques in list of unhashable elements.
    Source: https://stackoverflow.com/a/54964373
    """
    return [p[0] for p in partition(grids)]
