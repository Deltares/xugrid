from enum import IntEnum
from unittest.mock import MagicMock

import pytest
import shapely.geometry as sg

from xugrid import meshkernel_utils as mku
from xugrid.constants import MissingOptionalModule

from . import requires_meshkernel

try:
    import meshkernel as mk

except ImportError:
    mk = MagicMock()
    mk.RefinementType = IntEnum(
        "RefinementType", ["WAVE_COURANT", "REFINEMENT_LEVELS", "RIDGE_DETECTION"]
    )


class Dummy(IntEnum):
    A = 1
    B = 2
    C = 3


def test_either_string_or_enum():
    assert (
        mku.either_string_or_enum("wave_courant", mk.RefinementType)
        == mk.RefinementType.WAVE_COURANT
    )
    assert (
        mku.either_string_or_enum("WAVE_COURANT", mk.RefinementType)
        == mk.RefinementType.WAVE_COURANT
    )
    assert (
        mku.either_string_or_enum("refinement_levels", mk.RefinementType)
        == mk.RefinementType.REFINEMENT_LEVELS
    )
    with pytest.raises(ValueError, match="Invalid option"):
        mku.either_string_or_enum("none", mk.RefinementType)
    with pytest.raises(TypeError, match="Option should be one of"):
        mku.either_string_or_enum(Dummy.A, mk.RefinementType)


@requires_meshkernel
def test_to_geometry_list():
    polygon = sg.Polygon(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    actual = mku.to_geometry_list(polygon)
    assert isinstance(actual, mk.GeometryList)


def test_missing_optional_module():
    abc = MissingOptionalModule("abc")
    with pytest.raises(ImportError, match="abc is required for this functionality"):
        abc.attr
