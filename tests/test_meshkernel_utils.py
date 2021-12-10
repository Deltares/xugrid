import meshkernel as mk
import pytest
import shapely.geometry as sg

from xugrid import meshkernel_utils as mku


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
        mku.either_string_or_enum(mk.AveragingMethod.MAX, mk.RefinementType)


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
