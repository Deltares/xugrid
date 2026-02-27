import abc

import numpy as np
import pytest
import xarray as xr
from scipy.spatial import KDTree

import xugrid
from xugrid.regrid.grid.structured import StructuredGrid2d
from xugrid.regrid.grid.unstructured import UnstructuredGrid2d


class TestUnstructuredGrid2dBase(abc.ABC):
    """Base class for common test functionality."""

    def assert_identity_mapping(
        self, source, target, weights, expected_size, expected_weights=None
    ):
        valid = weights > 1.0e-5
        source_valid = source[valid]
        target_valid = target[valid]
        weights_valid = weights[valid]
        sorter = np.argsort(source_valid)

        assert np.array_equal(source_valid[sorter], np.arange(expected_size))
        assert np.array_equal(target_valid[sorter], np.arange(expected_size))

        if expected_weights is not None:
            assert np.allclose(weights_valid[sorter], expected_weights)
        else:
            assert np.allclose(weights_valid[sorter], np.ones(expected_size))


class TestUnstructuredGrid2dFace(TestUnstructuredGrid2dBase):
    """Tests for face-based unstructured grids."""

    @pytest.fixture
    def face_grid(self):
        grid = xugrid.data.disk().grid
        return UnstructuredGrid2d(grid, grid.face_dimension)

    def test_basic_properties(self, face_grid):
        assert face_grid.ndim == 1
        assert face_grid.facet == "face"  # Fixed the assignment bug!
        assert isinstance(face_grid.coordinates, np.ndarray)
        assert face_grid.dims == ("mesh2d_nFaces",)
        assert face_grid.shape == (384,)
        assert face_grid.size == 384
        assert isinstance(face_grid.area, np.ndarray)
        assert face_grid.area.size == 384
        assert isinstance(face_grid.kdtree, KDTree)

    def test_convert_to(self, face_grid):
        assert face_grid is face_grid.convert_to(UnstructuredGrid2d)

        with pytest.raises(TypeError, match="Cannot convert"):
            face_grid.convert_to(StructuredGrid2d)

    @pytest.mark.parametrize("relative", [True, False])
    def test_overlap_with_self(self, face_grid, relative):
        source, target, weights = face_grid.overlap(other=face_grid, relative=relative)

        expected_weights = np.ones(face_grid.size) if relative else face_grid.area
        self.assert_identity_mapping(
            source, target, weights, face_grid.size, expected_weights
        )

    def test_locate_points(self, face_grid):
        index = face_grid.locate_points(face_grid.coordinates, tolerance=None)
        assert np.array_equal(index, np.arange(face_grid.size))

    def test_locate_inside(self, face_grid):
        source, target, weights = face_grid.locate_inside(face_grid)
        self.assert_identity_mapping(source, target, weights, face_grid.size)

    def test_locate_nearest(self, face_grid):
        source, target, weights = face_grid.locate_nearest(face_grid)
        self.assert_identity_mapping(source, target, weights, face_grid.size)

    def test_inverse_distance_exact_match(self, face_grid):
        source, target, weights = face_grid.inverse_distance(
            face_grid,
            tolerance=None,
            max_distance=np.inf,
            min_points=3,
            max_points=3,
            power=2,
            smoothing=0.0,
        )
        self.assert_identity_mapping(source, target, weights, face_grid.size)

    def test_inverse_distance_with_smoothing(self, face_grid):
        source, target, weights = face_grid.inverse_distance(
            face_grid,
            tolerance=None,
            max_distance=np.inf,
            min_points=3,
            max_points=3,
            power=2,
            smoothing=0.5,
        )
        # With smoothing, should return three entries per point
        assert source.size == 3 * face_grid.size

    def test_barycentric(self, face_grid):
        source, target, weights = face_grid.barycentric(face_grid)
        self.assert_identity_mapping(source, target, weights, face_grid.size)

    def test_to_dataset(self, face_grid):
        ds = face_grid.to_dataset(name="regrid_circle")
        assert isinstance(ds, xr.Dataset)


class TestUnstructuredGrid2dNode(TestUnstructuredGrid2dBase):
    """Tests for node-based unstructured grids."""

    @pytest.fixture
    def node_grid(self):
        grid = xugrid.data.disk().grid
        return UnstructuredGrid2d(grid, grid.node_dimension)

    def test_basic_properties(self, node_grid):
        assert node_grid.ndim == 1
        assert node_grid.facet == "node"
        assert isinstance(node_grid.coordinates, np.ndarray)
        assert node_grid.dims == ("mesh2d_nNodes",)
        assert node_grid.shape == (217,)
        assert node_grid.size == 217
        assert isinstance(node_grid.kdtree, KDTree)

    def test_convert_to(self, node_grid):
        assert node_grid is node_grid.convert_to(UnstructuredGrid2d)

        with pytest.raises(TypeError, match="Cannot convert"):
            node_grid.convert_to(StructuredGrid2d)

    def test_overlap_with_self(self, node_grid):
        with pytest.raises(ValueError):
            node_grid.overlap(node_grid, relative=False)

    def test_locate_points(self, node_grid):
        # All edge cases...
        index = node_grid.locate_points(node_grid.coordinates, tolerance=None)
        # Should all be in bounds
        assert (index != -1).all()

    def test_locate_inside(self, node_grid):
        with pytest.raises(ValueError):
            node_grid.locate_inside(node_grid, tolerance=None)

    def test_locate_nearest(self, node_grid):
        source, target, weights = node_grid.locate_nearest(node_grid)
        self.assert_identity_mapping(source, target, weights, node_grid.size)

    def test_inverse_distance_exact_match(self, node_grid):
        source, target, weights = node_grid.inverse_distance(
            node_grid,
            tolerance=None,
            max_distance=np.inf,
            min_points=3,
            max_points=3,
            power=2,
            smoothing=0.0,
        )
        self.assert_identity_mapping(source, target, weights, node_grid.size)

    def test_inverse_distance_with_smoothing(self, node_grid):
        source, target, weights = node_grid.inverse_distance(
            node_grid,
            tolerance=None,
            max_distance=np.inf,
            min_points=3,
            max_points=3,
            power=2,
            smoothing=0.5,
        )
        # With smoothing, should return three entries per point
        assert source.size == 3 * node_grid.size

    def test_barycentric(self, node_grid):
        source, target, weights = node_grid.barycentric(node_grid)
        self.assert_identity_mapping(source, target, weights, node_grid.size)

    def test_to_dataset(self, node_grid):
        ds = node_grid.to_dataset(name="regrid_circle")
        assert isinstance(ds, xr.Dataset)
