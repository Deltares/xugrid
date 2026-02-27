import numpy as np
import pytest
import xarray as xr

from xugrid.regrid.grid.structured import StructuredGrid1d, StructuredGrid2d

# Testgrids
# --------
# grid a(x):               |______50_____|_____100_____|_____150_____|               -> source
# grid b(x):        |______25_____|______75_____|_____125_____|_____175_____|        -> target
# --------
# grid c(x):            |______40_____|______90_____|_____140_____|____190_____|     -> target
# --------
# grid d(x):              |__30__|__55__|__80_|__105__|                              -> target
# --------
# grid e(x):              |__30__|____67.5____|__105__|                              -> target
# --------


class TestStructuredGridBase:
    """Base class for structured grid test utilities."""

    @staticmethod
    def assert_expected_overlap(
        actual_source: np.ndarray,
        actual_target: np.ndarray,
        actual_weights: np.ndarray,
        expected_source: np.ndarray,
        expected_target: np.ndarray,
        expected_weights: np.ndarray,
    ):
        """
        Robust comparison method that works for numpy <2.0 and >=2.0.
        Handles non-stable sorting behavior changes in numpy 2.0.
        """
        actual_mapping = np.column_stack((actual_target, actual_source))
        expected_mapping = np.column_stack((expected_target, expected_source))
        actual, actual_sorter = np.unique(actual_mapping, axis=0, return_index=True)
        expected, expected_sorter = np.unique(
            expected_mapping, axis=0, return_index=True
        )
        assert np.array_equal(actual, expected)
        assert np.allclose(
            actual_weights[actual_sorter], expected_weights[expected_sorter]
        )

    @staticmethod
    def create_overlap_case(source_indices: list, target_indices: list, weights: list):
        return {
            "source": np.array(source_indices),
            "target": np.array(target_indices),
            "weights": np.array(weights),
        }


class TestStructuredGrid1d(TestStructuredGridBase):
    """Tests for 1D structured grids."""

    def test_init_valid(self, grid_data_a_1d):
        assert isinstance(grid_data_a_1d, StructuredGrid1d)

    def test_init_invalid(self):
        with pytest.raises(TypeError):
            StructuredGrid1d(1)

    def test_nonscalar_dx(self):
        """Test handling of non-scalar dx coordinate."""
        da = xr.DataArray(
            [1, 2, 3], coords={"x": [1, 2, 3], "dx": ("x", [1, 1, 1])}, dims=("x",)
        )
        grid = StructuredGrid1d(da, name="x")
        actual = xr.DataArray([1, 2, 3], coords=grid.coords, dims=grid.dims)
        assert actual.identical(da)

    def test_basic_properties(self, grid_data_a_1d):
        grid = grid_data_a_1d
        assert isinstance(grid.coords, dict)
        assert grid.facet == "face"
        assert isinstance(grid.coordinates, np.ndarray)
        assert grid.ndim == 1
        assert grid.dims == ("x",)
        assert np.allclose(grid.length, 50.0)

    def test_locate_points(self, grid_data_a_1d):
        x = np.array([0.0, 25.0, 40.0, 90.0, 175.0, 200.0])
        actual = grid_data_a_1d.locate_points(x)
        expected = [-1, 0, 0, 1, -1, -1]
        assert np.array_equal(actual, expected)

    @pytest.mark.parametrize(
        "case_name,target_grid,expected",
        [
            (
                "equidistant",
                "grid_data_b_1d",
                {
                    "source": [0, 0, 1, 1, 2, 2],
                    "target": [0, 1, 1, 2, 2, 3],
                    "weights": [25, 25, 25, 25, 25, 25],
                },
            ),
            (
                "non_equidistant",
                "grid_data_e_1d",
                {
                    "source": [0, 0, 1, 1],
                    "target": [0, 1, 1, 2],
                    "weights": [17.5, 32.5, 17.5, 25.0],
                },
            ),
        ],
    )
    def test_overlap_1d_absolute(
        self, grid_data_a_1d, case_name, target_grid, expected, request
    ):
        """Test 1D overlap with absolute weights."""
        target = request.getfixturevalue(target_grid)
        self.assert_expected_overlap(
            *grid_data_a_1d.overlap(target, relative=False),
            np.array(expected["source"]),
            np.array(expected["target"]),
            np.array(expected["weights"]),
        )

    def test_overlap_1d_relative(self, grid_data_a_1d, grid_data_e_1d):
        """Test 1D overlap with relative weights."""
        expected_weights = np.array(
            [17.5 / 50.0, 32.5 / 50.0, 17.5 / 50.0, 25.0 / 50.0]
        )
        self.assert_expected_overlap(
            *grid_data_a_1d.overlap(grid_data_e_1d, relative=True),
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 2]),
            expected_weights,
        )

    @pytest.mark.parametrize(
        "case_name,target_grid,expected",
        [
            (
                "equidistant",
                "grid_data_b_1d",
                [0, 0, 1, -1],
            ),
            (
                "non_equidistant",
                "grid_data_e_1d",
                [0, 0, 1],
            ),
        ],
    )
    def test_locate_points_other(
        self, grid_data_a_1d, case_name, target_grid, expected, request
    ):
        target = request.getfixturevalue(target_grid)
        index = grid_data_a_1d.locate_points(target.coordinates)
        assert np.array_equal(index, expected)

    @pytest.mark.parametrize(
        "case_name,target_grid,expected",
        [
            (
                "equidistant",
                "grid_data_b_1d",
                {
                    "source": [0, 0, 0, 1, 1, 2, -1, -1],
                    "target": [0, 0, 1, 1, 2, 2, 3, 3],
                    "weights": [0.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0],
                },
            ),
            (
                "equidistant_c",
                "grid_data_c_1d",
                {
                    "source": [0, 0, 1, 0, 2, 1, -1, -1],
                    "target": [0, 0, 1, 1, 2, 2, 3, 3],
                    "weights": [0.0, 1.0, 0.8, 0.2, 0.8, 0.2, 0.0, 0.0],
                },
            ),
            (
                "equidistant_d",
                "grid_data_d_1d",
                {
                    "source": [0, 0, 0, 1, 1, 0, 1, 2],
                    "target": [0, 0, 1, 1, 2, 2, 3, 3],
                    "weights": [0.0, 0.1, 0.9, 0.1, 0.6, 0.4, 0.9, 0.1],
                },
            ),
            (
                "non_equidistant",
                "grid_data_e_1d",
                {
                    "source": [0, 0, 0, 1, 1, 2],
                    "target": [0, 0, 1, 1, 2, 2],
                    "weights": [0.0, 1.0, 0.65, 0.35, 0.9, 0.1],
                },
            ),
        ],
    )
    def test_linear_weights(
        self, grid_data_a_1d, case_name, target_grid, expected, request
    ):
        target = request.getfixturevalue(target_grid)
        self.assert_expected_overlap(
            *grid_data_a_1d.barycentric(target.coordinates),
            np.array(expected["source"]),
            np.array(expected["target"]),
            np.array(expected["weights"]),
        )

    def test_linear_weights_identity(self, grid_data_b_1d):
        """Test linear weights for identity mapping (grid to itself)."""
        source, target, weights = grid_data_b_1d.linear_weights(
            grid_data_b_1d.coordinates
        )

        # Should have identity mapping with weights of 1.0 and 0.0
        expected_target = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        expected_weights_unique = [0, 1]

        assert np.array_equal(target, expected_target)
        assert np.array_equal(np.unique(weights), expected_weights_unique)

        # Check that non-zero weights correspond to identity mapping
        identity_indices = weights != 0
        identity_sources = source[identity_indices]
        identity_targets = target[identity_indices]
        assert np.array_equal(identity_sources, identity_targets)

    @pytest.mark.parametrize(
        "case_name,target_grid,expected",
        [
            (
                "equidistant",
                "grid_data_b_1d",
                {
                    "source": [0, 0, 1, -1],  # Left-inclusive
                    "target": [0, 1, 2, 3],
                    "weights": [25, 25, 25, 25],
                },
            ),
            (
                "non_equidistant",
                "grid_data_e_1d",
                {"source": [0, 0, 1], "target": [0, 1, 2], "weights": [20, 17.5, 5]},
            ),
        ],
    )
    def test_locate_nearest(
        self, grid_data_a_1d, case_name, target_grid, expected, request
    ):
        target = request.getfixturevalue(target_grid)
        self.assert_expected_overlap(
            *grid_data_a_1d.locate_nearest(target.coordinates),
            np.array(expected["source"]),
            np.array(expected["target"]),
            np.array(expected["weights"]),
        )


class TestStructuredGrid2d(TestStructuredGridBase):
    """Tests for 2D structured grids."""

    def test_init_valid(self, grid_data_a_2d):
        """Test valid initialization."""
        assert isinstance(grid_data_a_2d, StructuredGrid2d)

    def test_init_invalid(self):
        """Test invalid initialization."""
        with pytest.raises(TypeError):
            StructuredGrid2d(1)

    def test_overlap_2d(self, grid_data_a_2d, grid_data_b_2d):
        """
        Test 2D overlap with exact expected values.

        Tests the geometric overlap calculation between two 2D grids:
        - 3x3 source grid overlapping with 4x4 target grid
        - Each source cell overlaps with exactly 4 target cells
        - All overlaps have equal area (625 m²)
        """
        expected_source = np.array(
            [
                0,
                0,
                0,
                0,  # Source 0 -> targets 0,1,4,5
                1,
                1,
                1,
                1,  # Source 1 -> targets 1,2,5,6
                2,
                2,
                2,
                2,  # Source 2 -> targets 2,3,6,7
                3,
                3,
                3,
                3,  # Source 3 -> targets 4,5,8,9
                4,
                4,
                4,
                4,  # Source 4 -> targets 5,6,9,10
                5,
                5,
                5,
                5,  # Source 5 -> targets 6,7,10,11
                6,
                6,
                6,
                6,  # Source 6 -> targets 8,9,12,13
                7,
                7,
                7,
                7,  # Source 7 -> targets 9,10,13,14
                8,
                8,
                8,
                8,  # Source 8 -> targets 10,11,14,15
            ]
        )
        expected_target = np.array(
            [
                0,
                4,
                5,
                1,  # Source 0 overlaps
                2,
                6,
                5,
                1,  # Source 1 overlaps
                2,
                3,
                7,
                6,  # Source 2 overlaps
                8,
                9,
                5,
                4,  # Source 3 overlaps
                9,
                5,
                10,
                6,  # Source 4 overlaps
                10,
                11,
                7,
                6,  # Source 5 overlaps
                9,
                8,
                12,
                13,  # Source 6 overlaps
                10,
                14,
                13,
                9,  # Source 7 overlaps
                10,
                11,
                14,
                15,  # Source 8 overlaps
            ]
        )
        expected_weights = np.full(36, 625.0)  # All equal area overlaps

        self.assert_expected_overlap(
            *grid_data_a_2d.overlap(grid_data_b_2d, relative=False),
            expected_source,
            expected_target,
            expected_weights,
        )

    def test_locate_inside(self, grid_data_a_2d, grid_data_b_2d):
        self.assert_expected_overlap(
            *grid_data_a_2d.locate_inside(grid_data_b_2d, None),
            np.array([0, 0, 1, 0, 0, 1, 3, 3, 4]),  # Interior source points
            np.array([0, 1, 2, 4, 5, 6, 8, 9, 10]),  # Corresponding target cells
            np.ones(9),  # All weights = 1
        )

    def test_barycentric_uniform_spacing(self, grid_data_a_2d):
        """
        Test 2D linear weights with uniform grid spacing.

        With uniform spacing, each target point gets equal contribution (25%)
        from its 4 surrounding source points.
        """

        b = xr.DataArray(
            data=np.ones((2, 2)),
            coords={"y": [75.0, 125.0], "x": [75.0, 125.0]},
            dims=("y", "x"),
        )
        expected_source = np.array([0, 1, 3, 4, 1, 2, 4, 5, 3, 4, 6, 7, 4, 5, 7, 8])
        expected_target = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        expected_weights = np.full(16, 0.25)  # Equal contribution from all corners
        grid_b = StructuredGrid2d(b, "x", "y")

        self.assert_expected_overlap(
            *grid_data_a_2d.barycentric(grid_b, None),
            expected_source,
            expected_target,
            expected_weights,
        )

    def test_barycentric_identity(self, grid_data_b_2d):
        """
        Test 2D linear weights for identity mapping (grid to itself).

        When interpolating a grid to itself, should get perfect identity mapping
        with weights of 1.0 for exact matches and 0.0 for others.
        """
        source, target, weights = grid_data_b_2d.barycentric(grid_data_b_2d, None)

        # Should have 4 entries per target (16 targets * 4 = 64 total)
        expected_target = np.repeat(np.arange(16), 4)
        assert np.array_equal(target, expected_target)

        # Should only have weights of 0 and 1 (exact matches or no contribution)
        assert np.array_equal(np.unique(weights), [0, 1])

        # Non-zero weights should correspond to perfect identity mapping
        identity_mask = weights != 0
        identity_sources = source[identity_mask]
        identity_targets = target[identity_mask]
        assert np.array_equal(identity_sources, identity_targets)
