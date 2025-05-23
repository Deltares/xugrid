import numpy as np
import pytest
import xarray as xr
from pytest_cases import parametrize_with_cases

import xugrid as xu
from xugrid.ugrid import partitioning as pt


def generate_mesh_2d(nx, ny, name="mesh2d"):
    points = [
        (x, y) for y in np.linspace(0, ny, ny + 1) for x in np.linspace(0, nx, nx + 1)
    ]
    connectivity = [
        [
            it + jt * (nx + 1),
            it + jt * (nx + 1) + 1,
            it + (jt + 1) * (nx + 1) + 1,
            it + (jt + 1) * (nx + 1),
        ]
        for jt in range(ny)
        for it in range(nx)
    ]

    return xu.Ugrid2d(*np.array(points).T, -1, np.array(connectivity), name=name)


def generate_mesh_1d(n, name="mesh1d"):
    points = [(p, p) for p in np.linspace(0, n, n + 1)]
    connectivity = [[it, it + 1] for it in range(n)]

    return xu.Ugrid1d(*np.array(points).T, -1, np.array(connectivity), name=name)


def test_labels_to_indices():
    labels = np.array([0, 1, 0, 2, 2])
    indices = pt.labels_to_indices(labels)
    assert np.array_equal(indices[0], [0, 2])
    assert np.array_equal(indices[1], [1])
    assert np.array_equal(indices[2], [3, 4])


def test_single_ugrid_chunk():
    grid = generate_mesh_2d(3, 3)
    ugrid_dims = grid.dims
    da = xr.DataArray(np.ones(grid.n_face), dims=(grid.face_dimension,))
    assert pt.single_ugrid_chunk(da, ugrid_dims) is da

    da = da.chunk({grid.face_dimension: (3, 3, 3)})
    single = pt.single_ugrid_chunk(da, ugrid_dims)
    assert single.chunks == ((9,),)

    # Don't touch other dims
    da_time = (
        xr.DataArray(data=np.ones(3), dims=("time",)).chunk({"time": (1, 1, 1)}) * da
    )
    single = pt.single_ugrid_chunk(da_time, ugrid_dims)
    assert single.chunks == (
        (1, 1, 1),
        (9,),
    )


def case_grid_mesh2d():
    """
    Case simple 2D mesh connectivity with rectangular elements, eg:

            10 -- 11 -- 12 -- 13 -- 14
            |     |     |     |     |
            5 --- 6 --- 7 --- 8 --- 9
            |     |     |     |     |
            0 --- 1 --- 2 --- 3 --- 4
    """
    grid = generate_mesh_2d(5, 3)
    return grid


def case_grid_mesh1d():
    """
    Case simple 1D mesh connectivity:

            0 --- 1 --- 2 --- 3 --- 4 --- 5 --- 6
    """
    grid = generate_mesh_1d(6)
    return grid


@parametrize_with_cases("grid", cases=".", prefix="case_grid_")
def test_label_partitions(grid):
    n_part = 3
    labels = grid.label_partitions(n_part=n_part)
    assert isinstance(labels, xu.UgridDataArray)
    assert labels.name == "labels"
    assert labels.ugrid.grid == grid
    assert np.allclose(np.unique(labels.values), [0, 1, 2])


@parametrize_with_cases("grid", cases=".", prefix="case_grid_")
def test_partition(grid):
    n_part = 3
    grid_type = type(grid)
    grid_size = grid.sizes[grid.core_dimension]
    expected_part_size = grid_size // n_part
    parts = grid.partition(n_part=n_part)
    assert len(parts) == n_part
    for part in parts:
        assert isinstance(part, grid_type)
        part_size = part.sizes[grid.core_dimension]
        assert part_size == expected_part_size


@parametrize_with_cases("grid", cases=".", prefix="case_grid_")
def test_label_partitions_with_weights(grid):
    n_part = 3
    grid_size = grid.sizes[grid.core_dimension]
    half_size = grid_size // 2
    weights = np.ones(grid_size, dtype=int)
    weights[:half_size] = 2
    labels = grid.label_partitions(n_part=n_part, weights=weights)
    assert isinstance(labels, xu.UgridDataArray)
    assert labels.name == "labels"
    assert labels.ugrid.grid == grid
    uniques, counts = np.unique(labels.values, return_counts=True)
    np.testing.assert_array_equal(uniques, [0, 1, 2])
    # Test if the partition sizes are different
    assert np.max(counts) != np.min(counts)


@parametrize_with_cases("grid", cases=".", prefix="case_grid_")
def test_label_partitions_with_weights__error(grid):
    n_part = 3
    grid_size = grid.sizes[grid.core_dimension]
    weights = np.ones(grid_size + 10, dtype=int)
    with pytest.raises(ValueError, match="Wrong shape on weights."):
        grid.label_partitions(n_part=n_part, weights=weights)

    weights = np.ones(grid_size, dtype=float)
    with pytest.raises(TypeError, match="Wrong type on weights."):
        grid.label_partitions(n_part=n_part, weights=weights)

    weights = np.ones(grid_size, dtype=int) * -1
    with pytest.raises(ValueError, match="Wrong values on weights."):
        grid.label_partitions(n_part=n_part, weights=weights)


@parametrize_with_cases("grid", cases=".", prefix="case_grid_")
def test_partition_with_weights(grid):
    n_part = 3
    grid_type = type(grid)
    grid_size = grid.sizes[grid.core_dimension]
    half_size = grid_size // 2
    weights = np.ones(grid_size, dtype=int)
    weights[:half_size] = 2
    parts = grid.partition(n_part=n_part, weights=weights)
    assert len(parts) == n_part
    part_sizes = []
    for part in parts:
        assert isinstance(part, grid_type)
        part_sizes.append(part.sizes[grid.core_dimension])
    assert np.max(part_sizes) != np.min(part_sizes)


@parametrize_with_cases("grid", cases=".", prefix="case_grid_")
def test_label_partitions_dataarray_with_weights(grid):
    n_part = 3
    core_dim = grid.core_dimension
    grid_size = grid.sizes[core_dim]
    half_size = grid_size // 2
    weights = np.ones(grid_size, dtype=int)
    weights[:half_size] = 2
    weights_da = xr.DataArray(weights, dims=(core_dim,))
    weights_uda = xu.UgridDataArray(weights_da, grid=grid)
    labels = weights_uda.ugrid.label_partitions(n_part=n_part)
    assert isinstance(labels, xu.UgridDataArray)
    assert labels.name == "labels"
    assert labels.ugrid.grid == grid
    uniques, counts = np.unique(labels.values, return_counts=True)
    np.testing.assert_array_equal(uniques, [0, 1, 2])
    # Test if the partition sizes are different
    assert np.max(counts) != np.min(counts)


class TestDatasetPartition:
    @pytest.fixture(autouse=True)
    def setup(self):
        grid = generate_mesh_2d(5, 3)
        ds = xr.Dataset()
        ds["node_z"] = xr.DataArray(np.arange(grid.n_node), dims=(grid.node_dimension,))
        ds["edge_z"] = xr.DataArray(np.arange(grid.n_edge), dims=(grid.edge_dimension,))
        ds["face_z"] = xr.DataArray(np.arange(grid.n_face), dims=(grid.face_dimension,))
        uds = xu.UgridDataset(obj=ds, grids=[grid])
        self.uds = uds
        self.grid = grid
        self.obj = self.uds.ugrid.obj
        self.labels = xu.UgridDataArray(
            xr.DataArray(
                data=np.repeat([0, 1, 2], 5),
                dims=(self.grid.face_dimension,),
            ),
            self.grid,
        )

    def test_partition_by_labels__errors(self):
        with pytest.raises(TypeError, match="labels must be a UgridDataArray"):
            pt.partition_by_label(self.grid, self.obj, np.arange(15))

        with pytest.raises(TypeError, match="labels must have integer dtype"):
            pt.partition_by_label(self.grid, self.obj, self.labels.astype(float))

        other_grid = generate_mesh_2d(3, 3)
        with pytest.raises(
            ValueError, match="grid of labels does not match xugrid object"
        ):
            pt.partition_by_label(other_grid, self.obj, self.labels)

        dim_labels = self.labels.expand_dims("somedim", axis=0)
        with pytest.raises(ValueError, match="Can only partition this topology"):
            pt.partition_by_label(self.grid, self.obj, dim_labels)

        with pytest.raises(TypeError, match="Expected DataArray or Dataset"):
            pt.partition_by_label(self.grid, self.obj.values, self.labels)

    def test_partition_by_labels__dataset(self):
        partitions = pt.partition_by_label(self.grid, self.obj, self.labels)
        assert len(partitions) == 3
        for partition in partitions:
            assert isinstance(partition, xu.UgridDataset)
            assert "face_z" in partition
            assert "edge_z" in partition
            assert "node_z" in partition

    def test_partition_by_labels__dataarray(self):
        partitions = pt.partition_by_label(self.grid, self.obj["face_z"], self.labels)
        assert len(partitions) == 3
        for partition in partitions:
            assert isinstance(partition, xu.UgridDataArray)
            assert partition.name == "face_z"

    def test_partition_roundtrip(self):
        partitions = self.uds.ugrid.partition(n_part=4)
        back = pt.merge_partitions(partitions)
        assert isinstance(back, xu.UgridDataset)

        order = np.argsort(back["face_z"].values)
        reordered = back.isel(mesh2d_nFaces=order)
        assert reordered["face_z"].equals(self.uds["face_z"])

    def test_merge_partition_single(self):
        partitions = [self.uds]
        back = pt.merge_partitions(partitions)
        assert back == self.uds

    def test_merge_partitions__errors(self):
        partitions = self.uds.ugrid.partition(n_part=2)
        with pytest.raises(TypeError, match="Expected UgridDataArray or UgridDataset"):
            pt.merge_partitions([p.ugrid.obj for p in partitions])

        grid1 = partitions[1].ugrid.grid
        partitions[1]["extra"] = (grid1.face_dimension, np.ones(grid1.n_face))
        with pytest.raises(
            ValueError,
            match="Missing variables: {'extra'} in partition",
        ):
            pt.merge_partitions(partitions)

        partitions = self.uds.ugrid.partition(n_part=2)
        partitions[1]["face_z"] = partitions[1]["face_z"].expand_dims("layer", axis=0)
        with pytest.raises(ValueError, match="Dimensions for 'face_z' do not match"):
            pt.merge_partitions(partitions)

        uds = self.uds.copy()
        grid = uds.ugrid.grid
        uds["two_dim"] = (
            ("mesh2d_nNodes", "mesh2d_nEdges"),
            np.ones((grid.n_node, grid.n_edge)),
        )
        partitions = uds.ugrid.partition(n_part=2)
        with pytest.raises(
            ValueError, match="two_dim contains more than one UGRID dimension"
        ):
            pt.merge_partitions(partitions)

        with pytest.raises(
            ValueError, match="Cannot merge partitions: zero partitions provided."
        ):
            xu.merge_partitions([])

    def test_merge_partitions_no_duplicates(self):
        part1 = self.uds.isel(mesh2d_nFaces=[0, 1, 2, 3])
        part2 = self.uds.isel(mesh2d_nFaces=[2, 3, 4, 5])
        merged = pt.merge_partitions([part1, part2])
        assert np.bincount(merged["face_z"] == 1).all()

    def test_merge_inconsistent_chunks_across_partitions(self):
        part1, part2 = self.uds.ugrid.partition(n_part=2)
        time = xr.DataArray(data=np.ones(3), dims=("time",))
        part1 = (part1 * time).chunk({"time": (1, 1, 1)})
        part2 = (part2 * time).chunk({"time": (1, 2)})
        merged = pt.merge_partitions([part1, part2])
        assert isinstance(merged, xu.UgridDataset)
        assert merged.chunks["time"] == (1, 1, 1)

    def test_merge_inconsistent_chunks_across_variables(self):
        uds = self.uds * xr.DataArray(data=np.ones(3), dims=("time",))
        # Make them inconsistent across the variables
        uds["node_z"] = uds["node_z"].chunk({"time": (3,)})
        uds["edge_z"] = uds["edge_z"].chunk({"time": (2, 1)})
        uds["face_z"] = uds["face_z"].chunk({"time": (1, 2)})
        part1, part2 = uds.ugrid.partition(n_part=2)
        merged = pt.merge_partitions([part1, part2])
        # Test that it runs without encountering the xarray "inconsistent
        # chunks" ValueError.
        assert isinstance(merged, xu.UgridDataset)
        # Make sure they remain inconsistent after merging.
        assert uds["node_z"].chunks == ((self.grid.n_node,), (3,))
        assert uds["edge_z"].chunks == ((self.grid.n_edge,), (2, 1))
        assert uds["face_z"].chunks == ((self.grid.n_face,), (1, 2))


class TestMultiTopology2DMergePartitions:
    @pytest.fixture(autouse=True)
    def setup(self):
        grid_a = generate_mesh_2d(2, 3, "first")
        grid_b = generate_mesh_2d(4, 5, "second")
        parts_a = grid_a.partition(n_part=2)
        parts_b = grid_b.partition(n_part=2)

        datasets = []
        for i, (part_a, part_b) in enumerate(zip(parts_a, parts_b)):
            ds = xu.UgridDataset(grids=[part_a, part_b])
            ds["a"] = ((part_a.face_dimension), np.arange(part_a.n_face))
            ds["b"] = ((part_b.face_dimension), np.arange(part_b.n_face))
            ds["c"] = i
            datasets.append(ds)

        self.datasets = datasets

    def test_merge_partitions(self):
        merged = pt.merge_partitions(self.datasets)
        assert isinstance(merged, xu.UgridDataset)
        assert len(merged.ugrid.grids) == 2
        # In case of non-UGRID data, it should default to the last partition:
        assert merged["c"] == 1

        assert len(merged["first_nFaces"]) == 6
        assert len(merged["second_nFaces"]) == 20

    def test_merge_partitions__unique_grid_per_partition(self):
        pa = self.datasets[0][["a"]]
        pb = self.datasets[1][["b"]]
        merged = pt.merge_partitions([pa, pb])

        assert isinstance(merged, xu.UgridDataset)
        assert len(merged.ugrid.grids) == 2

        assert len(merged["first_nFaces"]) == 3
        assert len(merged["second_nFaces"]) == 10

    def test_merge_partitions__errors(self):
        pa = self.datasets[0][["a"]] * xr.DataArray([1.0, 1.0], dims=("error_dim",))
        pb = self.datasets[1][["a"]]
        with pytest.raises(
            ValueError, match="Dimensions for 'a' do not match across partitions: "
        ):
            pt.merge_partitions([pa, pb])

        grid_a = self.datasets[1].ugrid.grids[0].copy()
        grid_c = self.datasets[1].ugrid.grids[1].copy()
        grid_c._attrs["face_dimension"] = "abcdef"
        dataset2 = xu.UgridDataset(
            obj=self.datasets[1].ugrid.obj, grids=[grid_a, grid_c]
        )
        with pytest.raises(ValueError, match="Dimension names on UGRID topology"):
            pt.merge_partitions([self.datasets[0], dataset2])

        xy = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
            ]
        )
        grid_d = xu.Ugrid1d(
            node_x=xy[:, 0],
            node_y=xy[:, 1],
            fill_value=-1,
            edge_node_connectivity=np.array([[0, 1], [1, 2]]),
            name="second",
        )
        dataset3 = xu.UgridDataset(
            obj=self.datasets[1].ugrid.obj, grids=[grid_a, grid_d]
        )
        with pytest.raises(
            TypeError, match="All partition topologies with name second"
        ):
            pt.merge_partitions([self.datasets[0], dataset3])


class TestMergeDataset1D:
    @pytest.fixture(autouse=True)
    def setup(self):
        grid = generate_mesh_1d(6, "mesh1d")
        parts = grid.partition(n_part=2)

        values_parts = [np.arange(part.n_edge) for part in parts]

        datasets_partitioned = []
        for i, (part, values) in enumerate(zip(parts, values_parts)):
            ds = xu.UgridDataset(grids=[part])
            ds["a"] = ((part.edge_dimension), values)
            ds["c"] = i
            datasets_partitioned.append(ds)

        ds_expected = xu.UgridDataset(grids=[grid])
        ds_expected["a"] = ((grid.edge_dimension), np.concatenate(values_parts))
        ds_expected["c"] = 1
        # Assign coordinates also added during merge_partitions
        coords = {grid.edge_dimension: np.arange(grid.n_edge)}
        ds_expected = ds_expected.assign_coords(**coords)

        self.datasets_partitioned = datasets_partitioned
        self.dataset_expected = ds_expected

    def test_merge_partitions(self):
        merged = pt.merge_partitions(self.datasets_partitioned)
        assert isinstance(merged, xu.UgridDataset)
        assert len(merged.ugrid.grids) == 1
        # In case of non-UGRID data, it should default to the last partition of
        # the grid that's checked last.
        assert merged["c"] == 1
        # Ensure indexes are consistent with the expected dataset
        merged = merged.ugrid.reindex_like(self.dataset_expected.ugrid.grid)
        assert self.dataset_expected.ugrid.grid.equals(merged.ugrid.grid)
        assert self.dataset_expected["a"].equals(merged["a"])
        assert self.dataset_expected.equals(merged)


class TestMultiTopology1D2DMergePartitions:
    @pytest.fixture(autouse=True)
    def setup(self):
        grid_a = generate_mesh_2d(2, 3, "mesh2d")
        grid_b = generate_mesh_1d(6, "mesh1d")
        parts_a = grid_a.partition(n_part=2)
        parts_b = grid_b.partition(n_part=2)

        values_parts_a = [np.arange(part.n_face) for part in parts_a]
        values_parts_b = [np.arange(part.n_edge) for part in parts_b]

        datasets_parts = []
        for i, (part_a, part_b, values_a, values_b) in enumerate(
            zip(parts_a, parts_b, values_parts_a, values_parts_b)
        ):
            ds = xu.UgridDataset(grids=[part_a, part_b])
            ds["a"] = ((part_a.face_dimension), values_a)
            ds["b"] = ((part_b.edge_dimension), values_b)
            ds["c"] = i

            coords = {
                part_a.face_dimension: values_a,
                part_b.edge_dimension: values_b,
            }

            datasets_parts.append(ds.assign_coords(**coords))

        ds_expected = xu.UgridDataset(grids=[grid_a, grid_b])
        ds_expected["a"] = ((grid_a.face_dimension), np.concatenate(values_parts_a))
        ds_expected["b"] = ((grid_b.edge_dimension), np.concatenate(values_parts_b))
        ds_expected["c"] = 1
        # Assign coordinates also added during merge_partitions
        coords = {
            grid_a.face_dimension: np.concatenate(values_parts_a),
            grid_b.edge_dimension: np.concatenate(values_parts_b),
        }
        ds_expected = ds_expected.assign_coords(**coords)

        self.datasets_parts = datasets_parts
        self.dataset_expected = ds_expected

    def test_merge_partitions(self):
        merged = pt.merge_partitions(self.datasets_parts)
        assert isinstance(merged, xu.UgridDataset)
        assert len(merged.ugrid.grids) == 2
        # In case of non-UGRID data, it should default to the last partition of
        # the grid that's checked last.
        assert merged["c"] == 1

        assert self.dataset_expected.equals(merged)

    def test_merge_partitions__inconsistent_grid_types(self):
        self.datasets_parts[0] = self.datasets_parts[0].drop_vars(
            ["b", "mesh1d_nEdges"]
        )
        b = self.dataset_expected["b"].isel(mesh1d_nEdges=[0, 1, 2])
        self.dataset_expected = self.dataset_expected.drop_vars(["b", "mesh1d_nEdges"])
        self.dataset_expected["b"] = b
        self.dataset_expected["c"] = 1

        merged = pt.merge_partitions(self.datasets_parts)
        assert isinstance(merged, xu.UgridDataset)
        assert len(merged.ugrid.grids) == 2
        # In case of non-UGRID data, it should default to the last partition of
        # the grid that's checked last.
        assert merged["c"] == 1

        assert self.dataset_expected.equals(merged)

    def test_merge_partitions_merge_chunks(self):
        # Dataset has no chunks defined, chunks should not appear.
        merged = pt.merge_partitions(self.datasets_parts)
        assert len(merged.chunks) == 0

        # Dataset has chunks, keyword is True, chunks should be size 1.
        datasets_parts = [
            part.expand_dims({"time": 3}).chunk({"time": 1})
            for part in self.datasets_parts
        ]
        merged = pt.merge_partitions(datasets_parts)
        assert len(merged.chunks["mesh2d_nFaces"]) == 1
        assert len(merged.chunks["mesh1d_nEdges"]) == 1
        assert len(merged.chunks["time"]) == 3

        # Dataset has chunks, keyword is False, chunks should be size npartition.
        merged = pt.merge_partitions(datasets_parts, merge_ugrid_chunks=False)
        assert len(merged.chunks["mesh2d_nFaces"]) == 2
        assert len(merged.chunks["mesh1d_nEdges"]) == 2
        assert len(merged.chunks["time"]) == 3
