import numpy as np
import pytest
import xarray as xr

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


class TestGridPartitioning:
    @pytest.fixture(autouse=True)
    def setup(self):
        """
        Generate simple 2D mesh connectivity with rectangular elements, eg:

              10 -- 11 -- 12 -- 13 -- 14
              |     |     |     |     |
              5 --- 6 --- 7 --- 8 --- 9
              |     |     |     |     |
              0 --- 1 --- 2 --- 3 --- 4
        """
        self.grid = generate_mesh_2d(5, 3)

    def test_label_partition(self):
        labels = self.grid.label_partitions(n_part=3)
        assert isinstance(labels, xu.UgridDataArray)
        assert labels.name == "labels"
        assert labels.ugrid.grid == self.grid
        assert np.allclose(np.unique(labels.values), [0, 1, 2])

    def test_partition(self):
        parts = self.grid.partition(n_part=3)
        assert len(parts) == 3
        for part in parts:
            assert isinstance(part, xu.Ugrid2d)


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
        back = xu.merge_partitions(partitions)
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
        # TODO: If partitioning implemented for 1D grids, replace with that.
        i_edges = [[0, 1, 2], [3, 4, 5]]
        parts = [grid.isel(mesh1d_nEdges=np.array(ls)) for ls in i_edges]

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

        assert self.dataset_expected.ugrid.grid.equals(merged.ugrid.grid)
        assert self.dataset_expected["a"].equals(merged["a"])
        assert self.dataset_expected.equals(merged)


class TestMultiTopology1D2DMergePartitions:
    @pytest.fixture(autouse=True)
    def setup(self):
        grid_a = generate_mesh_2d(2, 3, "mesh2d")
        grid_b = generate_mesh_1d(6, "mesh1d")
        parts_a = grid_a.partition(n_part=2)
        # TODO: If partitioning implemented for 1D grids, replace with that.
        i_edges = [[0, 1, 2], [3, 4, 5]]
        parts_b = [grid_b.isel(mesh1d_nEdges=np.array(ls)) for ls in i_edges]

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
