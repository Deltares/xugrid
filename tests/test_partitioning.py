import numpy as np
import pytest
import xarray as xr

import xugrid as xu
from xugrid.ugrid import partitioning as pt


def generate_mesh_2d(nx, ny):
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

    return xu.Ugrid2d(*np.array(points).T, -1, np.array(connectivity))


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
        uds = xu.UgridDataset(grids=[grid])
        uds["node_z"] = xr.DataArray(
            np.arange(grid.n_face), dims=(grid.node_dimension,)
        )
        uds["edge_z"] = xr.DataArray(
            np.arange(grid.n_edge), dims=(grid.edge_dimension,)
        )
        uds["face_z"] = xr.DataArray(
            np.arange(grid.n_node), dims=(grid.face_dimension,)
        )
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
        assert self.uds.ugrid.obj.equals(back.ugrid.obj)
