Terminology
===========

This document builds on `Xarray's glossary
<https://xarray.pydata.org/en/stable/user-guide/terminology.html>`_.  We
strongly recommend reading the Xarray Terminology document before reading this
document. The `UGRID Conventions
<https://ugrid-conventions.github.io/ugrid-conventions/>`_ are also a
recommended read.

    Grid
        A representation of a larger geometric domain by smaller discrete
        cells. "Mesh" is used somewhat interchangeabely.

    Structured Grid
        Structured grids arrange cells in a simple (``n_row, n_column)`` array.
        Structured grids are identified by regular connectivity: every cell has
        the same number of neighbors, with the exception of boundaries. Each
        cross-section has the same number of cells, even though the cell shape
        and size may differ arbitrarily (non-equidistant spacing). Cells are
        quadrilateral (four sides) in 2D. Cell to cell connectivity is implicit
        and can be directly derived from row defined from the row and column
        numbers.

    Unstructured Grid
        In contrast to a structured grid, connectivity for an unstructured grid
        is irregular and has to be defined explicitly. The primary benefit of
        unstructured grids are possibilities for local refinement. Another
        benefit is that arbitrary geometries can be easily represented.
        Unstructured grids generally arrange cells in a flat (``n_cell,``)
        array and separate arrays are used to store the cell locations.
        "Unstructured mesh" or "flexible mesh" are used interchangeably.

    Topology
        In these pages, short for "grid topology". Grid topology refers to the
        location and connectivity of the grid cells and its constituent parts
        (nodes, edges). More broadly it also refers to any connectivity
        information with respect to a grid.

    UGRID
        `Conventions <https://ugrid-conventions.github.io/ugrid-conventions/>`_
        for specifying the topology of unstructured grids. The focus of the
        UGRID conventions is environmental applications and it builds on the
        `Climate & Forecast (CF) Metadata Conventions
        <http://cfconventions.org/>`_. Data stored according to the UGRID
        conventions is thus nearly always written to Unidata Network Common
        Data Form (NetCDF) files, but the convention applies to the data and
        metadata: they can be written to any sufficiently rich file format
        (e.g. `Zarr <https://zarr.readthedocs.io/en/stable/>`_).

    Node 
        A point, a coordinate pair (x, y): the most basic element of the
        topology. "Vertex" is used interchangeably.

    Edge
        A line or cuve bounded by two nodes.

    Face
        A plane or surface enclosed by a set of edges. "Cell" is used somewhat
        interchangeably; "polygon" also, but to a lesser degree.

    Sparse Array
        A sparse matrix or sparse array is a matrix in which most elements are
        zero. For efficiency reasons, sparse matrices are commonly stored in
        special data structures, storing only the non-zero values. A
        straightforward storage scheme is Coordinate list (COO) or triplet
        format: for every non-zero value, three values are stored:
        ``(row_index, column_index, value)``. In the Python data ecosystem,
        these data structures are provided by `SciPy
        <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_; in these
        pages, "sparse array" or "sparse matrix" refers specifically to data
        stored in one of these Scipy objects.

    Dense Array
        Contrast with "sparse array": a dense array is an array in which all
        values -- including zeros or fill values -- are stored. In these pages,
        "dense array" refers to "ordinary" NumPy arrays.

    Adjacency List
        A list describing which features of the grid (e.g. faces) are
        associated with each other (e.g. nodes, or the neighboring faces).
        This can be stored as a list of lists for a grid, a rectangular array
        for regular connectivity, or a "ragged array" for irregular
        connectivity. In the UGRID conventions, both regular and irregular
        connectivity is stored in (dense) rectangular arrays; ragged arrays are
        represented by rectangular arrays partially filled with a fill value.
        
    Adjacency Matrix
        An alternative to adjacency lists is an adjacency matrix, which is a
        matrix in which the row and column numbers correspond to the element
        numbers and wherein the cell value contains a Boolean value denoting
        connectivity (``True``, ``1``) or not (``False``, ``0``); such a matrix
        can be efficiently stored as a sparse matrix.

    Face node connectivity
        An index array of integers. For every face, a list of index values
        indicating which members of the list of nodes form its (exterior)
        edges. According to UGRID conventions, this data is stored in a
        (dense) rectangular array with explicit fill values of the shape
        ``(n_face, n_max_nodes_per_face``). For a grid consisting of
        exclusively triangles, ``n_max_nodes_per_face == 3`` and no fill value
        is required; for an exclusively quadrilateral grid
        ``n_max_nodes_per_face == 4``; the fill value is only used for mixed
        grids (e.g. triangles and quandrilaterals). The numbering of the faces
        is implicit in the first index (row number) of the array; we would
        collect the index values for the first face as follows:
        ``face_node_connectivity[0]``. 

    Edge node connectivity
        An index arrray of integers. For every edge, a list of index value
        indicating which two members of the list of nodes bound a curve or
        line. This data is stored in a (dense) rectangular array of the shape
        ``(n_edge, 2)``. The numbering of the edges is implicit in the first
        index (row number) of the array. Refer to the `UGRID Conventions
        <https://ugrid-conventions.github.io/ugrid-conventions/>`_ for an
        exhaustive description of connectivities.