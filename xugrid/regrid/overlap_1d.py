import numba
import numpy as np

from xugrid.constants import IntDType
from xugrid.regrid.utils import alt_cumsum


@numba.njit(inline="always")
def minmax(v, lower, upper):
    return min(upper, max(lower, v))


def _find_indices(side):
    @numba.njit(inline="always")
    def lt(x, y):
        return x < y

    @numba.njit(inline="always")
    def le(x, y):
        return x <= y

    if side == "right":
        func = le
        add = -1
    elif side == "left":
        func = lt
        add = 1
    else:
        raise ValueError("side must be left or right")

    @numba.njit
    def searchsorted_inner(a, v, v_last, lo, hi, n):
        """
        Perform inner loop of searchsorted (i.e. a binary search).
        This is loosely based on the NumPy implementation in [1]_.

        Parameters
        ----------
        a: 1-D array_like
            The input array.
        v: array_like
            The current value to insert into `a`.
        v_last: array_like
            The previous value inserted into `a`.
        lo: int
            The initial/previous "low" value of the binary search.
        hi: int
            The initial/previous "high" value of the binary search.
        n: int
            The length of `a`.
        .. [1] https://github.com/numpy/numpy/blob/809e8d26b03f549fd0b812a17b8a166bcd966889/numpy/core/source/npysort/binsearch.cpp#L173
        """  # noqa: E501
        if v_last < v:
            hi = n
        else:
            lo = 0
            hi = hi + 1 if hi < n else n

        while hi > lo:
            mid = (lo + hi) >> 1
            if func(a[mid], v):
                # mid is too low => go up
                lo = mid + 1
            else:
                # mid is too high, or is a NaN => go down
                hi = mid
        return lo

    @numba.njit
    def preallocated_searchsorted(a, v, out, nan_helper, sorter) -> None:
        # source bounds may contain NaN values. Eliminate those here and store
        # the original positions in the sorter.
        jj = 0
        for ii, vv in enumerate(a):
            if np.isnan(vv):
                continue
            nan_helper[jj] = vv
            sorter[jj] = ii
            jj += 1

        n = jj - 1
        lo = 0
        hi = n
        v_last = v[0]
        for i in range(len(v)):
            v_search = v[i]
            if np.isnan(v_search):
                continue
            lo = searchsorted_inner(nan_helper, v_search, v_last, lo, hi, n)
            v_last = v_search
            out[i] = minmax(sorter[lo] + add, 0, a.size)
        return

    @numba.njit
    def find_indices(
        source,
        target,
        source_index,
        target_index,
    ):
        """
        Find the indices of target in source. Allocate the result in slices of
        source_index and target_index.

        This is basically a workaround. Numpy searchsorted does not support an axis
        argument to search one nD array on the other.
        See: https://github.com/numpy/numpy/issues/4224

        Fortunately, numba implements searchsorted here:
        https://github.com/numba/numba/blob/f867999c7453141642ea9af21febef796da9ca93/numba/np/arraymath.py#L3647
        But it continously allocates new result arrays.

        Parameters
        ----------
        source: np.ndarray of shape (n, m)
            All other dims flatted to dimension with size n.
        target: np.ndarray of shape (n, m)
            All other dims flatted to dimension with size n.
        source_index: np.ndarray of shape (n_index,)
        target_index: np.ndarray of shape (n_index,)
        """
        _, n = source.shape
        _, m = target.shape
        # Will contain the index along dimension m.
        indices = np.full((source_index.size, m), -1, dtype=IntDType)
        sorter = np.empty(n, IntDType)
        nan_helper = np.empty(n, source.dtype)
        for k, (i, j) in enumerate(zip(source_index, target_index)):
            source_i = source[i, :]
            target_j = target[j, :]
            preallocated_searchsorted(
                source_i, target_j, indices[k], nan_helper, sorter
            )
        return indices

    return find_indices


find_lower_indices = _find_indices("right")
find_upper_indices = _find_indices("left")


def vectorized_overlap(bounds_a, bounds_b):
    """
    Vectorized overlap computation.

    Compare with:

    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    """
    return np.maximum(
        0.0,
        np.minimum(bounds_a[:, 1], bounds_b[:, 1])
        - np.maximum(bounds_a[:, 0], bounds_b[:, 0]),
    )


def overlap_1d_nd(
    source_bounds,
    target_bounds,
    source_index,
    target_index,
):
    """
    As this uses a binary search, bounds must be monotonic ascending.

    source_bounds and target_bounds may both contain NaN values to denote NoData.

    Parameters
    ----------
    source_bounds: np.ndarray with shape (n, source_size, 2)
    target_bounds: np.ndarray with shape (m, target_size, 2)
    source_index: np.ndarray with shape (o,)
        Used to index into n axis of source_bounds
    target_index: np.ndarray with shape (o,)
        Used to index into m axis of target_bounds

    Returns
    -------
    flat_source_index: np.ndarray of integers with shape (p,)
    flat_target_index: np.ndarray of integers with shape (p,)
    overlap: np.ndarray of floats with shape (p,)
    """
    # _d refers to the dimension of interest _nd refers to the (variable!)
    # number of other dimensions which are flattened and come before d.

    _, target_d_size, _ = target_bounds.shape

    source_lower = source_bounds[..., 0]
    source_upper = source_bounds[..., 1]
    target_lower = target_bounds[..., 0]
    target_upper = target_bounds[..., 1]

    lower_indices = find_lower_indices(
        source_lower, target_lower, source_index, target_index
    )
    upper_indices = find_upper_indices(
        source_upper, target_upper, source_index, target_index
    )

    n_overlap = upper_indices - lower_indices
    n_overlap_nd = n_overlap.sum(axis=1)
    n_total = n_overlap_nd.sum()

    # Create target index
    target_index_d = np.repeat(
        np.broadcast_to(
            np.arange(target_d_size, dtype=IntDType), target_bounds.shape[:2]
        ).ravel(),
        n_overlap.ravel(),
    )

    # Create source index
    increment = alt_cumsum(np.ones(n_total, dtype=IntDType)) - np.repeat(
        alt_cumsum(n_overlap.ravel()), n_overlap.ravel()
    )
    source_index_d = (
        np.repeat(
            lower_indices.ravel(),
            n_overlap.ravel(),
        )
        + increment
    )

    # Now turn them into a linear index.
    target_linear_index = np.ravel_multi_index(
        (np.repeat(target_index, n_overlap_nd), target_index_d),
        dims=target_bounds.shape[:2],
    )
    source_linear_index = np.ravel_multi_index(
        (np.repeat(source_index, n_overlap_nd), source_index_d),
        dims=source_bounds.shape[:2],
    )

    # Compute overlap.
    overlap = vectorized_overlap(
        source_bounds.reshape((-1, 2))[source_linear_index],
        target_bounds.reshape((-1, 2))[target_linear_index],
    )
    valid = overlap > 0.0
    return (source_linear_index[valid], target_linear_index[valid], overlap[valid])


def overlap_1d(
    source_bounds,
    target_bounds,
):
    return overlap_1d_nd(
        source_bounds[np.newaxis],
        target_bounds[np.newaxis],
        np.array([0]),
        np.array([0]),
    )
