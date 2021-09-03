from typing import Tuple
import numpy as np

from ..typing import IntArray


def intersect_coordinates_1d(
    a: np.ndarray, b: np.ndarray
) -> Tuple[IntArray, IntArray, np.ndarray]:
    """
    Intersects two arrays of one-dimensional coordinates.

    Returns indices of ``a``, ``b``, and the length of every occuring overlap.

    Parameters
    ----------
    a: ndarray with shape ``(n + 1,)``
        Cell boundaries along a single dimension for ``n`` cells.
    b: ndarray with shape ``(m + 1)``
        Cell boundaries along a single dimension for ``m`` cells.

    Returns
    -------
    indices_a: ndarray of integers with shape ``(n_overlap,)``
    indices_b: ndarray of integers with shape ``(n_overlap,)``
    overlap: ndarray with shape ``(n_overlap)``

    Examples
    --------
    >>> a = np.array([0.0, 2.5, 5.0])
    >>> b = np.array([0.0, 1.0, 2.0, 3.0])
    >>> ia, ib, overlap = intersect_coordinates(a, b)
    >>> ia
    array([0, 0, 0, 1])
    >> ib
    array([0, 1, 2, 2])
    >> overlap
    array([1., 1., 0.5, 0.5])
    """
    # Split into starts and ends
    start = a[:-1]
    end = a[1:]
    lower = b[:-1]
    upper = b[1:]

    # Find where the values of b belong in a.
    i_start = np.searchsorted(start, lower, side="right")
    i_end = np.searchsorted(end, upper, side="left") + 1

    # This is equal to "vectorized arange"; we generate a range of numbers for
    # for every start to every end.
    n = i_end - i_start + 1
    n_total = n.sum()
    i = np.repeat(i_end - n.cumsum(), n) + np.arange(n_total)
    # Also generate the accompanying numbers for b.
    j = np.repeat(np.arange(b.size - 1), n)
    # Get rid of values that are out of bounds: these might've been generated
    # by searchsorted.
    in_bounds = (i >= 0) & (i < a.size - 1)
    i = i[in_bounds]
    j = j[in_bounds]

    # Create the intervals to search against each other.
    left = np.column_stack((start[i], lower[j]))
    right = np.column_stack((end[i], upper[j]))
    overlap = np.min(right, axis=1) - np.max(left, axis=1)
    # Return only when actual overlap occurs.
    valid = overlap > 0
    return i[valid], j[valid], overlap[valid]
