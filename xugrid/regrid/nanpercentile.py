"""
Numba percentile methods allocate continuously on the heap.

This has significant overhead when calling the reduction method millions of
times -- as happens when regridding.

This is a simplified port of the percentile helpers:
# https://github.com/numba/numba/blob/0441bb17c7820efc2eba4fd141b68dac2afa4740/numba/np/arraymath.py#L1595
"""
import numba as nb
import numpy as np


@nb.njit(inline="always")
def nan_le(a, b) -> bool:
    # Nan-aware <
    if np.isnan(a):
        return False
    elif np.isnan(b):
        return True
    else:
        return a < b


@nb.njit
def _partition(A, low, high):
    mid = (low + high) >> 1
    # NOTE: the pattern of swaps below for the pivot choice and the
    # partitioning gives good results (i.e. regular O(n log n))
    # on sorted, reverse-sorted, and uniform arrays.  Subtle changes
    # risk breaking this property.

    # Use median of three {low, middle, high} as the pivot
    if nan_le(A[mid], A[low]):
        A[low], A[mid] = A[mid], A[low]
    if nan_le(A[high], A[mid]):
        A[high], A[mid] = A[mid], A[high]
    if nan_le(A[mid], A[low]):
        A[low], A[mid] = A[mid], A[low]
    pivot = A[mid]

    A[high], A[mid] = A[mid], A[high]
    i = low
    j = high - 1
    while True:
        while i < high and nan_le(A[i], pivot):
            i += 1
        while j >= low and nan_le(pivot, A[j]):
            j -= 1
        if i >= j:
            break
        A[i], A[j] = A[j], A[i]
        i += 1
        j -= 1
    # Put the pivot back in its final place (all items before `i`
    # are smaller than the pivot, all items at/after `i` are larger)
    A[i], A[high] = A[high], A[i]
    return i


@nb.njit
def _select(arry, k, low, high):
    """Select the k'th smallest element in array[low:high + 1]."""
    i = _partition(arry, low, high)
    while i != k:
        if i < k:
            low = i + 1
            i = _partition(arry, low, high)
        else:
            high = i - 1
            i = _partition(arry, low, high)
    return arry[k]


@nb.njit
def _select_two(arry, k, low, high):
    """
    Select the k'th and k+1'th smallest elements in array[low:high + 1].

    This is significantly faster than doing two independent selections
    for k and k+1.
    """
    while True:
        assert high > low  # by construction
        i = _partition(arry, low, high)
        if i < k:
            low = i + 1
        elif i > k + 1:
            high = i - 1
        elif i == k:
            _select(arry, k + 1, i + 1, high)
            break
        else:  # i == k + 1
            _select(arry, k, low, i - 1)
            break

    return arry[k], arry[k + 1]
