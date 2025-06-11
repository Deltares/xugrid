"""Contains common reduction methods."""

import math
from typing import Callable

import numba as nb
import numpy as np

from xugrid.regrid.nanpercentile import _select_two


def mean(values, weights, workspace):
    vsum = 0.0
    wsum = 0.0
    for v, w in zip(values, weights):
        if np.isnan(v):
            continue
        vsum += w * v
        wsum += w
    if wsum == 0:
        return np.nan
    else:
        return vsum / wsum


def harmonic_mean(values, weights, workspace):
    v_agg = 0.0
    w_sum = 0.0
    for v, w in zip(values, weights):
        if np.isnan(v) or v == 0:
            continue
        if w > 0:
            w_sum += w
            v_agg += w / v
    if v_agg == 0 or w_sum == 0:
        return np.nan
    else:
        return w_sum / v_agg


def geometric_mean(values, weights, workspace):
    v_agg = 0.0
    w_sum = 0.0

    # Compute sum to normalize weights to avoid tiny or huge values in exp
    normsum = 0.0
    for v, w in zip(values, weights):
        normsum += w
    # Early return if no values
    if normsum == 0:
        return np.nan

    for v, w in zip(values, weights):
        w = w / normsum
        # Skip if v is NaN or 0.
        if v > 0 and w > 0:
            v_agg += w * np.log(abs(v))
            w_sum += w
        elif v < 0:
            # Computing a geometric mean of negative numbers requires a complex
            # value.
            return np.nan

    if w_sum == 0:
        return np.nan
    else:
        # w_sum is generally 1.0, but might not be if there are NaNs present!
        return np.exp((1.0 / w_sum) * v_agg)


def sum(values, weights, workspace):
    v_sum = 0.0
    w_sum = 0.0

    for v, w in zip(values, weights):
        if np.isnan(v):
            continue
        v_sum += v
        w_sum += w
    if w_sum == 0:
        return np.nan
    else:
        return v_sum


@nb.njit(inline="always")
def _minimum(values, weights):
    v_min = np.inf
    w_max = 0.0
    for v, w in zip(values, weights):
        if np.isnan(v):
            continue
        v_min = min(v, v_min)
        w_max = max(w_max, w)
    if w_max == 0.0:
        return np.nan
    return v_min


def minimum(values, weights, workspace):
    return _minimum(values, weights)


@nb.njit(inline="always")
def _maximum(values, weights):
    v_max = -np.inf
    w_max = 0.0
    for v, w in zip(values, weights):
        if np.isnan(v):
            continue
        v_max = max(v, v_max)
        w_max = max(w_max, w)
    if w_max == 0.0:
        return np.nan
    return v_max


def maximum(values, weights, workspace):
    return _maximum(values, weights)


def mode(values, weights, workspace):
    # Area weighted mode. We use a linear search to accumulate weights, as we
    # generally expect a relatively small number of elements in the indices and
    # weights arrays.
    accum = workspace
    accum[: weights.size] = weights[:]
    w_sum = 0
    w_max = 0.0
    for running_total, (v, w) in enumerate(zip(values, weights)):
        if np.isnan(v):
            continue
        w_max = max(w, w_max)
        w_sum += 1
        for j in range(running_total):  # Compare with previously found values
            if values[j] == v:  # matches previous value
                accum[j] += w  # increase previous weight sum
                break

    if w_sum == 0 or w_max == 0.0:
        # Everything skipped (all nodata), or all weights zero
        return np.nan
    else:
        # Find value with highest frequency.
        # In case frequencies are equal (a tie), take the largest value.
        # This ensures the same result irrespective of value ordering.
        w_max = 0
        mode_value = values[0]
        for w_accum, v in zip(accum, values):
            if ~np.isnan(v):
                if (w_accum > w_max) or (w_accum == w_max and v > mode_value):
                    w_max = w_accum
                    mode_value = v
        return mode_value


@nb.njit
def percentile(values, weights, workspace, p):
    # This function is a simplified port of:
    # https://github.com/numba/numba/blob/0441bb17c7820efc2eba4fd141b68dac2afa4740/numba/np/arraymath.py#L1745

    # Exit early if all weights are 0.
    w_max = 0.0
    for w in weights:
        w_max = max(w, w_max)
    if w_max == 0.0:
        return np.nan

    if p == 0:
        return _minimum(values, weights)

    if p == 100:
        return _maximum(values, weights)

    # Everything should've been checked before:
    #
    # * a.dtype should be float
    # * 0 <= q <= 100.
    #
    # Filter the NaNs

    n = 0
    for v in values:
        if ~np.isnan(v):
            workspace[n] = v
            n += 1

    # Early returns
    if n == 0:
        return np.nan
    if n == 1:
        return workspace[0]

    # linear interp between closest ranks
    rank = 1 + (n - 1) * p / 100.0
    f = math.floor(rank)
    m = rank - f
    lower, upper = _select_two(workspace[:n], k=int(f - 1), low=0, high=(n - 1))
    return lower * (1 - m) + upper * m


def first_order_conservative(values, weights, workspace):
    # Uses relative weights!
    # Rename to: first order conservative?
    v_agg = 0.0
    w_sum = 0.0
    for v, w in zip(values, weights):
        if np.isnan(v):
            continue
        v_agg += v * w
        w_sum += w
    if w_sum == 0:
        return np.nan
    else:
        return v_agg


conductance = first_order_conservative


def max_overlap(values, weights, workspace):
    w_max = 0.0
    v_max = -np.inf
    # Find value with highest overlap.
    # In case frequencies are equal (a tie), take the largest value.
    # This ensures the same result irrespective of value ordering.
    for v, w in zip(values, weights):
        if ~np.isnan(v):
            if (w > w_max) or (w == w_max and v > v_max):
                w_max = w
                v_max = v
    if w_max == 0.0:
        return np.nan
    return v_max


def create_percentile_method(p: float) -> Callable:
    if not (0.0 <= p <= 100.0):
        raise ValueError(f"percentile must be in the range [0, 100], received: {p}")

    def f(values, weights, workspace) -> float:
        return percentile(values, weights, workspace, p)

    return f


median = create_percentile_method(50)


ABSOLUTE_OVERLAP_METHODS = {
    "mean": mean,
    "harmonic_mean": harmonic_mean,
    "geometric_mean": geometric_mean,
    "sum": sum,
    "minimum": minimum,
    "maximum": maximum,
    "mode": mode,
    "median": median,
    "max_overlap": max_overlap,
}
for p in (5, 10, 25, 50, 75, 90, 95):
    ABSOLUTE_OVERLAP_METHODS[f"p{p}"] = create_percentile_method(p)


RELATIVE_OVERLAP_METHODS = {
    "conductance": conductance,
    "first_order_conservative": first_order_conservative,
}
