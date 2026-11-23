# Locally weighted regression (LOESS) time series interpolation.
#
# Callable:
#   lwr_interpolate(x, y, x_new=None, step=1, poly_order=2, n_neighbours=7,
#                   max_gap=None, clip_min=None, clip_max=None, blend_days=0)
#
# Inputs:
#   x            observation times (numbers, datetime64 or date objects), shape (n,)
#   y            observations, shape (n, ...); time on axis 0, NaN = missing
#   x_new        optional output times (default: regular grid of `step` from first to last x)
#   step         grid spacing when x_new is None (days for dates)
#   poly_order   degree of the local polynomial
#   n_neighbours observations per fit, taken half from each side of the target (interpolation, not extrapolation)
#   max_gap      gaps between bracketing observations wider than this stay NaN
#   blend_days   fit errors at the bracketing observations are added back, fading to 0 over this many days
#                (0 disables); the output is continuous and equals every observation exactly
#   clip_min/max bounds for interpolated values (observed values untouched)
#
# Outputs:
#   x_new, y_new   output times (days for dates) and the filled series, shape (len(x_new), ...)
#                  NaN outside the first/last valid observation or in skipped gaps

import numpy as np
from numba import njit, prange

BLOCK_COLUMNS = 256

def to_numbers(x):
	x = np.asarray(x)
	if np.issubdtype(x.dtype, np.datetime64):
		return x.astype("datetime64[D]").astype(np.float64)
	if x.dtype == object:
		return np.array([v.toordinal() if hasattr(v, "toordinal") else v for v in x], dtype=np.float64)
	return x.astype(np.float64)

def nan_if_none(value):
	return np.nan if value is None else float(value)

@njit(cache=True)
def local_fit(vx, vy, target, p, poly_order, s):
	# Window = s points left of split p and s right of it (shifted to one side near the series ends).
	n = vx.size
	nl = min(p, s)
	nr = min(n - p, s)
	if nl < s:
		nr = min(n - p, 2 * s - nl)
	elif nr < s:
		nl = min(p, 2 * s - nr)
	lo = p - nl
	hi = p + nr
	dx = vx[lo:hi] - target
	d = np.abs(dx)
	root_w = np.sqrt(np.maximum(0.0, 1 - (d / d.max()) ** 3) ** 3)  # zero weight at the farthest point
	design = np.empty((hi - lo, poly_order + 1))
	for r in range(hi - lo):
		for c in range(poly_order + 1):
			design[r, c] = dx[r] ** c * root_w[r]
	return np.linalg.lstsq(design, vy[lo:hi] * root_w)[0][0]


@njit(cache=True)
def blend_weight(d, blend_days):
	if d >= blend_days:
		return 0.0
	return (1 - (d / blend_days) ** 3) ** 3


@njit(cache=True)
def fill_valid(vx, vy, x_new, poly_order, n_neighbours, max_gap, blend_days, clip_min, clip_max, out):
	n = vx.size
	out[:] = np.nan
	if n == 0:
		return
	if n == 1:
		for i in range(x_new.size):
			if x_new[i] == vx[0]:
				out[i] = vy[0]
		return
	order = min(poly_order, n - 2)
	k = min(max(n_neighbours, order + 1), n - 1)
	s = (k + 3) // 2
	blend = blend_days > 0
	# Fit error at each observation, from the window just before (left) and just after (right) it.
	res_left = np.zeros(n)
	res_right = np.zeros(n)
	if blend:
		for j in range(n):
			res_left[j] = vy[j] - local_fit(vx, vy, vx[j], j, order, s)
			res_right[j] = vy[j] - local_fit(vx, vy, vx[j], j + 1, order, s)
	for i in range(x_new.size):
		target = x_new[i]
		if target < vx[0] or target > vx[-1]:
			continue
		p = np.searchsorted(vx, target)
		if vx[p] == target:
			out[i] = vy[p]
			continue
		if not np.isnan(max_gap) and vx[p] - vx[p - 1] > max_gap:
			continue
		value = local_fit(vx, vy, target, p, order, s)
		if blend:
			dl = target - vx[p - 1]
			dr = vx[p] - target
			gap = dl + dr
			value += (dr / gap) * blend_weight(dl, blend_days) * res_right[p - 1]
			value += (dl / gap) * blend_weight(dr, blend_days) * res_left[p]
		if not np.isnan(clip_min):
			value = max(value, clip_min)
		if not np.isnan(clip_max):
			value = min(value, clip_max)
		out[i] = value


@njit(cache=True, parallel=True)
def lwr_block(x, block, x_new, poly_order, n_neighbours, max_gap, blend_days, clip_min, clip_max, out):
	for j in prange(block.shape[1]):
		col = block[:, j]
		valid = ~np.isnan(col)
		fill_valid(x[valid], col[valid], x_new, poly_order, n_neighbours, max_gap, blend_days, clip_min, clip_max, out[j])


def fill_series(x, y, x_new, poly_order, n_neighbours, max_gap, clip_min, clip_max, blend_days=0.0):
	x = np.ascontiguousarray(x, dtype=np.float64)
	y = np.asarray(y, dtype=np.float64)
	valid = ~np.isnan(y)
	out = np.empty(x_new.size)
	fill_valid(x[valid], y[valid], np.ascontiguousarray(x_new, dtype=np.float64), poly_order, n_neighbours,
				nan_if_none(max_gap), float(blend_days or 0.0), nan_if_none(clip_min), nan_if_none(clip_max), out)
	return out


def lwr_interpolate(x, y, x_new=None, step=1, poly_order=2, n_neighbours=7,
					max_gap=None, clip_min=None, clip_max=None, progress=None, blend_days=0.0):
	x = to_numbers(x)
	y = np.asarray(y, dtype=np.float64)
	if y.shape[0] != x.size:
		raise ValueError("y must have the same length as x along axis 0")
	order = np.argsort(x, kind="stable")
	x, y = x[order], y[order]
	if np.any(np.diff(x) == 0):
		raise ValueError("x contains duplicate times")
	if x_new is None:
		x_new = np.arange(x[0], x[-1] + step / 2, step)
	else:
		x_new = np.sort(to_numbers(x_new))
	x = np.ascontiguousarray(x, dtype=np.float64)
	x_new = np.ascontiguousarray(x_new, dtype=np.float64)
	flat = y.reshape(x.size, -1)
	out = np.full((flat.shape[1], x_new.size), np.nan)
	for start in range(0, flat.shape[1], BLOCK_COLUMNS):
		stop = min(start + BLOCK_COLUMNS, flat.shape[1])
		block = np.ascontiguousarray(flat[:, start:stop])
		lwr_block(x, block, x_new, poly_order, n_neighbours, nan_if_none(max_gap), float(blend_days or 0.0),
				   nan_if_none(clip_min), nan_if_none(clip_max), out[start:stop])
		if progress is not None:
			progress(stop - start)
	return x_new, out.T.reshape((x_new.size,) + y.shape[1:])
