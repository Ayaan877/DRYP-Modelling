from datetime import date, timedelta
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import watershed_operations

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

# Names: "AET 750m", "AET 5500m", "SSEBop 1000m", "SSEBop 30m", "ETf 750m", "ETf gapfilled 750m",
# "ref ET ERA5", "ref ET agERA5". Combine with + - * /, e.g. "SSEBop 30m/ref ET ERA5"
# Generated ETf folders are available as, e.g., "ETf Bhuvan ERA5 Land 5500m" or "ETf SSEBop ERA5 Land 1000m".
# Gap-filled ETf: e.g., "ETf Bhuvan ERA5 Land gapfilled 750m" or "ETf SSEBop ERA5 Land gapfilled 30m".
VARIABLES = ["ETf SSEBop ERA5 Land gapfilled 30m", "ETf SSEBop ERA5 Land 30m"]
YEARS = list(range(2015, 2017))
N_COLUMNS = 2

# True: count missing days before the first / after the last observation
INCLUDE_EDGE_GAPS = True
# True: skip pixels with no valid value all year (e.g. outside the watershed)
EXCLUDE_ALWAYS_NODATA_PIXELS = True
# True: dates absent from the file count as nodata in every pixel's series
MISSING_DATES_COUNT_AS_NODATA = True

BIN_WIDTH = 1  # days
SUBPLOT_SIZE = (6, 3)  # width, height of each panel
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def period_slots(year, cadence_days):
	"""Expected observation dates, daily or at the starts of dekads."""
	if cadence_days >= 10:
		return [date(year, month, day) for month in range(1, 13) for day in (1, 11, 21)]
	n_days = (date(year, 12, 31) - date(year, 1, 1)).days + 1
	return [date(year, 1, 1) + timedelta(days=i) for i in range(n_days)]


def load_validity_cube(variable, year):
	"""Boolean cube (periods, rows, columns) for a dataset or arithmetic expression."""
	series = watershed_operations.resolve_request(variable, year)
	slots = period_slots(year, series.cadence_days)
	slot_index = {d: i for i, d in enumerate(slots)}
	cube = np.zeros((len(slots),) + series.values.shape[1:], dtype=bool)
	listed = np.zeros(len(slots), dtype=bool)
	for index, observed_date in enumerate(series.dates):
		if observed_date not in slot_index:
			continue
		slot = slot_index[observed_date]
		cube[slot] = np.isfinite(series.values[index])
		listed[slot] = True
	return cube, listed, series.cadence_days


def pixel_gaps(cube, listed):
	"""Nodata gap lengths pooled over all pixels."""
	if not MISSING_DATES_COUNT_AS_NODATA:
		cube = cube[listed]
	pixels = cube.reshape(cube.shape[0], -1)
	if EXCLUDE_ALWAYS_NODATA_PIXELS:
		pixels = pixels[:, pixels.any(axis=0)]
	if pixels.shape[1] == 0 or pixels.shape[0] == 0:
		return np.array([], dtype=int), pixels.shape[1]
	runs = np.zeros(pixels.shape[1], dtype=np.int32)
	seen_valid = np.zeros(pixels.shape[1], dtype=bool)
	gap_lengths = []
	for present in pixels:
		missing = ~present
		runs += missing
		completed = present & (runs > 0)
		keep = completed if INCLUDE_EDGE_GAPS else completed & seen_valid
		if keep.any():
			gap_lengths.append(runs[keep].copy())
		runs[completed] = 0
		seen_valid |= present
	if INCLUDE_EDGE_GAPS:
		trailing = runs > 0
		if trailing.any():
			gap_lengths.append(runs[trailing].copy())
	if not gap_lengths:
		return np.array([], dtype=int), pixels.shape[1]
	return np.concatenate(gap_lengths), pixels.shape[1]


def plot_year(ax, year):
	results = []
	for variable in VARIABLES:
		cube, listed, cadence_days = load_validity_cube(variable, year)
		gaps, n_pixels = pixel_gaps(cube, listed)
		results.append((variable, gaps * cadence_days, n_pixels))
	max_gap = max((int(gaps.max()) for _, gaps, _ in results if gaps.size), default=0)
	if max_gap:
		bins = np.arange(0.5, max_gap + BIN_WIDTH + 0.5, BIN_WIDTH)
		for index, (variable, gaps, n_pixels) in enumerate(results):
			if gaps.size and n_pixels:
				weights = np.full(gaps.shape, 1 / n_pixels)
				ax.hist(gaps, bins=bins, weights=weights, color=COLORS[index % len(COLORS)], alpha=0.45, label=variable)
	ax.set_xlabel("Gap length (consecutive nodata days)")
	ax.set_ylabel("Average number of gaps per pixel")
	counts = ", ".join(
		f"{name}: {pixels} px, {len(gaps)} gaps ({len(gaps) / pixels:.2f} gaps/px)" if pixels
		else f"{name}: 0 px, {len(gaps)} gaps"
		for name, gaps, pixels in results
	)
	ax.set_title(f"{year}: {counts}")


def plot_years(years):
	n_cols = min(N_COLUMNS, len(years))
	n_rows = -(-len(years) // n_cols)
	fig, axes = plt.subplots(
		n_rows, n_cols,
		figsize=(SUBPLOT_SIZE[0] * n_cols, SUBPLOT_SIZE[1] * n_rows),
		squeeze=False,
	)
	flat_axes = axes.ravel()
	for ax, year in zip(flat_axes, years):
		plot_year(ax, year)
	for ax in flat_axes[len(years):]:
		ax.set_visible(False)
	handles = [
		Patch(facecolor=COLORS[index % len(COLORS)], alpha=0.45, label=variable)
		for index, variable in enumerate(VARIABLES)
	]
	fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), frameon=False)
	fig.tight_layout(rect=(0, 0.08, 1, 1))
	plt.show()


if __name__ == "__main__":
	plot_years(YEARS)
