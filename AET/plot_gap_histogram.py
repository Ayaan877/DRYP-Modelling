from datetime import date, datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import watershed_operations

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

# Names: "<data> <source> <timescale> <resolution> [gapfilling] [version]"
# data: AET, ETf, Irr, GW, PET, refET
# source: bhuvan, ssebop, ssebop_era5, rainfed, bhuvan_era5, converted, agera5, era5
# timescale: hourly, daily, dekadal
# resolution: 30m, 250m, 750m, 1000m, 5500m, 10km, native
# gapfilling: temporal, spatiotemporal (ETf only)
# version: v1, v2 (hourly and gapfilled datasets only)
# Combine with + - * /, e.g. "AET ssebop daily 30m/refET era5 daily 10km"
VARIABLES = ["Irr ssebop_era5 hourly 250m v2", "GW ssebop_era5 hourly 250m v2"]
YEARS = list(range(2015, 2025))
N_COLUMNS = 3
SAVE_PLOT = True

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

def period_slots(year, cadence_days, hourly=False):
	"""Expected observation times, hourly, daily, or at the starts of dekads."""
	if hourly:
		start = datetime(year, 1, 1)
		n_hours = (datetime(year + 1, 1, 1) - start).days * 24
		return [start + timedelta(hours=i) for i in range(n_hours)]
	if cadence_days >= 10:
		return [date(year, month, day) for month in range(1, 13) for day in (1, 11, 21)]
	n_days = (date(year, 12, 31) - date(year, 1, 1)).days + 1
	return [date(year, 1, 1) + timedelta(days=i) for i in range(n_days)]


def load_validity_cube(variable, year):
	"""Boolean cube (periods, rows, columns) for a dataset or arithmetic expression."""
	series = watershed_operations.resolve_request(variable, year)
	hourly = bool(series.dates and isinstance(series.dates[0], datetime))
	slots = period_slots(year, series.cadence_days, hourly=hourly)
	slot_index = {d: i for i, d in enumerate(slots)}
	cube = np.zeros((len(slots),) + series.values.shape[1:], dtype=bool)
	listed = np.zeros(len(slots), dtype=bool)
	for index, observed_date in enumerate(series.dates):
		if observed_date not in slot_index:
			continue
		slot = slot_index[observed_date]
		cube[slot] = np.isfinite(series.values[index])
		listed[slot] = True
	return cube, listed, series.cadence_days, hourly


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
		cube, listed, cadence_days, hourly = load_validity_cube(variable, year)
		gaps, n_pixels = pixel_gaps(cube, listed)
		results.append((variable, gaps, n_pixels, cadence_days, hourly))
	use_hours = any(hourly for _, _, _, _, hourly in results)
	results = [
		(variable, gaps * (1 if hourly else cadence_days * (24 if use_hours else 1)), n_pixels)
		for variable, gaps, n_pixels, cadence_days, hourly in results
	]
	max_gap = max((int(gaps.max()) for _, gaps, _ in results if gaps.size), default=0)
	if max_gap:
		bin_width = BIN_WIDTH * (24 if use_hours else 1)
		bins = np.arange(0.5, max_gap + bin_width + 0.5, bin_width)
		for index, (variable, gaps, n_pixels) in enumerate(results):
			if gaps.size and n_pixels:
				weights = np.full(gaps.shape, 1 / n_pixels)
				ax.hist(gaps, bins=bins, weights=weights, color=COLORS[index % len(COLORS)], alpha=0.45, label=variable)
	unit = "hours" if use_hours else "days"
	ax.set_xlabel(f"Gap length (consecutive nodata {unit})")
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
	if SAVE_PLOT:
		fig.savefig(Path(__file__).with_suffix(".png"), dpi=300, bbox_inches="tight")
		plt.close(fig)
	else:
		plt.show()


if __name__ == "__main__":
	plot_years(YEARS)
