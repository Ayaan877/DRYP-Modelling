import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
from pathlib import Path
from netCDF4 import Dataset

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET = "SSEBop"  # "Bhuvan" or "SSEBop"
YEARS = [2021, 2022, 2023, 2024, 2025]
N_COLUMNS = 2

# Exported watershed NetCDFs from watershed_aet.py
if DATASET == "SSEBop":
	RESOLUTION_LABEL = "1000m"
	DATA_DIR = SCRIPT_DIR / "Watershed AET SSEBop"
	FILE_PREFIX = "watershed_ssebop_AET"
elif DATASET == "Bhuvan":
	RESOLUTION_LABEL = "750m"
	DATA_DIR = SCRIPT_DIR / f"Watershed AET {RESOLUTION_LABEL}"
	FILE_PREFIX = "watershed_bhuvan_AET"
else:
	raise ValueError('DATASET must be set to "Bhuvan" or "SSEBop"')

TIME_ORIGIN = date(2015, 1, 1)  # days-since origin of the "time" variable
TIME_VARIABLE = "time"
AET_VARIABLE = "aet"
NODATA = -99999

# True: count missing days before the first / after the last observation
INCLUDE_EDGE_GAPS = True
# True: skip pixels with no valid value all year (e.g. outside the watershed)
EXCLUDE_ALWAYS_NODATA_PIXELS = True
# True: dates absent from the file count as nodata in every pixel's series
MISSING_DATES_COUNT_AS_NODATA = True

BIN_WIDTH = 1  # days for Bhuvan, dekads for SSEBop
SUBPLOT_SIZE = (6, 3)  # width, height of each panel

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def nc_file_for(year):
	return DATA_DIR / f"{FILE_PREFIX}_{RESOLUTION_LABEL}_{year}.nc"


def period_slots(year):
	"""Expected observation dates of the year: every day (Bhuvan) or dekad starts (SSEBop)."""
	if DATASET == "SSEBop":
		return [date(year, month, day) for month in range(1, 13) for day in (1, 11, 21)]
	n_days = (date(year, 12, 31) - date(year, 1, 1)).days + 1
	return [date(year, 1, 1) + timedelta(days=i) for i in range(n_days)]


def load_validity_cube(year):
	"""Boolean cube (periods in year, lat, lon): True where the exported watershed file has valid data."""
	slots = period_slots(year)
	slot_index = {d: i for i, d in enumerate(slots)}
	with Dataset(nc_file_for(year)) as ds:
		times = np.ma.filled(ds[TIME_VARIABLE][:], -1).astype(int)
		aet = ds[AET_VARIABLE]
		cube = np.zeros((len(slots),) + tuple(aet.shape[1:]), dtype=bool)
		listed = np.zeros(len(slots), dtype=bool)
		for i, t in enumerate(times):
			d = TIME_ORIGIN + timedelta(days=int(t))
			if d not in slot_index:
				continue
			frame = np.ma.filled(aet[i].astype(float), np.nan)
			cube[slot_index[d]] = np.isfinite(frame) & (frame != NODATA)
			listed[slot_index[d]] = True
	return cube, listed

def run_lengths(present):
	"""Lengths of consecutive False runs in a 1-D boolean series."""
	n = len(present)
	if not present.any():
		return [n] if INCLUDE_EDGE_GAPS else []
	first = int(np.argmax(present))
	last = n - 1 - int(np.argmax(present[::-1]))
	start, end = (0, n) if INCLUDE_EDGE_GAPS else (first, last + 1)
	gaps, run = [], 0
	for p in present[start:end]:
		if p:
			if run:
				gaps.append(run)
			run = 0
		else:
			run += 1
	if run:
		gaps.append(run)
	return gaps


def pixel_gaps(cube, listed):
	"""Nodata gap lengths pooled over all pixels."""
	if not MISSING_DATES_COUNT_AS_NODATA:
		# Drop absent dates so gaps are measured across observed time steps only
		cube = cube[listed]
	gaps, n_pixels = [], 0
	for r in range(cube.shape[1]):
		for c in range(cube.shape[2]):
			series = cube[:, r, c]
			if EXCLUDE_ALWAYS_NODATA_PIXELS and not series.any():
				continue
			n_pixels += 1
			gaps.extend(run_lengths(series))
	return gaps, n_pixels


def plot_year(ax, year):
	cube, listed = load_validity_cube(year)
	gaps, n_pixels = pixel_gaps(cube, listed)
	if gaps:
		bins = [float(b) for b in np.arange(0.5, max(gaps) + BIN_WIDTH + 0.5, BIN_WIDTH)]
		ax.hist(gaps, bins=bins, edgecolor="black")
	period_label = "dekads" if DATASET == "SSEBop" else "days"
	ax.set_xlabel(f"Gap length (consecutive nodata {period_label})")
	ax.set_ylabel("Number of gaps (all pixels)")
	dataset_label = f"{'SSEBop AET' if DATASET == 'SSEBop' else 'AET'} {RESOLUTION_LABEL}"
	ax.set_title(f"{dataset_label} {year}: {n_pixels} pixels, {len(gaps)} gaps")


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
	fig.tight_layout()
	plt.show()


if __name__ == "__main__":
	plot_years(YEARS)
