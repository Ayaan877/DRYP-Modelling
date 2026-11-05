import re
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from datetime import datetime, timedelta
from netCDF4 import Dataset
from pathlib import Path

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
VARIABLES = ["ETf", "ETf gapfilled"]
YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
RESOLUTION_LABEL = "750m" 
SSEBOP_RESOLUTION_LABEL = "1000m"
N_COLUMNS = 3

# AET / ETf: daily GeoTIFFs in per-year folders. PET: one NetCDF file per year.
VARIABLE_SETTINGS = {
	"AET": (f"Watershed AET {RESOLUTION_LABEL}", f"watershed_bhuvan_AET_{RESOLUTION_LABEL}_*.tif", None, "tab:blue"),
	"ETf": (f"Watershed ETf {RESOLUTION_LABEL}", f"watershed_ETf_{RESOLUTION_LABEL}_*.tif", None, "tab:green"),
	"ETf gapfilled": (f"Watershed ETf {RESOLUTION_LABEL} - gapfilled", f"watershed_ETf_{RESOLUTION_LABEL}_*.tif", None, "tab:orange"),
	"PET": ("Daily PET", "{year}_daily_pet_Badenhalli_subwatershed.nc", "pet", "tab:red"),
	"SSEBop": ("Watershed AET SSEBop", f"watershed_ssebop_AET_{SSEBOP_RESOLUTION_LABEL}_*.tif", None, "tab:purple"),
}
VARIABLE_RESOLUTIONS = {"SSEBop": SSEBOP_RESOLUTION_LABEL, "ETf gapfilled": RESOLUTION_LABEL}
DATE_PATTERN = re.compile(r"(\d{8})")
VARIABLE_UNITS = {
	"AET": "mm/day",
	"ETf": "dim-less",
	"ETf gapfilled": "dim-less",
	"PET": "mm/day",
	"SSEBop": "mm/day",
}
DEKAD_DAYS = 10
NC_TIME_VARIABLE = "time"
NODATA = -99999

PIXEL_ALPHA = 0.1
PIXEL_LINEWIDTH = 0.6
MEAN_LINEWIDTH = 2
SD_ALPHA = 0.5
SINGLE_MEAN_COLOR = "C0"
SINGLE_SD_COLOR = "C1" 
SUBPLOT_SIZE = (6, 3)  # width, height of each panel

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def load_netcdf_series(variable, year):
	"""Dates and cube (time, lat, lon) for the year from a yearly NetCDF file."""
	folder, pattern, nc_variable, _ = VARIABLE_SETTINGS[variable]
	with Dataset(SCRIPT_DIR / folder / pattern.format(year=year)) as ds:
		time_var = ds[NC_TIME_VARIABLE]
		origin = re.search(r"since (\d{4}-\d{2}-\d{2})", time_var.units)
		if not origin:
			raise ValueError(f"Unsupported time units: {time_var.units}")
		start = datetime.strptime(origin.group(1), "%Y-%m-%d").date()
		offsets = np.ma.filled(time_var[:], -1).astype(int)
		dates = [start + timedelta(days=int(t)) for t in offsets]
		keep = [i for i, d in enumerate(dates) if d.year == year]
		data = np.ma.filled(ds[nc_variable][keep].astype(float), np.nan)
	data[data == NODATA] = np.nan
	return [dates[i] for i in keep], data


def load_series(variable, year):
	"""Dates and cube (time, row, col) for the year, nodata as NaN."""
	folder, pattern, nc_variable, _ = VARIABLE_SETTINGS[variable]
	if nc_variable:
		return load_netcdf_series(variable, year)
	resolution = VARIABLE_RESOLUTIONS.get(variable, RESOLUTION_LABEL)
	year_dir = SCRIPT_DIR / folder / f"{year}_{resolution}"
	dates, frames = [], []
	for tif in sorted(year_dir.glob(pattern)):
		match = DATE_PATTERN.search(tif.stem)
		if not match:
			continue
		with rasterio.open(tif) as src:
			frames.append(src.read(1).astype(float))
		dates.append(datetime.strptime(match.group(1), "%Y%m%d").date())
	data = np.array(frames)
	data[data == NODATA] = np.nan
	if variable == "SSEBop":
		data = data / DEKAD_DAYS  # mm/dekad -> average mm/day
	return dates, data


def draw_variable(ax, variable, year, show_pixels):
	"""Draw one dataset's mean and ±1 std dev band (plus pixel lines if show_pixels); return pixel count."""
	color = VARIABLE_SETTINGS[variable][3]
	dates, data = load_series(variable, year)
	series = data.reshape(len(dates), -1)
	series = series[:, np.isfinite(series).any(axis=0)]
	x = np.array(dates, dtype="datetime64[D]")

	# Each pixel is drawn through its own valid observations only, so lines stay
	# continuous across gaps (straight segments are display only, no values added)
	if show_pixels:
		for j in range(series.shape[1]):
			valid = np.isfinite(series[:, j])
			ax.plot(x[valid], series[valid, j], color=color, alpha=PIXEL_ALPHA,
			        linewidth=PIXEL_LINEWIDTH, zorder=1)

	# Days with a single valid pixel get mean = that value and std dev = 0
	n_valid = np.isfinite(series).sum(axis=1)
	has_data = n_valid > 0
	xs, s = x[has_data], series[has_data]
	mean = np.nanmean(s, axis=1)
	std = np.where(n_valid[has_data] > 1, np.nanstd(s, axis=1, ddof=1), 0.0)
	# Single dataset: raw pixels keep the dataset colour; mean and std dev get their own
	mean_color = SINGLE_MEAN_COLOR if show_pixels else color
	sd_color = SINGLE_SD_COLOR if show_pixels else color
	ax.fill_between(xs, mean - std, mean + std, color=sd_color, alpha=SD_ALPHA, zorder=2,
	                label="±1 std dev" if show_pixels else None)
	ax.plot(xs, mean, color=mean_color, linewidth=MEAN_LINEWIDTH, zorder=3,
	        label="Mean" if show_pixels else variable)
	return series.shape[1]


def plot_year(ax, year):
	"""All datasets share one y-axis."""
	show_pixels = len(VARIABLES) == 1
	counts = [draw_variable(ax, variable, year, show_pixels) for variable in VARIABLES]
	ax.set_xlabel("Date")
	units = {VARIABLE_UNITS[v] for v in VARIABLES}
	ax.set_ylabel(units.pop() if len(units) == 1 else "mm/day")
	label = ", ".join(f"{v}: {n} px" for v, n in zip(VARIABLES, counts))
	ax.set_title(f"{year} ({label})")
	ax.legend()


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
