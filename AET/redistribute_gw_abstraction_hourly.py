from datetime import datetime
from itertools import groupby
from pathlib import Path
import numpy as np
from netCDF4 import Dataset, num2date
from tqdm import tqdm

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
VERSION = "v2"
INPUT_DIR = SCRIPT_DIR / f"Watershed gw abstraction hourly 250m {VERSION}"
INPUT_NETCDF_FILE = INPUT_DIR / "gw_abstraction_hourly_250m_all_years.nc"
OUTPUT_NETCDF_FILE = INPUT_DIR / "gw_abstraction_hourly_250m_redistributed_all_years.nc"

ABSTRACTION_VARIABLE = "gw_abs"
TIME_VARIABLE = "time"
X_VARIABLE = "x"
Y_VARIABLE = "y"
CRS_VARIABLE = "crs"
COPIED_VARIABLES = (TIME_VARIABLE, X_VARIABLE, Y_VARIABLE, CRS_VARIABLE)
EXPECTED_UNITS = "m3/hr"
LONG_NAME = "Hourly gw abstraction redistributed to pumping hours"
HOURS_PER_DAY = 24
# Hour-beginning labels: 03:00-06:00 and 09:00-13:00 (7 hours in total)
TARGET_HOURS = (3, 4, 5, 9, 10, 11, 12)
NODATA = -9999.0
NETCDF_COMPRESSION_LEVEL = 4

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------


def copy_variable(source: Dataset, destination: Dataset, name: str) -> None:
	source_variable = source.variables[name]
	attributes = {
		attribute: source_variable.getncattr(attribute)
		for attribute in source_variable.ncattrs()
		if attribute != "_FillValue"
	}
	destination_variable = destination.createVariable(
		name, source_variable.datatype, source_variable.dimensions,
	)
	destination_variable.setncatts(attributes)
	destination_variable[...] = source_variable[...]


def read_timestamps(dataset: Dataset, path: Path) -> list[datetime]:
	for name in (ABSTRACTION_VARIABLE, TIME_VARIABLE, X_VARIABLE, Y_VARIABLE):
		if name not in dataset.variables:
			raise ValueError(f"Configured variable {name!r} was not found in {path}")

	abstraction = dataset.variables[ABSTRACTION_VARIABLE]
	if abstraction.dimensions != (TIME_VARIABLE, Y_VARIABLE, X_VARIABLE):
		raise ValueError(f"{ABSTRACTION_VARIABLE!r} must use dimensions (time, y, x) in {path}")
	units = str(getattr(abstraction, "units", ""))
	if units != EXPECTED_UNITS:
		raise ValueError(f"Expected units {EXPECTED_UNITS!r}, found {units!r}")

	time_variable = dataset.variables[TIME_VARIABLE]
	decoded = num2date(
		time_variable[:], units=time_variable.units,
		calendar=getattr(time_variable, "calendar", "standard"),
		only_use_cftime_datetimes=False,
	)
	timestamps = [
		datetime(value.year, value.month, value.day, value.hour)
		for value in np.asarray(decoded, dtype=object).reshape(-1)
	]
	if len(timestamps) != abstraction.shape[0]:
		raise ValueError(f"Time/data length mismatch in {path}")
	return timestamps


def get_day_slices(timestamps: list[datetime]) -> list[slice]:
	slices = []
	for _, group in groupby(range(len(timestamps)), key=lambda i: timestamps[i].date()):
		indices = list(group)
		hours = [timestamps[i].hour for i in indices]
		if hours != list(range(HOURS_PER_DAY)):
			raise ValueError(f"Day {timestamps[indices[0]].date()} does not have 24 consecutive hourly steps")
		slices.append(slice(indices[0], indices[-1] + 1))
	return slices


def redistribute_day(block: np.ndarray) -> np.ndarray:
	"""Spread each pixel's daily total evenly over TARGET_HOURS; block is (24, y, x) with NaN for nodata."""
	target_mask = np.isin(np.arange(HOURS_PER_DAY), TARGET_HOURS)
	per_hour = np.nansum(block, axis=0) / len(TARGET_HOURS)
	redistributed = np.zeros(block.shape, dtype=np.float64)
	redistributed[target_mask] = per_hour
	valid = np.isfinite(block).any(axis=0)
	return np.where(valid[np.newaxis], redistributed, np.nan)


def redistribute_abstraction() -> tuple[int, float, float]:
	"""Redistribute each day's gw abstraction into TARGET_HOURS and return (days, source total, output total)."""
	source_total = 0.0
	output_total = 0.0
	with Dataset(INPUT_NETCDF_FILE, "r") as source:
		timestamps = read_timestamps(source, INPUT_NETCDF_FILE)
		days = get_day_slices(timestamps)
		abstraction = source.variables[ABSTRACTION_VARIABLE]
		height, width = abstraction.shape[1], abstraction.shape[2]

		OUTPUT_NETCDF_FILE.parent.mkdir(parents=True, exist_ok=True)
		with Dataset(OUTPUT_NETCDF_FILE, "w", format="NETCDF4") as destination:
			destination.setncatts({name: source.getncattr(name) for name in source.ncattrs()})
			destination.title = LONG_NAME
			destination.createDimension(TIME_VARIABLE, len(timestamps))
			destination.createDimension(Y_VARIABLE, height)
			destination.createDimension(X_VARIABLE, width)
			for name in COPIED_VARIABLES:
				copy_variable(source, destination, name)

			output_abstraction = destination.createVariable(
				ABSTRACTION_VARIABLE, "f4", (TIME_VARIABLE, Y_VARIABLE, X_VARIABLE),
				zlib=True, complevel=NETCDF_COMPRESSION_LEVEL,
				chunksizes=(1, height, width), fill_value=NODATA,
			)
			output_abstraction.units = EXPECTED_UNITS
			output_abstraction.long_name = LONG_NAME
			output_abstraction.grid_mapping = CRS_VARIABLE

			for day_slice in tqdm(days, desc="Redistributing days", unit="day"):
				block = np.ma.filled(abstraction[day_slice, :, :], np.nan).astype(np.float64)
				redistributed = redistribute_day(block)
				source_total += float(np.nansum(block))
				output_total += float(np.nansum(redistributed))
				output_abstraction[day_slice, :, :] = np.where(
					np.isfinite(redistributed), redistributed, NODATA,
				).astype(np.float32)

		if not np.isclose(source_total, output_total, rtol=1e-9):
			raise ValueError(f"Total mismatch: source {source_total} vs output {output_total}")
	return len(days), source_total, output_total


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	days, source_total, output_total = redistribute_abstraction()
	print(
		f"Redistributed {days} days into {OUTPUT_NETCDF_FILE} "
		f"(source total {source_total:.3f} m3, output total {output_total:.3f} m3)"
	)
