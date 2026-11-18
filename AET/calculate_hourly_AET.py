import re
from datetime import date, datetime
from pathlib import Path
from typing import Any
import numpy as np
import rasterio
from netCDF4 import Dataset, num2date
from rasterio.crs import CRS
from rasterio.transform import array_bounds, from_origin
from rasterio.warp import Resampling, reproject
from tqdm import tqdm

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PET_INPUT_DIR = SCRIPT_DIR / "PET_hourly_converted"
PET_FILE_PATTERN = "*_hourly_converted_pet_*.nc"
PET_VARIABLE = "pet"
PET_TIME_VARIABLE = "time"
PET_X_VARIABLE = "longitude"
PET_Y_VARIABLE = "latitude"
PET_SOURCE_CRS = "EPSG:4326"

ETF_INPUT_DIR = SCRIPT_DIR / "Watershed ETf SSEBop-ERA5 Land 30m_gapfilled_spatial"
ETF_YEAR_FOLDER_PATTERN = "{year}_30m"
ETF_FILE_PATTERN = "watershed_ETf_30m_*.tif"
ETF_NODATA_VALUES = (-99999.0, -9999.0)
ETF_RESOLUTION_LABEL = "30m"

TARGET_RESOLUTION_M = 250
TARGET_CRS: str | None = "EPSG:32643"
OUTPUT_30M_DIR = SCRIPT_DIR / "Watershed AET hourly 30m"
OUTPUT_TARGET_DIR = SCRIPT_DIR / f"Watershed AET hourly {TARGET_RESOLUTION_M}m"
OUTPUT_30M_PREFIX = "watershed_AET_hourly"
OUTPUT_TARGET_PREFIX = "watershed_AET_hourly"

NODATA = -99999.0
NETCDF_COMPRESSION_LEVEL = 4
DATE_PATTERN = re.compile(r"_(\d{8})\.tif$", re.IGNORECASE)
TIME_ORIGIN = datetime(2015, 1, 1)

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------


def input_etf_files() -> dict[date, Path]:
	"""Index daily ETf rasters by acquisition date."""
	files: dict[date, Path] = {}
	for year_folder in sorted(ETF_INPUT_DIR.glob(ETF_YEAR_FOLDER_PATTERN.format(year="*"))):
		for path in sorted(year_folder.glob(ETF_FILE_PATTERN)):
			match = DATE_PATTERN.search(path.name)
			if match is None:
				continue
			text = match.group(1)
			day = date(int(text[:4]), int(text[4:6]), int(text[6:8]))
			if day in files:
				raise ValueError(f"Duplicate ETf date {day}: {files[day]} and {path}")
			files[day] = path
	if not files:
		raise FileNotFoundError(f"No daily ETf rasters found in {ETF_INPUT_DIR}")
	return files


def netcdf_grid(
	dataset: Dataset, values: Any
) -> tuple[np.ndarray, rasterio.Affine, CRS, str, str]:
	"""Return time-first data normalized to north-up row/column order."""
	if PET_X_VARIABLE not in dataset.variables or PET_Y_VARIABLE not in dataset.variables:
		raise ValueError("Configured PET X/Y coordinate variables were not found")
	x_variable = dataset.variables[PET_X_VARIABLE]
	y_variable = dataset.variables[PET_Y_VARIABLE]
	data_variable = dataset.variables[PET_VARIABLE]
	if len(x_variable.dimensions) != 1 or len(y_variable.dimensions) != 1:
		raise ValueError("PET X/Y coordinates must be one-dimensional")
	x_dimension = x_variable.dimensions[0]
	y_dimension = y_variable.dimensions[0]
	if PET_TIME_VARIABLE not in data_variable.dimensions:
		raise ValueError(f"{PET_VARIABLE!r} must include the configured time dimension")
	if x_dimension not in data_variable.dimensions or y_dimension not in data_variable.dimensions:
		raise ValueError("PET data variable does not use the configured X/Y dimensions")
	ordered = np.ma.filled(values, np.nan).astype(np.float32)
	ordered = np.transpose(
		ordered,
		(
			data_variable.dimensions.index(PET_TIME_VARIABLE),
			data_variable.dimensions.index(y_dimension),
			data_variable.dimensions.index(x_dimension),
		),
	)
	x_values = np.asarray(x_variable[:], dtype=np.float64)
	y_values = np.asarray(y_variable[:], dtype=np.float64)
	if x_values.size < 2 or y_values.size < 2:
		raise ValueError("PET grid requires at least two coordinates on each axis")
	x_steps = np.diff(x_values)
	y_steps = np.diff(y_values)
	dx = float(np.median(np.abs(x_steps)))
	dy = float(np.median(np.abs(y_steps)))
	if dx == 0 or dy == 0 or not np.allclose(np.abs(x_steps), dx) or not np.allclose(np.abs(y_steps), dy):
		raise ValueError("PET coordinates must be regularly spaced and monotonic")
	if x_values[0] > x_values[-1]:
		ordered = ordered[:, :, ::-1]
	if y_values[0] < y_values[-1]:
		ordered = ordered[:, ::-1, :]
	transform = from_origin(float(np.min(x_values)) - dx / 2, float(np.max(y_values)) + dy / 2, dx, dy)
	return ordered, transform, CRS.from_user_input(PET_SOURCE_CRS), y_dimension, x_dimension


def pet_times(dataset: Dataset) -> list[datetime]:
	time_variable = dataset.variables[PET_TIME_VARIABLE]
	units = getattr(time_variable, "units", None)
	if not units:
		raise ValueError(f"PET time variable {PET_TIME_VARIABLE!r} has no units")
	calendar = getattr(time_variable, "calendar", "standard")
	decoded = num2date(
		time_variable[:], units=units, calendar=calendar,
		only_use_cftime_datetimes=False,
	)
	return [datetime(value.year, value.month, value.day, value.hour, value.minute, value.second)
		for value in np.asarray(decoded, dtype=object).reshape(-1)]


def read_etf(path: Path) -> tuple[np.ndarray, rasterio.Affine, CRS, dict[str, Any]]:
	with rasterio.open(path) as source:
		if source.crs is None:
			raise ValueError(f"ETf raster has no CRS: {path}")
		data = np.ma.filled(source.read(1, masked=True), np.nan).astype(np.float32)
		for nodata in ETF_NODATA_VALUES:
			data[data == nodata] = np.nan
		data[~np.isfinite(data)] = np.nan
		return data, source.transform, source.crs, source.profile.copy()


def read_pet_frame(values: np.ndarray, index: int, nodata: float | None) -> np.ndarray:
	frame = values[index].copy()
	if nodata is not None:
		frame[frame == nodata] = np.nan
	frame[~np.isfinite(frame)] = np.nan
	return frame


def align_frame(
	frame: np.ndarray,
	source_transform: rasterio.Affine,
	source_crs: CRS,
	target_shape: tuple[int, int],
	target_transform: rasterio.Affine,
	target_crs: CRS,
	resampling: Resampling,
) -> np.ndarray:
	aligned = np.full(target_shape, np.nan, dtype=np.float32)
	reproject(
		frame,
		aligned,
		src_transform=source_transform,
		src_crs=source_crs,
		src_nodata=np.nan,
		dst_transform=target_transform,
		dst_crs=target_crs,
		dst_nodata=np.nan,
		resampling=resampling,
	)
	return aligned


def target_grid(
	pet_transform: rasterio.Affine,
	pet_width: int,
	pet_height: int,
	pet_crs: CRS,
) -> tuple[rasterio.Affine, int, int, CRS]:
	"""Keep the PET grid when already at target size; otherwise align a new grid."""
	target_crs = CRS.from_user_input(TARGET_CRS) if TARGET_CRS else pet_crs
	if target_crs != pet_crs:
		bounds = array_bounds(pet_height, pet_width, pet_transform)
		from rasterio.warp import calculate_default_transform
		transform, width, height = calculate_default_transform(
			pet_crs, target_crs, pet_width, pet_height, *bounds,
			resolution=TARGET_RESOLUTION_M,
		)
		if width is None or height is None:
			raise ValueError("Could not determine the transformed PET grid dimensions")
		return transform, width, height, target_crs
	if not pet_crs.is_projected:
		lat = pet_transform.f - abs(pet_transform.e) * pet_height / 2
		meters_per_x = abs(pet_transform.a) * 111320 * np.cos(np.deg2rad(lat))
		meters_per_y = abs(pet_transform.e) * 111320
		pet_resolution = float(np.sqrt(meters_per_x * meters_per_y))
	else:
		pet_resolution = float(np.sqrt(abs(pet_transform.a * pet_transform.e)) * pet_crs.linear_units_factor[1])
	if np.isclose(pet_resolution, TARGET_RESOLUTION_M, rtol=0.1):
		return pet_transform, pet_width, pet_height, pet_crs
	if not pet_crs.is_projected:
		raise ValueError(
			"A different target resolution requires a projected TARGET_CRS; "
		"otherwise the native PET grid is retained."
		)
	meters_per_unit = pet_crs.linear_units_factor[1]
	resolution = TARGET_RESOLUTION_M / meters_per_unit
	left, bottom, right, top = array_bounds(pet_height, pet_width, pet_transform)
	origin_x, origin_y = pet_transform.c, pet_transform.f
	left = origin_x + np.floor((left - origin_x) / resolution) * resolution
	right = origin_x + np.ceil((right - origin_x) / resolution) * resolution
	top = origin_y - np.floor((origin_y - top) / resolution) * resolution
	bottom = origin_y - np.ceil((origin_y - bottom) / resolution) * resolution
	width = int(round((right - left) / resolution))
	height = int(round((top - bottom) / resolution))
	return from_origin(left, top, resolution, resolution), width, height, pet_crs


def create_year_netcdf(
	path: Path,
	variable_name: str,
	long_name: str,
	year_times: list[datetime],
	transform: rasterio.Affine,
	width: int,
	height: int,
	crs: CRS,
) -> tuple[Dataset, Any]:
	path.parent.mkdir(parents=True, exist_ok=True)
	output = Dataset(path, "w", format="NETCDF4")
	output.Conventions = "CF-1.8"
	output.title = long_name
	output.createDimension("time", len(year_times))
	output.createDimension("y", height)
	output.createDimension("x", width)
	time_var = output.createVariable("time", "i8", ("time",))
	time_var.units = f"hours since {TIME_ORIGIN:%Y-%m-%d %H:%M:%S}"
	time_var.calendar = "gregorian"
	time_var.standard_name = "time"
	time_var[:] = [(value - TIME_ORIGIN).total_seconds() / 3600 for value in year_times]
	x_var = output.createVariable("x", "f4", ("x",))
	y_var = output.createVariable("y", "f4", ("y",))
	x_var[:] = transform.c + (np.arange(width) + 0.5) * transform.a
	y_var[:] = transform.f + (np.arange(height) + 0.5) * transform.e
	x_var.axis = "X"
	y_var.axis = "Y"
	x_var.standard_name = "projection_x_coordinate"
	y_var.standard_name = "projection_y_coordinate"
	x_var.units = "m" if crs.is_projected else "degrees_east"
	y_var.units = "m" if crs.is_projected else "degrees_north"
	crs_var = output.createVariable("crs", "i4")
	crs_var.spatial_ref = crs.to_wkt()
	crs_var.crs_wkt = crs.to_wkt()
	data_var = output.createVariable(
		variable_name, "f4", ("time", "y", "x"), zlib=True,
		complevel=NETCDF_COMPRESSION_LEVEL, chunksizes=(1, height, width),
		fill_value=NODATA,
	)
	data_var.units = "mm/hr"
	data_var.long_name = long_name
	data_var.grid_mapping = "crs"
	return output, data_var


def write_frame(path: Path, frame: np.ndarray, transform: rasterio.Affine, crs: CRS) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	profile = {
		"driver": "GTiff", "dtype": "float32", "count": 1,
		"height": frame.shape[0], "width": frame.shape[1],
		"crs": crs, "transform": transform, "nodata": NODATA,
		"compress": "deflate",
	}
	with rasterio.open(path, "w", **profile) as destination:
		destination.write(np.where(np.isfinite(frame), frame, NODATA).astype(np.float32), 1)
		destination.update_tags(units="mm/hr", long_name="hourly actual evapotranspiration")


def calculate_hourly_aet() -> tuple[int, int]:
	"""Multiply every hourly PET slice by its matching daily spatially-filled ETf."""
	etf_by_date = input_etf_files()
	pet_files = sorted(PET_INPUT_DIR.glob(PET_FILE_PATTERN))
	if not pet_files:
		raise FileNotFoundError(f"No hourly PET NetCDF files found in {PET_INPUT_DIR}")
	count_30m = 0
	count_target = 0
	etf_grid: tuple[rasterio.Affine, CRS, tuple[int, int]] | None = None

	for pet_path in tqdm(pet_files, desc="PET yearly files", unit="file"):
		with Dataset(pet_path, "r") as pet_dataset:
			if PET_VARIABLE not in pet_dataset.variables or PET_TIME_VARIABLE not in pet_dataset.variables:
				raise ValueError(f"Configured PET/time variable missing from {pet_path}")
			pet_variable = pet_dataset.variables[PET_VARIABLE]
			pet_values, pet_transform, pet_crs, _, _ = netcdf_grid(pet_dataset, pet_variable[:])
			timestamps = pet_times(pet_dataset)
			if len(timestamps) != pet_values.shape[0]:
				raise ValueError(f"PET time/data length mismatch in {pet_path}")
			eligible = [index for index, timestamp in enumerate(timestamps) if timestamp.date() in etf_by_date]
			if not eligible:
				continue
			selected_times = [timestamps[index] for index in eligible]
			year = selected_times[0].year
			if any(value.year != year for value in selected_times):
				raise ValueError(f"PET file contains matching timestamps from multiple years: {pet_path}")
			first_etf, etf_transform, etf_crs, etf_profile = read_etf(etf_by_date[selected_times[0].date()])
			current_grid = (etf_transform, etf_crs, first_etf.shape)
			if etf_grid is None:
				etf_grid = current_grid
			elif current_grid != etf_grid:
				raise ValueError(f"ETf rasters do not share one grid: {etf_by_date[selected_times[0].date()]}")
			pet_target_transform, pet_target_width, pet_target_height, pet_target_crs = target_grid(
				pet_transform, pet_values.shape[2], pet_values.shape[1], pet_crs
			)
			out_30_dir = OUTPUT_30M_DIR / f"{year}_{ETF_RESOLUTION_LABEL}"
			out_target_dir = OUTPUT_TARGET_DIR / f"{year}_{TARGET_RESOLUTION_M}m"
			out_30_nc, out_30_var = create_year_netcdf(
				OUTPUT_30M_DIR / f"{OUTPUT_30M_PREFIX}_{ETF_RESOLUTION_LABEL}_{year}.nc",
				"aet", "Hourly actual evapotranspiration at 30 m", selected_times,
				etf_transform, first_etf.shape[1], first_etf.shape[0], etf_crs,
			)
			out_target_nc, out_target_var = create_year_netcdf(
				OUTPUT_TARGET_DIR / f"{OUTPUT_TARGET_PREFIX}_{TARGET_RESOLUTION_M}m_{year}.nc",
				"aet", f"Hourly actual evapotranspiration at {TARGET_RESOLUTION_M} m",
				selected_times, pet_target_transform, pet_target_width, pet_target_height, pet_target_crs,
			)
			try:
				for output_index, pet_index in enumerate(tqdm(eligible, desc=f"AET {year}", unit="hour", leave=False)):
					timestamp = timestamps[pet_index]
					etf_path = etf_by_date[timestamp.date()]
					if timestamp.date() == selected_times[0].date():
						etf_frame = first_etf
					else:
						etf_frame, current_transform, current_crs, _ = read_etf(etf_path)
						if current_transform != etf_transform or current_crs != etf_crs or etf_frame.shape != first_etf.shape:
							raise ValueError(f"ETf rasters do not share one grid: {etf_path}")
					pet_frame = read_pet_frame(pet_values, pet_index, getattr(pet_variable, "_FillValue", None))
					pet_30m = align_frame(
						pet_frame, pet_transform, pet_crs, first_etf.shape,
						etf_transform, etf_crs, Resampling.bilinear,
					)
					with np.errstate(invalid="ignore", over="ignore"):
						aet_30m = (pet_30m * etf_frame).astype(np.float32)
					aet_30m[~np.isfinite(aet_30m)] = np.nan
					stamp = timestamp.strftime("%Y%m%d%H")
					write_frame(out_30_dir / f"{OUTPUT_30M_PREFIX}_{ETF_RESOLUTION_LABEL}_{stamp}.tif", aet_30m, etf_transform, etf_crs)
					out_30_var[output_index] = np.where(np.isfinite(aet_30m), aet_30m, NODATA)
					count_30m += 1
					aet_target = align_frame(
						aet_30m, etf_transform, etf_crs,
						(pet_target_height, pet_target_width), pet_target_transform,
						pet_target_crs, Resampling.average,
					)
					write_frame(
						out_target_dir / f"{OUTPUT_TARGET_PREFIX}_{TARGET_RESOLUTION_M}m_{stamp}.tif",
						aet_target, pet_target_transform, pet_target_crs,
					)
					out_target_var[output_index] = np.where(np.isfinite(aet_target), aet_target, NODATA)
					count_target += 1
			finally:
				out_30_nc.close()
				out_target_nc.close()
	return count_30m, count_target


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	written_30m, written_target = calculate_hourly_aet()
	print(f"Wrote {written_30m} hourly 30 m AET frames to {OUTPUT_30M_DIR}")
	print(f"Wrote {written_target} hourly {TARGET_RESOLUTION_M} m AET frames to {OUTPUT_TARGET_DIR}")