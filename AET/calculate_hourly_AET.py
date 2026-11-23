import re
from contextlib import ExitStack
from datetime import date, datetime
from pathlib import Path
from typing import Any
import numpy as np
import rasterio
from netCDF4 import Dataset, num2date
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy.sparse import csr_matrix, hstack
from tqdm import tqdm

from extract_watershed_rasters import make_watershed_mask, read_watershed

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

ETF_VERSION = "v2"
ETF_INPUT_DIR = SCRIPT_DIR / f"Watershed ETf SSEBop-ERA5 Land 30m_gapfilled_{ETF_VERSION}_spatial_{ETF_VERSION}"
ETF_YEAR_FOLDER_PATTERN = "{year}_30m"
ETF_FILE_PATTERN = "watershed_ETf_30m_*.tif"
ETF_NODATA_VALUES = (-99999.0, -9999.0)
ETF_RESOLUTION_LABEL = "30m"

TARGET_RESOLUTION_M = 250
TARGET_CRS = "EPSG:32643"
TARGET_WIDTH = 156
TARGET_HEIGHT = 105
TARGET_XLL = 798778.801047321060
TARGET_YLL = 1484563.296254347777
WATERSHED_FILE = SCRIPT_DIR / "Updated Watershed" / "Updated_watershed.shp"
OUTPUT_30M_DIR = SCRIPT_DIR / f"Watershed AET hourly 30m {ETF_VERSION}"
OUTPUT_TARGET_DIR = SCRIPT_DIR / f"Watershed AET hourly {TARGET_RESOLUTION_M}m {ETF_VERSION}"
OUTPUT_30M_PREFIX = f"watershed_AET_hourly_{ETF_VERSION}"
OUTPUT_TARGET_PREFIX = f"watershed_AET_hourly_{ETF_VERSION}"
OUTPUT_30M_NETCDF = OUTPUT_30M_DIR / f"{OUTPUT_30M_PREFIX}_{ETF_RESOLUTION_LABEL}_all_years.nc"
OUTPUT_TARGET_NETCDF = OUTPUT_TARGET_DIR / f"{OUTPUT_TARGET_PREFIX}_{TARGET_RESOLUTION_M}m_all_years.nc"

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


def read_etf(path: Path) -> tuple[np.ndarray, rasterio.Affine, CRS]:
	with rasterio.open(path) as source:
		if source.crs is None:
			raise ValueError(f"ETf raster has no CRS: {path}")
		data = np.ma.filled(source.read(1, masked=True), np.nan).astype(np.float32)
		for nodata in ETF_NODATA_VALUES:
			data[data == nodata] = np.nan
		data[~np.isfinite(data)] = np.nan
		return data, source.transform, source.crs


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


def pet_resampling_weights(
	source_shape: tuple[int, int], source_transform: rasterio.Affine, source_crs: CRS,
	target_shape: tuple[int, int], target_transform: rasterio.Affine, target_crs: CRS,
) -> tuple[csr_matrix, np.ndarray]:
	weights = []
	coverage = np.zeros(target_shape, dtype=bool)
	for index in range(source_shape[0] * source_shape[1]):
		basis = np.zeros(source_shape, dtype=np.float32)
		basis.flat[index] = 1.0
		warped = align_frame(basis, source_transform, source_crs, target_shape, target_transform, target_crs, Resampling.bilinear)
		if index == 0:
			coverage = np.isfinite(warped)
		weights.append(csr_matrix(np.nan_to_num(warped, nan=0.0).reshape(-1, 1)))
	return csr_matrix(hstack(weights, format="csr")), coverage


def overlap_weights(source_count: int, source_origin: float, source_step: float,
	target_count: int, target_origin: float, target_step: float) -> csr_matrix:
	rows, columns, weights = [], [], []
	for target_index in range(target_count):
		start = target_origin + target_index * target_step
		end = start + target_step
		low, high = sorted((start, end))
		first = max(0, int(np.floor(min((low - source_origin) / source_step, (high - source_origin) / source_step))))
		last = min(source_count, int(np.ceil(max((low - source_origin) / source_step, (high - source_origin) / source_step))))
		for source_index in range(first, last):
			source_start = source_origin + source_index * source_step
			source_end = source_start + source_step
			source_low, source_high = sorted((source_start, source_end))
			overlap = min(high, source_high) - max(low, source_low)
			if overlap > 0:
				rows.append(target_index)
				columns.append(source_index)
				weights.append(overlap)
	return csr_matrix((weights, (rows, columns)), shape=(target_count, source_count), dtype=np.float64)


def aggregation_weights(source_shape: tuple[int, int], source_transform: rasterio.Affine,
	source_crs: CRS, target_transform: rasterio.Affine, target_crs: CRS) -> tuple[csr_matrix, csr_matrix]:
	if source_crs != target_crs or source_transform.b != 0 or source_transform.d != 0 or source_transform.a <= 0 or source_transform.e >= 0:
		raise ValueError("ETf and target grids must share a CRS and use north-up, unrotated pixels")
	rows = overlap_weights(source_shape[0], source_transform.f, source_transform.e, TARGET_HEIGHT, target_transform.f, target_transform.e)
	columns = overlap_weights(source_shape[1], source_transform.c, source_transform.a, TARGET_WIDTH, target_transform.c, target_transform.a)
	return rows, columns


def aggregate_frame(frame: np.ndarray, rows: csr_matrix, columns: csr_matrix) -> np.ndarray:
	valid = np.isfinite(frame)
	totals = (columns @ (rows @ np.where(valid, frame, 0.0)).T).T
	areas = (columns @ (rows @ valid.astype(np.float32)).T).T
	result = np.full(totals.shape, np.nan, dtype=np.float32)
	np.divide(totals, areas, out=result, where=areas > 0)
	return result


def create_grid_variable(
	output: Dataset,
	label: str,
	transform: rasterio.Affine,
	width: int,
	height: int,
	crs: CRS,
) -> Any:
	x_name, y_name, crs_name = f"x_{label}", f"y_{label}", f"crs_{label}"
	output.createDimension(y_name, height)
	output.createDimension(x_name, width)
	x_var = output.createVariable(x_name, "f8", (x_name,))
	y_var = output.createVariable(y_name, "f8", (y_name,))
	x_var[:] = transform.c + (np.arange(width) + 0.5) * transform.a
	y_var[:] = transform.f + (np.arange(height) + 0.5) * transform.e
	x_var.axis = "X"
	y_var.axis = "Y"
	x_var.standard_name = "projection_x_coordinate"
	y_var.standard_name = "projection_y_coordinate"
	x_var.units = "m" if crs.is_projected else "degrees_east"
	y_var.units = "m" if crs.is_projected else "degrees_north"
	crs_var = output.createVariable(crs_name, "i4")
	crs_var.spatial_ref = crs.to_wkt()
	crs_var.crs_wkt = crs.to_wkt()
	data_var = output.createVariable(
		f"aet_{label}", "f4", ("time", y_name, x_name), zlib=True,
		complevel=NETCDF_COMPRESSION_LEVEL, chunksizes=(1, height, width),
		fill_value=NODATA,
	)
	data_var.units = "mm/hr"
	data_var.long_name = f"Hourly actual evapotranspiration at {label}"
	data_var.grid_mapping = crs_name
	return data_var


def create_netcdf(
	path: Path,
	label: str,
	transform: rasterio.Affine,
	width: int,
	height: int,
	crs: CRS,
) -> tuple[Dataset, Any, Any]:
	path.parent.mkdir(parents=True, exist_ok=True)
	output = Dataset(path, "w", format="NETCDF4")
	output.Conventions = "CF-1.8"
	output.title = f"Hourly actual evapotranspiration at {label}"
	output.createDimension("time", None)
	time_var = output.createVariable("time", "f8", ("time",))
	time_var.units = f"hours since {TIME_ORIGIN:%Y-%m-%d %H:%M:%S}"
	time_var.calendar = "gregorian"
	time_var.standard_name = "time"
	data_var = create_grid_variable(output, label, transform, width, height, crs)
	return output, data_var, time_var


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
	print(f"Step 1/3: indexing daily ETf rasters in {ETF_INPUT_DIR.name}")
	etf_by_date = input_etf_files()
	print(f"  {len(etf_by_date)} ETf days found")
	print(f"Step 2/3: listing PET yearly files in {PET_INPUT_DIR.name}")
	pet_files = sorted(PET_INPUT_DIR.glob(PET_FILE_PATTERN))
	if not pet_files:
		raise FileNotFoundError(f"No hourly PET NetCDF files found in {PET_INPUT_DIR}")
	print(f"  {len(pet_files)} PET files found")
	print("Step 3/3: reading watershed and building 250 m target mask")
	geometries, watershed_crs = read_watershed(WATERSHED_FILE)
	target_crs = CRS.from_user_input(TARGET_CRS)
	target_transform = from_origin(TARGET_XLL, TARGET_YLL + TARGET_HEIGHT * TARGET_RESOLUTION_M, TARGET_RESOLUTION_M, TARGET_RESOLUTION_M)
	target_mask = make_watershed_mask((TARGET_HEIGHT, TARGET_WIDTH), target_transform, target_crs, geometries, watershed_crs)
	count = 0
	etf_grid: tuple[rasterio.Affine, CRS, tuple[int, int]] | None = None
	pet_grid = None
	cached_day = None
	previous_time = None
	etf_mask = None
	pet_coverage = None
	with ExitStack() as outputs:
		for pet_path in tqdm(pet_files, desc="PET years", unit="file"):
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
					tqdm.write(f"Skipping {pet_path.name}: no hours with a matching ETf day")
					continue
				year = timestamps[eligible[0]].year
				tqdm.write(f"{pet_path.name}: computing {len(eligible)} hours of {year}")
				for pet_index in tqdm(eligible, desc=f"Hourly AET {year}", unit="hour", leave=False):
					timestamp = timestamps[pet_index]
					if timestamp.minute or timestamp.second or timestamp.microsecond or (previous_time is not None and timestamp <= previous_time):
						raise ValueError(f"PET timestamps must be unique, hourly, and increasing: {timestamp} in {pet_path}")
					etf_path = etf_by_date[timestamp.date()]
					if timestamp.date() != cached_day:
						etf_frame, etf_transform, etf_crs = read_etf(etf_path)
						current_grid = (etf_transform, etf_crs, etf_frame.shape)
						if etf_grid is None:
							etf_grid = current_grid
							etf_mask = make_watershed_mask(etf_frame.shape, etf_transform, etf_crs, geometries, watershed_crs)
							row_weights, column_weights = aggregation_weights(etf_frame.shape, etf_transform, etf_crs, target_transform, target_crs)
							tqdm.write(f"  Creating output NetCDFs: {OUTPUT_30M_NETCDF.name}, {OUTPUT_TARGET_NETCDF.name}")
							out_30_nc, out_30_var, out_30_time = create_netcdf(
								OUTPUT_30M_NETCDF, ETF_RESOLUTION_LABEL, etf_transform, etf_frame.shape[1], etf_frame.shape[0], etf_crs,
							)
							out_target_nc, out_target_var, out_target_time = create_netcdf(
								OUTPUT_TARGET_NETCDF, f"{TARGET_RESOLUTION_M}m", target_transform, TARGET_WIDTH, TARGET_HEIGHT, target_crs,
							)
							outputs.enter_context(out_30_nc)
							outputs.enter_context(out_target_nc)
						elif current_grid != etf_grid:
							raise ValueError(f"ETf rasters do not share one grid: {etf_path}")
						assert etf_mask is not None
						etf_frame[~etf_mask] = np.nan
						cached_day = timestamp.date()
					current_pet_grid = (pet_values.shape[1:], pet_transform, pet_crs)
					if current_pet_grid != pet_grid:
						tqdm.write("  Building PET-to-30 m resampling weights (once per PET grid)")
						pet_weights, pet_coverage = pet_resampling_weights(
							pet_values.shape[1:], pet_transform, pet_crs, etf_frame.shape, etf_transform, etf_crs,
						)
						pet_grid = current_pet_grid
					pet_frame = read_pet_frame(pet_values, pet_index, getattr(pet_variable, "_FillValue", None))
					if np.isfinite(pet_frame).all():
						pet_30m = np.asarray(pet_weights @ pet_frame.ravel()).reshape(etf_frame.shape)
						assert pet_coverage is not None
						pet_30m[~pet_coverage] = np.nan
					else:
						pet_30m = align_frame(pet_frame, pet_transform, pet_crs, etf_frame.shape, etf_transform, etf_crs, Resampling.bilinear)
					with np.errstate(invalid="ignore", over="ignore"):
						aet_30m = (pet_30m * etf_frame).astype(np.float32)
					aet_30m[~np.isfinite(aet_30m)] = np.nan
					aet_target = aggregate_frame(aet_30m, row_weights, column_weights)
					aet_target[~target_mask] = np.nan
					stamp = timestamp.strftime("%Y%m%d%H")
					out_target_dir = OUTPUT_TARGET_DIR / f"{timestamp.year}_{TARGET_RESOLUTION_M}m"
					write_frame(out_target_dir / f"{OUTPUT_TARGET_PREFIX}_{TARGET_RESOLUTION_M}m_{stamp}.tif", aet_target, target_transform, target_crs)
					hours = (timestamp - TIME_ORIGIN).total_seconds() / 3600
					out_30_time[count] = hours
					out_target_time[count] = hours
					out_30_var[count] = np.where(np.isfinite(aet_30m), aet_30m, NODATA)
					out_target_var[count] = np.where(np.isfinite(aet_target), aet_target, NODATA)
					count += 1
					previous_time = timestamp
	return count, count


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	written_30m, written_target = calculate_hourly_aet()
	print(f"Done: {written_30m} hourly time steps written to {OUTPUT_30M_NETCDF.name}")
	print(f"Done: {written_target} hourly {TARGET_RESOLUTION_M} m GeoTIFFs in {OUTPUT_TARGET_DIR.name} and NetCDF {OUTPUT_TARGET_NETCDF.name}")