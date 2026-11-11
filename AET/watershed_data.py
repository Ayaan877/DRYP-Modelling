import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal
import numpy as np
import rasterio
from netCDF4 import Dataset, num2date
from rasterio.crs import CRS
from rasterio.transform import from_origin


NODATA = -99999
DATE_PATTERN = re.compile(r"(\d{8})")


@dataclass(frozen=True)
class DatasetInput:
	kind: Literal["netcdf", "daily_netcdf", "geotiff"]
	folder: str
	pattern: str
	variable: str
	units: str
	cadence_days: int
	resolution: str
	x_variable: str = "lon"
	y_variable: str = "lat"
	source_crs: str | None = None
	scale: float = 1.0
	nodata_values: tuple[float, ...] = (NODATA, -9999)


@dataclass
class DataSeries:
	dates: list[date]
	values: np.ndarray
	transform: rasterio.Affine
	crs: CRS
	resolution_m: float
	units: str
	cadence_days: int


def filled_float(values, nodata_values):
	data = np.ma.filled(values, np.nan).astype(np.float32)
	invalid = ~np.isfinite(data)
	for nodata in nodata_values:
		invalid |= data == nodata
	data[invalid] = np.nan
	return data


def dates_from_time(time_variable):
	calendar = getattr(time_variable, "calendar", "standard")
	values = num2date(
		time_variable[:],
		units=time_variable.units,
		calendar=calendar,
		only_use_cftime_datetimes=False,
	)
	date_values = values.flat if isinstance(values, np.ndarray) else [values]
	return [date(int(value.year), int(value.month), int(value.day)) for value in date_values]


def grid_from_coordinates(x_values, y_values, values):
	x = np.asarray(x_values, dtype=float)
	y = np.asarray(y_values, dtype=float)
	if x.size < 2 or y.size < 2:
		raise ValueError("A spatial grid must have at least two coordinates on each axis")
	dx = float(np.median(np.abs(np.diff(x))))
	dy = float(np.median(np.abs(np.diff(y))))
	if x[0] > x[-1]:
		x = x[::-1]
		values = values[:, :, ::-1]
	if y[0] < y[-1]:
		y = y[::-1]
		values = values[:, ::-1, :]
	return values, from_origin(x[0] - dx / 2, y[0] + dy / 2, dx, dy)


def netcdf_grid(dataset, config):
	values = filled_float(dataset.variables[config.variable][:], config.nodata_values)
	if values.ndim == 2:
		values = values[np.newaxis, :, :]
	values *= config.scale
	values, transform = grid_from_coordinates(
		dataset.variables[config.x_variable][:],
		dataset.variables[config.y_variable][:],
		values,
	)
	crs = CRS.from_user_input(config.source_crs) if config.source_crs else None
	for crs_name in ("crs", "spatial_ref"):
		if crs_name not in dataset.variables:
			continue
		crs_variable = dataset.variables[crs_name]
		wkt = getattr(crs_variable, "crs_wkt", None) or getattr(crs_variable, "spatial_ref", None)
		if wkt:
			crs = CRS.from_wkt(wkt)
			break
	if crs is None:
		raise ValueError(f"No CRS is configured or available in {dataset.filepath()}")
	dates = dates_from_time(dataset.variables["time"])
	resolution_m = resolution_in_m(transform, crs)
	return dates, values, transform, crs, resolution_m


def resolution_in_m(transform, crs):
	x_size, y_size = abs(transform.a), abs(transform.e)
	if crs.is_projected:
		meters_per_unit = crs.linear_units_factor[1]
		return (x_size * y_size) ** 0.5 * meters_per_unit
	latitude = transform.f + transform.e / 2
	return (x_size * 111320 * np.cos(np.deg2rad(latitude)) * y_size * 111320) ** 0.5


def load_netcdf(path, config, year):
	with Dataset(path) as dataset:
		dates, values, transform, crs, resolution_m = netcdf_grid(dataset, config)
	keep = [index for index, value in enumerate(dates) if value.year == year]
	return DataSeries(
		[dates[index] for index in keep],
		values[keep],
		transform,
		crs,
		resolution_m,
		config.units,
		config.cadence_days,
	)


def load_daily_netcdfs(path, config, year):
	files = sorted(path.glob("*.nc"))
	if not files:
		raise FileNotFoundError(f"No NetCDF files found in {path}")
	dates, frames = [], []
	grid = None
	for file_path in files:
		with Dataset(file_path) as dataset:
			file_dates, values, transform, crs, resolution_m = netcdf_grid(dataset, config)
			for index, value in enumerate(file_dates):
				if value.year == year:
					dates.append(value)
					frames.append(values[index])
			grid = transform, crs, resolution_m
	if not dates or grid is None:
		raise FileNotFoundError(f"No {year} observations found in {path}")
	order = np.argsort(dates)
	if len(set(dates)) != len(dates):
		raise ValueError(f"Duplicate dates found in {path} for {year}")
	transform, crs, resolution_m = grid
	return DataSeries(
		[dates[index] for index in order],
		np.stack([frames[index] for index in order]),
		transform,
		crs,
		resolution_m,
		config.units,
		config.cadence_days,
	)


def load_geotiffs(path, config, year):
	year_path = path / f"{year}_{config.resolution}"
	files = sorted(year_path.glob(config.pattern))
	if not files:
		raise FileNotFoundError(f"No rasters found in {year_path}")
	dates, frames = [], []
	transform = None
	crs = None
	resolution_m = 0.0
	for file_path in files:
		match = DATE_PATTERN.search(file_path.stem)
		if not match:
			continue
		with rasterio.open(file_path) as source:
			if source.crs is None:
				raise ValueError(f"Raster has no CRS: {file_path}")
			frame = filled_float(source.read(1), config.nodata_values)
			if source.nodata is not None:
				frame[frame == source.nodata] = np.nan
			frames.append(frame)
			dates.append(date.fromisoformat(f"{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:]}"))
			transform, crs = source.transform, source.crs
			resolution_m = resolution_in_m(transform, crs)
	if not frames or transform is None or crs is None:
		raise FileNotFoundError(f"No dated rasters found in {year_path}")
	return DataSeries(dates, np.stack(frames), transform, crs, resolution_m, config.units, config.cadence_days)


ROOT = Path(__file__).resolve().parent

DATASETS = {
	"AET 750m": DatasetInput("netcdf", "Watershed AET Bhuvan 750m", "watershed_bhuvan_AET_750m_{year}.nc", "aet", "mm/day", 1, "750m", source_crs="EPSG:32643"),
	"AET 5500m": DatasetInput("netcdf", "Watershed AET Bhuvan 5500m", "watershed_bhuvan_AET_5500m_{year}.nc", "aet", "mm/day", 1, "5500m", source_crs="EPSG:32643"),
	"SSEBop 1000m": DatasetInput("netcdf", "Watershed AET SSEBop 1000m", "watershed_ssebop_AET_1000m_{year}.nc", "aet", "mm/day", 10, "1000m", source_crs="EPSG:32643", scale=0.1),
	"SSEBop 30m": DatasetInput("netcdf", "Watershed AET SSEBop 30m", "watershed_ssebop_AET_30m_{year}.nc", "aet", "mm/day", 1, "30m", source_crs="EPSG:32643"),
	"ETf 750m": DatasetInput("geotiff", "Watershed ETf Bhuvan-ERA5 750m", "watershed_ETf_750m_*.tif", "", "1", 1, "750m"),
	"ETf gapfilled 750m": DatasetInput("geotiff", "Watershed ETf Bhuvan-ERA5 750m - gapfilled", "watershed_ETf_750m_*.tif", "", "1", 1, "750m"),
	"ref ET agERA5": DatasetInput("netcdf", "Watershed ref ET agERA5 10km", "watershed_agERA5_refET_10km_{year}.nc", "refET", "mm/day", 1, "10km", source_crs="EPSG:32643"),
	"ref ET ERA5": DatasetInput("netcdf", "Watershed ref ET ERA5 Land 10km", "watershed_ERA5Land_refET_10km_{year}.nc", "refET", "mm/day", 1, "10km", source_crs="EPSG:32643"),
}


def register_etf_datasets():
	aet_configs = [config for config in DATASETS.values() if config.folder.startswith("Watershed AET ")]
	for folder in sorted(ROOT.glob("Watershed ETf *")):
		match = re.fullmatch(
			r"Watershed ETf (.+) (\d+(?:m|km))( - gapfilled|_gapfilled)?(_spatial)?",
			folder.name,
		)
		if not folder.is_dir() or not match:
			continue
		sources, resolution, gapfilled, spatial = match.groups()
		if spatial and not gapfilled:
			continue
		cadence = 1
		for config in aet_configs:
			source = config.folder.removeprefix("Watershed AET ").removesuffix(f" {config.resolution}")
			if config.resolution == resolution and sources.startswith(f"{source}-"):
				cadence = config.cadence_days
				break
		if gapfilled or spatial:
			cadence = 1
		name = (
			f"ETf {sources.replace('-', ' ')}"
			f"{' gapfilled' if gapfilled else ''}"
			f"{' spatial' if spatial else ''} {resolution}"
		)
		DATASETS.setdefault(name, DatasetInput(
			"geotiff", folder.name, f"watershed_ETf_{resolution}_*.tif", "", "1", cadence, resolution,
		))


register_etf_datasets()


def load_data(name, year):
	"""Load a named dataset for one year as a plotting-ready DataSeries."""
	if name not in DATASETS:
		raise KeyError(f"Unknown dataset {name!r}. Available: {', '.join(DATASETS)}")
	config = DATASETS[name]
	folder = ROOT / config.folder
	path = folder / config.pattern.format(year=year)
	if config.kind == "geotiff":
		return load_geotiffs(folder, config, year)
	if config.kind == "daily_netcdf":
		return load_daily_netcdfs(path, config, year)
	return load_netcdf(path, config, year)