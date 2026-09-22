import numpy as np
from netCDF4 import Dataset
from pathlib import Path
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
KC_VERSION = "v3.0"
INPUT_FILE = SCRIPT_DIR / f"Kc_spatiotemporal_{KC_VERSION}_10m.nc"
OUTPUT_FILE = SCRIPT_DIR / f"Kc_spatiotemporal_{KC_VERSION}_250m.nc"

TARGET_CRS = "EPSG:32643"
TARGET_NCOLS = 156
TARGET_NROWS = 105
TARGET_XLLCORNER = 798778.801047321060
TARGET_YLLCORNER = 1484563.296254347777
TARGET_CELLSIZE = 250.0
NODATA = -99999

TIME_VARIABLE = "time"
LATITUDE_VARIABLE = "lat"
LONGITUDE_VARIABLE = "lon"
KC_VARIABLE = "kc"
CRS_VARIABLE = "crs"
TIME_BOUNDS_VARIABLE = "time_bounds"
CRS_ATTRIBUTE_NAMES = ("spatial_ref", "crs_wkt", "epsg_code")
NETCDF_FORMAT = "NETCDF4"
NETCDF_COMPRESSION_LEVEL = 4

TARGET_CRS_OBJECT = CRS.from_string(TARGET_CRS)
TARGET_TRANSFORM = from_origin(
	TARGET_XLLCORNER,
	TARGET_YLLCORNER + TARGET_NROWS * TARGET_CELLSIZE,
	TARGET_CELLSIZE,
	TARGET_CELLSIZE,
)


def copy_attributes(source, target):
	for attribute in source.ncattrs():
		if attribute != "_FillValue":
			target.setncattr(attribute, source.getncattr(attribute))


def get_source_transform(x_coordinates, y_coordinates):
	if len(x_coordinates) < 2 or len(y_coordinates) < 2:
		raise ValueError("Source coordinates must contain at least two values")

	x_step = float(np.median(np.diff(x_coordinates)))
	y_step = float(np.median(np.diff(y_coordinates)))
	if not np.isclose(np.diff(x_coordinates), x_step).all():
		raise ValueError("Source X coordinates are not regularly spaced")
	if not np.isclose(np.diff(y_coordinates), y_step).all():
		raise ValueError("Source Y coordinates are not regularly spaced")
	if x_step <= 0 or y_step >= 0:
		raise ValueError("Source coordinates must increase in X and decrease in Y")

	return from_origin(
		float(x_coordinates[0] - x_step / 2),
		float(y_coordinates[0] - y_step / 2),
		x_step,
		-y_step,
	)


def regrid(source, source_transform, source_crs):
	target = np.full((TARGET_NROWS, TARGET_NCOLS), NODATA, dtype=np.float32)
	reproject(
		source=np.asarray(source, dtype=np.float32),
		destination=target,
		src_transform=source_transform,
		src_crs=source_crs,
		src_nodata=NODATA,
		dst_transform=TARGET_TRANSFORM,
		dst_crs=TARGET_CRS_OBJECT,
		dst_nodata=NODATA,
		resampling=Resampling.average,
	)
	return target


def get_crs(source):
	crs_variable = source.variables[CRS_VARIABLE]
	for attribute in CRS_ATTRIBUTE_NAMES:
		crs = getattr(crs_variable, attribute, None)
		if crs is not None:
			return crs
	raise ValueError("The Kc NetCDF has no usable CRS metadata")


def downscale_kc_netcdf(input_file, output_file):
	with Dataset(input_file, "r") as source:
		time_source = source.variables[TIME_VARIABLE]
		lat_source = source.variables[LATITUDE_VARIABLE]
		lon_source = source.variables[LONGITUDE_VARIABLE]
		kc_source = source.variables[KC_VARIABLE]
		time_count = kc_source.shape[0]
		source_transform = get_source_transform(
			np.asarray(lon_source[:], dtype=np.float64),
			np.asarray(lat_source[:], dtype=np.float64),
		)
		source_crs = get_crs(source)

		with Dataset(output_file, "w", format=NETCDF_FORMAT) as output:
			for attribute in source.ncattrs():
				output.setncattr(attribute, source.getncattr(attribute))
			output.history = (
				getattr(source, "history", "")
				+ " | Regridded to the fixed target grid with average resampling"
			)
			output.createDimension(TIME_VARIABLE, len(source.dimensions[TIME_VARIABLE]))
			output.createDimension(LATITUDE_VARIABLE, TARGET_NROWS)
			output.createDimension(LONGITUDE_VARIABLE, TARGET_NCOLS)
			if "nv" in source.dimensions:
				output.createDimension("nv", len(source.dimensions["nv"]))

			time_output = output.createVariable(TIME_VARIABLE, time_source.dtype, (TIME_VARIABLE,))
			lat_output = output.createVariable(LATITUDE_VARIABLE, "f4", (LATITUDE_VARIABLE,))
			lon_output = output.createVariable(LONGITUDE_VARIABLE, "f4", (LONGITUDE_VARIABLE,))
			kc_output = output.createVariable(
				KC_VARIABLE,
				"f4",
				(TIME_VARIABLE, LATITUDE_VARIABLE, LONGITUDE_VARIABLE),
				zlib=True,
				complevel=NETCDF_COMPRESSION_LEVEL,
				chunksizes=(1, min(256, TARGET_NROWS), min(256, TARGET_NCOLS)),
				fill_value=NODATA,
			)
			copy_attributes(time_source, time_output)
			copy_attributes(lat_source, lat_output)
			copy_attributes(lon_source, lon_output)
			copy_attributes(kc_source, kc_output)
			time_output[:] = time_source[:]
			lat_output[:] = TARGET_TRANSFORM.f + (np.arange(TARGET_NROWS) + 0.5) * TARGET_TRANSFORM.e
			lon_output[:] = TARGET_TRANSFORM.c + (np.arange(TARGET_NCOLS) + 0.5) * TARGET_TRANSFORM.a
			kc_output.cell_methods = "area: mean"

			if TIME_BOUNDS_VARIABLE in source.variables and "nv" in output.dimensions:
				bounds_source = source.variables[TIME_BOUNDS_VARIABLE]
				bounds_output = output.createVariable(
					TIME_BOUNDS_VARIABLE, bounds_source.dtype, (TIME_VARIABLE, "nv")
				)
				copy_attributes(bounds_source, bounds_output)
				bounds_output[:] = bounds_source[:]
				time_output.bounds = TIME_BOUNDS_VARIABLE

			crs_output = output.createVariable(CRS_VARIABLE, "i4")
			copy_attributes(source.variables[CRS_VARIABLE], crs_output)
			crs_output.spatial_ref = TARGET_CRS_OBJECT.to_wkt()
			crs_output.crs_wkt = TARGET_CRS_OBJECT.to_wkt()
			crs_output.epsg_code = f"EPSG:{TARGET_CRS_OBJECT.to_epsg()}"
			kc_output.grid_mapping = CRS_VARIABLE

			for day_index in tqdm(range(time_count), desc="Regridding Kc", unit="day"):
				source_day = kc_source[day_index, :, :]
				if np.ma.isMaskedArray(source_day):
					source_day = source_day.filled(NODATA)
				source_day = np.asarray(source_day, dtype=np.float32)
				source_day[np.isnan(source_day)] = NODATA
				kc_output[day_index, :, :] = regrid(source_day, source_transform, source_crs)

	print(f"Wrote {output_file}: {time_count} x {TARGET_NROWS} x {TARGET_NCOLS}")


if __name__ == "__main__":
	downscale_kc_netcdf(INPUT_FILE, OUTPUT_FILE)