import numpy as np
import pandas as pd
import rasterio
import fiona
from pathlib import Path
from netCDF4 import Dataset
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom
from tqdm import tqdm

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
KHARIF_RASTER = SCRIPT_DIR / "CropMap_Kharif_2024-25_new.tif"
RABI_RASTER = SCRIPT_DIR / "CropMap_Rabi_2024-25_new.tif"
WATERSHED_FILE = SCRIPT_DIR.parent / "Updated Watershed" / "Updated_watershed.shp"
VERSION = "v3.0"
KC_FILE = SCRIPT_DIR / f"classwise_Kc_curves_{VERSION}.csv"
OUTPUT_FILE = SCRIPT_DIR / f"Kc_spatiotemporal_{VERSION}_10m.nc"

TIME_ORIGIN = "2015-01-01 00:00:00"
TIME_CALENDAR = "gregorian"
EXPECTED_CRS = "EPSG:32643"
NODATA = -99999
FALLOW_KC = 0.1

NETCDF_FORMAT = "NETCDF4"
NETCDF_COMPRESSION_LEVEL = 4
NETCDF_CHUNK_SIZE = 256

TIME_COLUMN = "Time"
CLASS_PREFIX = "Class_"
TIME_VARIABLE = "time"
TIME_BOUNDS_VARIABLE = "time_bounds"
LATITUDE_VARIABLE = "lat"
LONGITUDE_VARIABLE = "lon"
KC_VARIABLE = "kc"
CRS_VARIABLE = "crs"

MILLET_CLASS = 2
AVARE_CLASS = 4
INTERCROP_CLASS = 7


# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def read_kc_curves(kc_file):
	kc_data = pd.read_csv(kc_file)
	time = kc_data[TIME_COLUMN].to_numpy(dtype=np.int64)
	class_curves = {
		int(column.removeprefix(CLASS_PREFIX)): kc_data[column].to_numpy(dtype=np.float32)
		for column in kc_data.columns
		if column.startswith(CLASS_PREFIX)
	}

	if not class_curves:
		raise ValueError("No Class_* columns found in the Kc CSV")
	if len(time) == 0:
		raise ValueError("The Kc CSV contains no time records")
	if not np.array_equal(time, np.arange(len(time))):
		raise ValueError("The Kc Time column must be continuous from 0")

	return time, class_curves


def read_crop_maps(kharif_file, rabi_file):
	with rasterio.open(kharif_file) as kharif_source:
		kharif_map = kharif_source.read(1)
		transform = kharif_source.transform
		crs = kharif_source.crs
		width = kharif_source.width
		height = kharif_source.height
		if crs is None or crs.to_string() != EXPECTED_CRS:
			raise ValueError(f"Kharif raster must use {EXPECTED_CRS}; found {crs}")

	with rasterio.open(rabi_file) as rabi_source:
		rabi_map = rabi_source.read(1)
		if (
			rabi_source.width != width
			or rabi_source.height != height
			or rabi_source.transform != transform
			or rabi_source.crs != crs
		):
			raise ValueError("Kharif and Rabi rasters do not share the same grid")

	return kharif_map, rabi_map, transform, crs, width, height


def read_watershed_mask(watershed_file, transform, crs, width, height):
	with fiona.open(watershed_file) as watershed:
		if watershed.crs_wkt:
			source_crs = watershed.crs_wkt
		elif watershed.crs:
			source_crs = watershed.crs
		else:
			raise ValueError("The watershed shapefile has no CRS information")

		geometries = [
			transform_geom(source_crs, crs, feature["geometry"])
			for feature in watershed
		]

	if not geometries:
		raise ValueError("The watershed shapefile contains no geometries")

	return geometry_mask(
		geometries,
		out_shape=(height, width),
		transform=transform,
		invert=True,
	)


def make_class_lookup(class_curves, kharif_map, rabi_map, time_length):
	map_classes = set(np.unique(kharif_map)) | set(np.unique(rabi_map))
	if any(class_value < 0 for class_value in map_classes):
		raise ValueError("Raster class values must be non-negative")

	missing_classes = sorted(
		int(class_value)
		for class_value in map_classes
		if int(class_value) not in class_curves
	)
	if missing_classes:
		raise ValueError(f"No Kc column found for raster classes: {missing_classes}")

	lookup = np.full(
		(max(max(map_classes), max(class_curves)) + 1, time_length),
		np.nan,
		dtype=np.float32,
	)
	for class_value, class_curve in class_curves.items():
		if len(class_curve) != time_length:
			raise ValueError(f"Kc curve for Class_{class_value} has the wrong length")
		lookup[class_value, :] = class_curve
	lookup[0, :] = FALLOW_KC
	return lookup


def overlay_daily_kc(kharif_map, rabi_map, lookup, watershed_mask, day_index):
	daily_kc = np.maximum(lookup[kharif_map, day_index], lookup[rabi_map, day_index])
	intercrop_mask = (
		((kharif_map == MILLET_CLASS) & (rabi_map == AVARE_CLASS))
		| ((kharif_map == AVARE_CLASS) & (rabi_map == MILLET_CLASS))
	)
	if np.any(intercrop_mask):
		daily_kc[intercrop_mask] = lookup[INTERCROP_CLASS, day_index]
	daily_kc[~watershed_mask] = NODATA
	return daily_kc


def get_pixel_coordinates(transform, width, height):
	x_coordinates = transform.c + (np.arange(width) + 0.5) * transform.a
	y_coordinates = transform.f + (np.arange(height) + 0.5) * transform.e
	return x_coordinates, y_coordinates


def create_netcdf_variables(output, time, x_coordinates, y_coordinates):
	height = len(y_coordinates)
	width = len(x_coordinates)
	output.Conventions = "CF-1.8"
	output.title = "Daily Crop Coefficient (Kc)"
	output.history = "Created by make_spatiotemporal_Kc.py"
	output.createDimension(TIME_VARIABLE, len(time))
	output.createDimension(LATITUDE_VARIABLE, height)
	output.createDimension(LONGITUDE_VARIABLE, width)
	output.createDimension("nv", 2)

	time_variable = output.createVariable(TIME_VARIABLE, "i8", (TIME_VARIABLE,))
	time_bounds = output.createVariable(
		TIME_BOUNDS_VARIABLE, "i8", (TIME_VARIABLE, "nv"))
	
	lat_variable = output.createVariable(LATITUDE_VARIABLE, "f4", (LATITUDE_VARIABLE,))
	lon_variable = output.createVariable(LONGITUDE_VARIABLE, "f4", (LONGITUDE_VARIABLE,))
	kc_variable = output.createVariable(
		KC_VARIABLE,
		"f4",
		(TIME_VARIABLE, LATITUDE_VARIABLE, LONGITUDE_VARIABLE),
		zlib=True,
		complevel=NETCDF_COMPRESSION_LEVEL,
		chunksizes=(1, min(NETCDF_CHUNK_SIZE, height), min(NETCDF_CHUNK_SIZE, width)),
		fill_value=NODATA)

	time_variable[:] = time
	time_bounds[:, 0] = time
	time_bounds[:, 1] = time + 1
	lat_variable[:] = y_coordinates.astype(np.float32)
	lon_variable[:] = x_coordinates.astype(np.float32)
	
	return time_variable, time_bounds, lat_variable, lon_variable, kc_variable


def add_netcdf_metadata(output, variables, crs):
	time_variable, time_bounds, lat_variable, lon_variable, kc_variable = variables
	time_variable.units = f"days since {TIME_ORIGIN}"
	time_variable.calendar = TIME_CALENDAR
	time_variable.standard_name = "time"
	time_variable.long_name = "time"
	time_variable.axis = "T"
	time_variable.bounds = TIME_BOUNDS_VARIABLE
	time_bounds.units = time_variable.units
	time_bounds.calendar = TIME_CALENDAR
	lat_variable.units = "m"
	lon_variable.units = "m"
	lat_variable.long_name = "UTM northing pixel center"
	lon_variable.long_name = "UTM easting pixel center"
	lat_variable.standard_name = "projection_y_coordinate"
	lon_variable.standard_name = "projection_x_coordinate"
	lat_variable.axis = "Y"
	lon_variable.axis = "X"
	kc_variable.long_name = "daily crop coefficient"
	kc_variable.units = "1"
	kc_variable.coordinates = f"{LATITUDE_VARIABLE} {LONGITUDE_VARIABLE}"
	kc_variable.grid_mapping = CRS_VARIABLE

	crs_variable = output.createVariable(CRS_VARIABLE, "i4")
	crs_variable.long_name = "CRS definition"
	crs_variable.grid_mapping_name = "universal_transverse_mercator"
	crs_variable.epsg_code = EXPECTED_CRS
	crs_variable.spatial_ref = crs.to_wkt()
	crs_variable.crs_wkt = crs.to_wkt()



def write_spatiotemporal_kc(output_file, time, class_curves, kharif_map, rabi_map, 
							watershed_mask, transform, crs, width, height):
	lookup = make_class_lookup(class_curves, kharif_map, rabi_map, len(time))
	x_coordinates, y_coordinates = get_pixel_coordinates(transform, width, height)

	with Dataset(output_file, "w", format=NETCDF_FORMAT) as output:
		variables = create_netcdf_variables(output, time, x_coordinates, y_coordinates)
		add_netcdf_metadata(output, variables, crs)
		kc_variable = variables[-1]
		for day_index in tqdm(range(len(time)), desc="Writing Kc rasters", unit="day"):
			kc_variable[day_index, :, :] = overlay_daily_kc(
				kharif_map, rabi_map, lookup, watershed_mask, day_index)

# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

def run_spatiotemporal_kc():
	time, class_curves = read_kc_curves(KC_FILE)
	kharif_map, rabi_map, transform, crs, width, height = read_crop_maps(
		KHARIF_RASTER, RABI_RASTER)
	watershed_mask = read_watershed_mask(
		WATERSHED_FILE, transform, crs, width, height)
	
	write_spatiotemporal_kc(
		OUTPUT_FILE,
		time,
		class_curves,
		kharif_map,
		rabi_map,
		watershed_mask,
		transform,
		crs,
		width,
		height)
	
	print(f"Wrote {OUTPUT_FILE}: {len(time)} x {height} x {width}")


if __name__ == "__main__":
	run_spatiotemporal_kc()
