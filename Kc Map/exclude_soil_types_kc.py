import numpy as np
import rasterio
from netCDF4 import Dataset
from pathlib import Path
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject


SCRIPT_DIR = Path(__file__).resolve().parent
KC_VERSION = "v3.0"
OUTPUT_VERSION = "v3.1"
INPUT_FILE = SCRIPT_DIR / f"Kc_spatiotemporal_{KC_VERSION}_250m.nc"
OUTPUT_FILE = SCRIPT_DIR / f"Kc_spatiotemporal_{OUTPUT_VERSION}_250m.nc"
SOIL_TEXTURE_FILE = SCRIPT_DIR / "Soil_Texture.tif"

NODATA = -99999
MAJORITY_THRESHOLD = 0.5
KC_VARIABLE = "kc"
LATITUDE_VARIABLE = "lat"
LONGITUDE_VARIABLE = "lon"
CRS_VARIABLE = "crs"
CRS_ATTRIBUTE_NAMES = ("spatial_ref", "crs_wkt", "epsg_code")

SOIL_TEXTURE_CODES = {
	"No Data": 0,
	"Clay": 1,
	"Dyke and Ridges": 2,
	"Gravelly Clay": 3,
	"Habitation Mask": 4,
	"Loamy Sand": 5,
	"Rock Outcrops": 6,
	"Sandy Clay": 7,
	"Sandy Clay Loam": 8,
	"Sandy Loam": 9,
	"Water Body Mask": 10,
}
EXCLUDED_SOIL_TEXTURES = {
	SOIL_TEXTURE_CODES[name]
	for name in (
		"No Data",
		"Rock Outcrops",
		"Habitation Mask",
		"Water Body Mask",
		"Dyke and Ridges",
	)
}


def copy_attributes(source, target):
	for attribute in source.ncattrs():
		if attribute != "_FillValue":
			target.setncattr(attribute, source.getncattr(attribute))


def get_grid_transform(x_coordinates, y_coordinates):
	if len(x_coordinates) < 2 or len(y_coordinates) < 2:
		raise ValueError("Kc coordinates must contain at least two values")

	x_step = float(np.median(np.diff(x_coordinates)))
	y_step = float(np.median(np.diff(y_coordinates)))
	if not np.isclose(np.diff(x_coordinates), x_step).all():
		raise ValueError("Kc X coordinates are not regularly spaced")
	if not np.isclose(np.diff(y_coordinates), y_step).all():
		raise ValueError("Kc Y coordinates are not regularly spaced")
	if x_step <= 0 or y_step >= 0:
		raise ValueError("Kc coordinates must increase in X and decrease in Y")

	return from_origin(
		float(x_coordinates[0] - x_step / 2),
		float(y_coordinates[0] - y_step / 2),
		x_step,
		-y_step,
	)


def get_crs(source):
	crs_variable = source.variables[CRS_VARIABLE]
	for attribute in CRS_ATTRIBUTE_NAMES:
		crs_value = getattr(crs_variable, attribute, None)
		if crs_value is not None:
			return CRS.from_user_input(crs_value)
	raise ValueError("The Kc NetCDF has no usable CRS metadata")


def read_soil_exclusion_mask(soil_texture_file, target_transform, target_crs, target_shape):
	with rasterio.open(soil_texture_file) as soil_source:
		if soil_source.crs is None:
			raise ValueError("The soil texture raster has no CRS information")

		soil_texture = soil_source.read(1)
		excluded_source = np.isin(
			soil_texture,
			list(EXCLUDED_SOIL_TEXTURES),
		).astype(np.float32)
		soil_cell_source = np.ones(soil_texture.shape, dtype=np.float32)
		excluded_count = np.zeros(target_shape, dtype=np.float32)
		soil_cell_count = np.zeros(target_shape, dtype=np.float32)
		reproject(
			source=excluded_source,
			destination=excluded_count,
			src_transform=soil_source.transform,
			src_crs=soil_source.crs,
			dst_transform=target_transform,
			dst_crs=target_crs,
			resampling=Resampling.sum,
			dst_nodata=0,
		)
		reproject(
			source=soil_cell_source,
			destination=soil_cell_count,
			src_transform=soil_source.transform,
			src_crs=soil_source.crs,
			dst_transform=target_transform,
			dst_crs=target_crs,
			resampling=Resampling.sum,
			dst_nodata=0,
		)

	return (
		(soil_cell_count > 0)
		& (excluded_count > MAJORITY_THRESHOLD * soil_cell_count)
	)


def mask_downscaled_kc(input_file, output_file, soil_texture_file):
	with Dataset(input_file, "r") as source:
		kc_source = source.variables[KC_VARIABLE]
		x_coordinates = np.asarray(source.variables["lon"][:], dtype=np.float64)
		y_coordinates = np.asarray(source.variables["lat"][:], dtype=np.float64)
		target_transform = get_grid_transform(x_coordinates, y_coordinates)
		target_crs = get_crs(source)
		soil_mask = read_soil_exclusion_mask(
			soil_texture_file,
			target_transform,
			target_crs,
			(kc_source.shape[1], kc_source.shape[2]),
		)

		with Dataset(output_file, "w", format="NETCDF4") as output:
			for name, dimension in source.dimensions.items():
				output.createDimension(name, len(dimension))

			output.setncatts({
				attribute: source.getncattr(attribute)
				for attribute in source.ncattrs()
			})
			output.history = (
				getattr(source, "history", "")
				+ " | Soil exclusion applied at 250 m using source-cell majority"
			)
			output.soil_exclusion_threshold = MAJORITY_THRESHOLD
			output.soil_exclusion_classes = ", ".join(
				str(code) for code in sorted(EXCLUDED_SOIL_TEXTURES)
			)

			output_variables = {}
			for name, source_variable in source.variables.items():
				fill_value = getattr(source_variable, "_FillValue", None)
				output_variable = output.createVariable(
					name,
					source_variable.dtype,
					source_variable.dimensions,
					fill_value=fill_value,
				)
				copy_attributes(source_variable, output_variable)
				output_variables[name] = output_variable

			for name, source_variable in source.variables.items():
				if name != KC_VARIABLE:
					output_variables[name][:] = source_variable[:]

			kc_output = output_variables[KC_VARIABLE]
			for day_index in range(kc_source.shape[0]):
				source_day = kc_source[day_index, :, :]
				if np.ma.isMaskedArray(source_day):
					source_day = source_day.filled(NODATA)
				daily_kc = np.asarray(source_day, dtype=np.float32)
				daily_kc[soil_mask] = NODATA
				kc_output[day_index, :, :] = daily_kc

	print(f"Wrote {output_file}")
	print(f"Masked 250 m cells: {int(soil_mask.sum())}")


if __name__ == "__main__":
	mask_downscaled_kc(INPUT_FILE, OUTPUT_FILE, SOIL_TEXTURE_FILE)
