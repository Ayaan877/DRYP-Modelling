import numpy as np
from pathlib import Path
import pandas as pd
import fiona
import rasterio
from rasterio.features import geometry_mask

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

VERSION = "v2.0"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "Rooting Depth FAO56.csv"
DEFAULT_KHARIF = SCRIPT_DIR / "CropMap_Kharif_2024-25_new.tif"
DEFAULT_RABI = SCRIPT_DIR / "CropMap_Rabi_2024-25_new.tif"
DEFAULT_WATERSHED = SCRIPT_DIR / "Updated Watershed" / "Updated_watershed.shp"
DEFAULT_OUTPUT = SCRIPT_DIR / f"Rooting_Depth_{VERSION}_10m.asc"
NODATA_VALUE = -99999


# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def read_depth_lookup(csv_path: Path) -> dict[int, float]:
	data = pd.read_csv(csv_path)
	required_columns = {"Class", "Rooting Depth max (m)"}
	missing_columns = required_columns.difference(data.columns)
	if missing_columns:
		raise ValueError(
			f"{csv_path} is missing required columns: {sorted(missing_columns)}"
		)

	data = data.dropna(subset=["Class", "Rooting Depth max (m)"])
	data["Class"] = pd.to_numeric(data["Class"], errors="raise").astype(int)
	data["Rooting Depth max (m)"] = pd.to_numeric(
		data["Rooting Depth max (m)"], errors="raise"
	)
	if (data["Class"] < 0).any():
		raise ValueError("CSV class values must be non-negative")
	if (data["Rooting Depth max (m)"] < 0).any():
		raise ValueError("Rooting depths must be non-negative")

	grouped_depths = data.groupby("Class")["Rooting Depth max (m)"].max()
	lookup: dict[int, float] = {}
	for class_value, depth in grouped_depths.items():
		if not isinstance(class_value, (int, np.integer)):
			raise TypeError(f"Crop class must be an integer, got {class_value!r}")
		lookup[int(class_value)] = float(depth)
	return lookup


def read_crop_maps(kharif_path: Path, rabi_path: Path):
	with rasterio.open(kharif_path) as kharif_source:
		kharif = kharif_source.read(1, masked=True)
		profile = kharif_source.profile.copy()

	with rasterio.open(rabi_path) as rabi_source:
		rabi = rabi_source.read(1, masked=True)
		if (
			rabi_source.shape != (profile["height"], profile["width"])
			or rabi_source.transform != profile["transform"]
			or rabi_source.crs != profile["crs"]
		):
			raise ValueError("Kharif and Rabi rasters do not share the same grid")

	return kharif, rabi, profile


def read_watershed_mask(watershed_path: Path, profile: dict) -> np.ndarray:
	with fiona.open(watershed_path) as watershed_source:
		if watershed_source.crs_wkt != profile["crs"].to_wkt():
			raise ValueError("Watershed and crop rasters do not share the same CRS")
		geometries = [feature["geometry"] for feature in watershed_source]

	if not geometries:
		raise ValueError(f"Watershed boundary contains no geometries: {watershed_path}")

	return geometry_mask(
		geometries,
		out_shape=(profile["height"], profile["width"]),
		transform=profile["transform"],
		invert=True,
	)


def class_depth_grid(crop_map, depth_lookup: dict[int, float], map_name: str):
	class_values = np.asarray(crop_map.data)
	valid_values = class_values[~np.ma.getmaskarray(crop_map)]
	positive_classes = set(np.unique(valid_values)) - {0}
	unknown_classes = sorted(
		int(class_value)
		for class_value in positive_classes
		if int(class_value) not in depth_lookup
	)
	if unknown_classes:
		raise ValueError(
		f"{map_name} contains crop classes missing from the CSV: {unknown_classes}"
		)

	lookup_size = max(depth_lookup, default=0) + 1
	lookup = np.zeros(lookup_size, dtype=np.float32)
	for class_value, depth in depth_lookup.items():
		lookup[class_value] = depth

	depth_grid = np.zeros(class_values.shape, dtype=np.float32)
	valid_mask = ~np.ma.getmaskarray(crop_map)
	depth_grid[valid_mask] = lookup[class_values[valid_mask]]
	return depth_grid


def write_ascii_grid(output_path: Path, grid: np.ndarray, profile: dict):
	transform = profile["transform"]
	if not np.isclose(transform.b, 0) or not np.isclose(transform.d, 0):
		raise ValueError("Rotated rasters cannot be written as ESRI ASCII grids")
	if not np.isclose(transform.a, -transform.e):
		raise ValueError("Raster cells must be square for an ESRI ASCII grid")

	nrows, ncols = grid.shape
	xllcorner = transform.c
	yllcorner = transform.f + nrows * transform.e
	cellsize = transform.a

	with output_path.open("w", encoding="ascii", newline="\n") as output:
		output.write(f"ncols        {ncols}\n")
		output.write(f"nrows        {nrows}\n")
		output.write(f"xllcorner    {xllcorner:.12f}\n")
		output.write(f"yllcorner    {yllcorner:.12f}\n")
		output.write(f"cellsize     {cellsize:.12f}\n")
		output.write(f"NODATA_value {NODATA_VALUE}\n")
		for row in grid:
			output.write(" ".join(f"{value:.6g}" for value in row) + "\n")


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

def make_rooting_depth_map(
	csv_path: Path,
	kharif_path: Path,
	rabi_path: Path,
	watershed_path: Path,
	output_path: Path):

	depth_lookup = read_depth_lookup(csv_path)
	kharif, rabi, profile = read_crop_maps(kharif_path, rabi_path)
	watershed_mask = read_watershed_mask(watershed_path, profile)
	kharif_depth = class_depth_grid(kharif, depth_lookup, "Kharif raster")
	rabi_depth = class_depth_grid(rabi, depth_lookup, "Rabi raster")
	rooting_depth: np.ndarray = np.full(
		kharif_depth.shape, NODATA_VALUE, dtype=np.float32)
	rooting_depth[watershed_mask] = np.maximum(
		kharif_depth[watershed_mask], rabi_depth[watershed_mask])
	write_ascii_grid(output_path, rooting_depth, profile)

	print(f"Wrote {output_path}")
	print(f"Grid: {rooting_depth.shape[1]} columns x {rooting_depth.shape[0]} rows")
	print(f"Rooting depth range: {rooting_depth.min():g} to {rooting_depth.max():g} m")


if __name__ == "__main__":
	make_rooting_depth_map(
		DEFAULT_CSV,
		DEFAULT_KHARIF,
		DEFAULT_RABI,
		DEFAULT_WATERSHED,
		DEFAULT_OUTPUT,
	)

