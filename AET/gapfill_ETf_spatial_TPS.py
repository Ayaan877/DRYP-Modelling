import re
from datetime import date
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.interpolate import RBFInterpolator
from scipy.spatial import KDTree
from tqdm import tqdm
import watershed_data

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ETF_DATASET = "ETf SSEBop ERA5 Land 30m" 
DATE_PATTERN = re.compile(r"_(\d{8})\.tif$")  
NODATA = -99999  

TILE_SIZE = 64  
TILE_HALO = 8  
MAX_EXTRAPOLATION_PIXELS = 4  
TPS_NEIGHBOURS = 32  
TPS_SMOOTHING = 0.0  
MIN_VALID_POINTS = 6  
CLIP_MIN = 0.0  
CLIP_MAX = None 

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def input_files(config):
	"""Find temporally gapfilled rasters and index them by acquisition date."""
	temporal_dir = (SCRIPT_DIR / config.folder).with_name(f"{config.folder}_gapfilled")
	paths = sorted(temporal_dir.glob(
		f"[0-9][0-9][0-9][0-9]_{config.resolution}/{config.pattern}"
	))
	files = {}
	for path in paths:
		match = DATE_PATTERN.search(path.name)
		if not match:
			continue
		text = match.group(1)
		day = date(int(text[:4]), int(text[4:6]), int(text[6:])).toordinal()
		if day in files:
			raise ValueError(f"Duplicate ETf date {text}: {files[day]} and {path}")
		files[day] = path
	if not files:
		raise FileNotFoundError(f"No temporally gapfilled rasters found in {temporal_dir}")
	return temporal_dir, files


def read_tile(source, window, nodata_values):
	"""Read one raster window and represent all NoData values as NaN."""
	data = np.ma.filled(
		source.read(1, window=window, masked=True).astype(np.float64), np.nan
	)
	for nodata in nodata_values:
		data[data == nodata] = np.nan
	if source.nodata is not None:
		data[data == source.nodata] = np.nan
	data[~np.isfinite(data)] = np.nan
	return data


def fill_tile(tile, core_window, read_window):
	"""TPS-fill nearby missing pixels in a core tile, using its halo for support."""
	row_offset = int(core_window.row_off - read_window.row_off)
	col_offset = int(core_window.col_off - read_window.col_off)
	core_height = int(core_window.height)
	core_width = int(core_window.width)
	core = tile[
		row_offset:row_offset + core_height,
		col_offset:col_offset + core_width,
	]
	missing_rows, missing_cols = np.nonzero(~np.isfinite(core))
	if not len(missing_rows):
		return core, 0

	valid_rows, valid_cols = np.nonzero(np.isfinite(tile))
	if len(valid_rows) < MIN_VALID_POINTS:
		return core, 0

	points = np.column_stack((valid_cols, valid_rows)).astype(np.float64)
	values = tile[valid_rows, valid_cols]
	if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
		return core, 0

	targets = np.column_stack((
		missing_cols + col_offset,
		missing_rows + row_offset,
	)).astype(np.float64)
	tree = KDTree(points)
	distances, _ = tree.query(targets, k=1)
	nearby = np.flatnonzero(distances <= MAX_EXTRAPOLATION_PIXELS)
	if not len(nearby):
		return core, 0

	neighbours = min(TPS_NEIGHBOURS, len(points))
	if neighbours < MIN_VALID_POINTS:
		return core, 0

	selected = nearby

	try:
		interpolator = RBFInterpolator(
			points,
			values,
			neighbors=neighbours,
			kernel="thin_plate_spline",
			smoothing=TPS_SMOOTHING,
		)
		predictions = interpolator(targets[selected])
	except (ValueError, np.linalg.LinAlgError):
		return core, 0

	finite = np.isfinite(predictions)
	if CLIP_MIN is not None:
		predictions = np.maximum(predictions, CLIP_MIN)
	if CLIP_MAX is not None:
		predictions = np.minimum(predictions, CLIP_MAX)
	finite &= np.isfinite(predictions)
	if not np.any(finite):
		return core, 0

	rows = missing_rows[selected[finite]]
	cols = missing_cols[selected[finite]]
	core[rows, cols] = predictions[finite]
	return core, int(finite.sum())


def processing_windows(width, height):
	"""Yield output tiles and their larger, halo-inclusive input windows."""
	for row in range(0, height, TILE_SIZE):
		for col in range(0, width, TILE_SIZE):
			core = Window.from_slices(
				(row, min(row + TILE_SIZE, height)),
				(col, min(col + TILE_SIZE, width)),
			)
			read_col = max(0, col - TILE_HALO)
			read_row = max(0, row - TILE_HALO)
			read_right = min(width, col + int(core.width) + TILE_HALO)
			read_bottom = min(height, row + int(core.height) + TILE_HALO)
			read = Window.from_slices(
				(read_row, read_bottom),
				(read_col, read_right),
			)
			yield core, read


def gapfill_dataset(dataset_name):
	"""Spatially fill temporal rasters and write them to a separate output folder."""
	config = watershed_data.DATASETS[dataset_name]
	if not dataset_name.startswith("ETf ") or config.kind != "geotiff":
		raise ValueError(f"Expected a GeoTIFF ETf dataset, got {dataset_name!r}")
	if TILE_SIZE <= 0 or TILE_HALO < int(np.ceil(MAX_EXTRAPOLATION_PIXELS)):
		raise ValueError("Tile size must be positive and halo must cover the extrapolation distance")

	temporal_dir, files = input_files(config)
	output_dir = temporal_dir.with_name(f"{temporal_dir.name}_spatial")
	filled_total = 0

	for day, input_path in tqdm(sorted(files.items()), desc="Spatial TPS rasters", unit="raster"):
		current = date.fromordinal(day)
		relative_path = input_path.relative_to(temporal_dir)
		output_path = output_dir / relative_path
		output_path.parent.mkdir(parents=True, exist_ok=True)

		with rasterio.open(input_path) as source:
			profile = source.profile.copy()
			profile.update(dtype="float32", nodata=NODATA, compress="deflate", count=1)
			with rasterio.open(output_path, "w", **profile) as destination:
				for core, read in processing_windows(source.width, source.height):
					tile = read_tile(source, read, config.nodata_values)
					filled_core, count = fill_tile(tile, core, read)
					filled_total += count
					data = np.where(np.isfinite(filled_core), filled_core, NODATA).astype(np.float32)
					destination.write(data, 1, window=core)
				destination.update_tags(
					units="1",
					long_name="ET fraction (AET/ref ET), temporal and thin-plate spatial gap-filled",
					spatial_interpolation="local thin-plate spline RBF",
					max_extrapolation_pixels=str(MAX_EXTRAPOLATION_PIXELS),
				)

	print(f"Wrote {len(files)} rasters to {output_dir}; filled {filled_total} pixels")
	return len(files), filled_total


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	gapfill_dataset(ETF_DATASET)