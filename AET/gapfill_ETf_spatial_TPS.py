import re
import hashlib
from collections import OrderedDict
from datetime import date
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.interpolate import RBFInterpolator
from scipy.spatial import KDTree
from tqdm import tqdm
import watershed_data
from extract_watershed_rasters import make_watershed_mask, read_watershed

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
WATERSHED_FILE = SCRIPT_DIR / "Updated Watershed" / "Updated_watershed.shp"
PATTERN_CACHE_BYTES = 4 * 1024**3  # RAM budget for cached per-pattern TPS weights
INPUT_VERSION = "v2"
OUTPUT_VERSION = "v2"
ETF_DATASET = "ETf ssebop_era5 daily 30m" 
DATE_PATTERN = re.compile(r"_(\d{8})\.tif$")  
NODATA = -99999  

TILE_SIZE = 64  
TILE_HALO = 65  
MAX_EXTRAPOLATION_PIXELS = 65  
TPS_NEIGHBOURS = 128  
TPS_SMOOTHING = 100  
MIN_VALID_POINTS = 32  
CLIP_MIN = 0.0  
CLIP_MAX = 1.2 

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def input_files(config):
	"""Find temporally gapfilled rasters and index them by acquisition date."""
	temporal_dir = (SCRIPT_DIR / config.folder).with_name(f"{config.folder}_gapfilled_{INPUT_VERSION}")
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
	"""Read one raster window with NoData and non-finite values as NaN."""
	data = np.ma.filled(source.read(1, window=window, masked=True).astype(np.float64), np.nan)
	data[~np.isfinite(data) | np.isin(data, nodata_values)] = np.nan
	return data


class PatternCache:
	"""LRU cache of TPS weights keyed by tile and valid-pixel pattern, bounded in bytes."""

	def __init__(self, max_bytes):
		self.max_bytes = max_bytes
		self.entries = OrderedDict()
		self.nbytes = 0

	def get(self, key):
		entry = self.entries.get(key)
		if entry is not None:
			self.entries.move_to_end(key)
		return entry

	def put(self, key, entry):
		size = sum(array.nbytes for array in entry)
		if size > self.max_bytes:
			return
		while self.nbytes + size > self.max_bytes:
			_, evicted = self.entries.popitem(last=False)
			self.nbytes -= sum(array.nbytes for array in evicted)
		self.entries[key] = entry
		self.nbytes += size


def build_pattern(valid, target_mask):
	"""Precompute neighbour indices and TPS weights for each fillable target pixel."""
	empty = (np.empty(0, np.intp), np.empty(0, np.intp), np.empty((0, 1), np.int32), np.empty((0, 1)))
	target_rc = np.argwhere(target_mask)
	valid_rc = np.argwhere(valid)
	if len(valid_rc) < MIN_VALID_POINTS or not len(target_rc):
		return empty

	points = valid_rc[:, ::-1].astype(np.float64)
	if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
		return empty

	targets = target_rc[:, ::-1].astype(np.float64)
	tree = KDTree(points)
	distances = np.asarray(tree.query(targets, k=1)[0])
	near = distances <= MAX_EXTRAPOLATION_PIXELS
	target_rc, targets = target_rc[near], targets[near]
	if not len(targets):
		return empty

	neighbours = min(TPS_NEIGHBOURS, len(points))
	indices = np.asarray(tree.query(targets, k=neighbours)[1], dtype=np.intp)
	weights = np.zeros((len(targets), neighbours))
	solved = np.zeros(len(targets), dtype=bool)
	identity = np.eye(neighbours)
	for i in range(len(targets)):
		try:
			# Values enter the TPS linearly, so identity columns yield each neighbour's weight.
			interpolator = RBFInterpolator(
				points[indices[i]], identity, kernel="thin_plate_spline", smoothing=TPS_SMOOTHING,
			)
			weights[i] = interpolator(targets[i:i + 1])[0]
			solved[i] = True
		except (ValueError, np.linalg.LinAlgError):
			continue
	return target_rc[solved, 0], target_rc[solved, 1], indices[solved].astype(np.int32), weights[solved]


def fill_tile(tile, inside, core_slices, tile_id, cache):
	"""TPS-fill missing in-watershed pixels of a read tile's core, in place; returns the count filled."""
	valid = np.isfinite(tile)
	core = np.zeros(tile.shape, dtype=bool)
	core[core_slices] = True
	target_mask = core & inside & ~valid
	if not target_mask.any():
		return 0

	digest = hashlib.blake2b(np.packbits(valid).tobytes(), digest_size=16).digest()
	key = (tile_id, digest)
	entry = cache.get(key)
	if entry is None:
		entry = build_pattern(valid, target_mask)
		cache.put(key, entry)

	rows, cols, neighbours, weights = entry
	if not len(rows):
		return 0
	predictions = np.einsum("ij,ij->i", weights, tile[valid][neighbours])
	predictions = np.clip(predictions, CLIP_MIN, CLIP_MAX)
	finite = np.isfinite(predictions)
	tile[rows[finite], cols[finite]] = predictions[finite]
	return int(finite.sum())


def processing_windows(width, height):
	"""Yield each core slice, its halo-extended read slice, and the core's slice inside the read."""
	for row in range(0, height, TILE_SIZE):
		for col in range(0, width, TILE_SIZE):
			core_rows = slice(row, min(row + TILE_SIZE, height))
			core_cols = slice(col, min(col + TILE_SIZE, width))
			read_rows = slice(max(0, row - TILE_HALO), min(height, core_rows.stop + TILE_HALO))
			read_cols = slice(max(0, col - TILE_HALO), min(width, core_cols.stop + TILE_HALO))
			core_in_read = (
				slice(core_rows.start - read_rows.start, core_rows.stop - read_rows.start),
				slice(core_cols.start - read_cols.start, core_cols.stop - read_cols.start),
			)
			yield (core_rows, core_cols), (read_rows, read_cols), core_in_read


def gapfill_dataset(dataset_name):
	"""Spatially fill temporal rasters inside the watershed and write them to a separate output folder."""
	config = watershed_data.DATASETS[dataset_name]
	if config.kind != "geotiff":
		raise ValueError(f"Expected a GeoTIFF dataset, got {dataset_name!r}")
	if TILE_SIZE <= 0 or TILE_HALO < MAX_EXTRAPOLATION_PIXELS:
		raise ValueError("Tile size must be positive and halo must cover the extrapolation distance")

	temporal_dir, files = input_files(config)
	output_dir = temporal_dir.with_name(f"{temporal_dir.name}_spatial_{OUTPUT_VERSION}")
	geometries, watershed_crs = read_watershed(WATERSHED_FILE)
	with rasterio.open(files[min(files)]) as first:
		inside = make_watershed_mask(
			(first.height, first.width), first.transform, first.crs, geometries, watershed_crs,
		)
	cache = PatternCache(PATTERN_CACHE_BYTES)
	filled_total = 0

	for day, input_path in tqdm(sorted(files.items()), desc="Spatial TPS rasters", unit="raster"):
		relative_path = input_path.relative_to(temporal_dir)
		output_path = output_dir / relative_path
		output_path.parent.mkdir(parents=True, exist_ok=True)

		with rasterio.open(input_path) as source:
			if source.shape != inside.shape:
				raise ValueError(f"Raster grid differs from the first input: {input_path}")
			profile = source.profile.copy()
			profile.update(dtype="float32", nodata=NODATA, compress="deflate", count=1)
			with rasterio.open(output_path, "w", **profile) as destination:
				for core_slices, read_slices, core_in_read in processing_windows(source.width, source.height):
					tile = read_tile(source, Window.from_slices(*read_slices), config.nodata_values)
					tile[~inside[read_slices]] = np.nan
					filled_total += fill_tile(
						tile, inside[read_slices], core_in_read, (core_slices[0].start, core_slices[1].start), cache,
					)
					filled = tile[core_in_read]
					data = np.where(np.isfinite(filled), filled, NODATA).astype(np.float32)
					destination.write(data, 1, window=Window.from_slices(*core_slices))
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