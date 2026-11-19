from pathlib import Path

import fiona
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.mask import raster_geometry_mask
from rasterio.warp import transform_geom
from tqdm import tqdm

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "Watershed ETf SSEBop-ERA5 Land 30m_gapfilled_spatial"
OUTPUT_DIR = INPUT_DIR / "watershed area"
WATERSHED_FILE = SCRIPT_DIR / "Updated Watershed" / "Updated_watershed.shp"
FILL_NODATA_ZERO = True
NODATA = -99999.0
EXTRA_NODATA_VALUES = (-9999.0, NODATA)
RASTER_PATTERN = "[0-9][0-9][0-9][0-9]_*/**/*.tif"

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------


def read_watershed(path: Path) -> tuple[list[dict], str]:
	with fiona.open(path) as boundary:
		watershed_crs = boundary.crs_wkt or boundary.crs
		if not watershed_crs:
			raise ValueError(f"Watershed has no CRS: {path}")
		geometries = [feature["geometry"] for feature in boundary if feature["geometry"]]
	if not geometries:
		raise ValueError(f"Watershed has no geometries: {path}")
	return geometries, watershed_crs


def make_watershed_mask(
	shape: tuple[int, int],
	transform: rasterio.Affine,
	crs: rasterio.CRS,
	geometries: list[dict],
	watershed_crs: str,
) -> np.ndarray:
	projected = [transform_geom(watershed_crs, crs, geometry) for geometry in geometries]
	return geometry_mask(
		projected,
		out_shape=shape,
		transform=transform,
		all_touched=True,
		invert=True,
	)


def process_watershed_frame(
	frame: np.ndarray,
	inside_mask: np.ndarray,
	nodata_values: tuple[float, ...] = EXTRA_NODATA_VALUES,
) -> np.ndarray:
	"""Fill missing in-watershed pixels with zero and mask pixels outside."""
	values = np.asarray(frame, dtype=np.float32).copy()
	missing = ~np.isfinite(values)
	for nodata in nodata_values:
		missing |= values == nodata
	values[missing & inside_mask] = 0.0
	values[~inside_mask] = NODATA
	return values


def extract_raster(
	input_path: Path,
	output_path: Path,
	geometries: list[dict],
	watershed_crs: str,
	fill_nodata_zero: bool,
) -> None:
	with rasterio.open(input_path) as source:
		if source.crs is None:
			raise ValueError(f"Raster has no CRS: {input_path}")
		shapes = [transform_geom(watershed_crs, source.crs, geometry) for geometry in geometries]
		outside, transform, window = raster_geometry_mask(
			source, shapes, crop=True, all_touched=True,
		)
		data = source.read(1, window=window, masked=True).astype(np.float32)
		values = np.ma.filled(data, np.nan)
		if fill_nodata_zero:
			values = process_watershed_frame(values, ~outside, EXTRA_NODATA_VALUES)
		else:
			missing = ~np.isfinite(values)
			for nodata in EXTRA_NODATA_VALUES:
				missing |= values == nodata
			values[missing & ~outside] = NODATA
		values[outside] = NODATA

		profile = source.profile.copy()
		profile.update(
			width=values.shape[1], height=values.shape[0], transform=transform,
			dtype="float32", nodata=NODATA, count=1, compress="deflate",
		)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		with rasterio.open(output_path, "w", **profile) as destination:
			destination.write(values, 1)
			destination.update_tags(**source.tags())
			destination.update_tags(1, **source.tags(1))


def extract_dataset(
	input_dir: Path,
	output_dir: Path,
	watershed_path: Path,
	fill_nodata_zero: bool = False,
	raster_pattern: str = RASTER_PATTERN,
) -> None:
	input_dir = input_dir.resolve()
	output_dir = output_dir.resolve()
	if input_dir == output_dir:
		raise ValueError("Output directory must be different from the input directory")
	files = sorted(input_dir.glob(raster_pattern))
	if not files:
		raise FileNotFoundError(f"No rasters matching {raster_pattern!r} found in {input_dir}")
	geometries, watershed_crs = read_watershed(watershed_path)
	for input_path in tqdm(files, desc="Extracting watershed", unit="raster"):
		output_path = output_dir / input_path.relative_to(input_dir)
		extract_raster(input_path, output_path, geometries, watershed_crs, fill_nodata_zero)
	print(f"Wrote {len(files)} rasters to {output_dir}")


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	extract_dataset(INPUT_DIR, OUTPUT_DIR, WATERSHED_FILE, FILL_NODATA_ZERO)