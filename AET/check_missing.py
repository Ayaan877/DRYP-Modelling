import csv
import datetime as dt
from pathlib import Path
import fiona
import numpy as np
from openpyxl import Workbook
from rasterio.features import geometry_mask
import rasterio
from rasterio.warp import transform_geom
import watershed_data

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

DATASET = "ETf SSEBop ERA5 Land 30m"
START_DATE = dt.date(2015, 1, 1)
END_DATE = dt.date(2025, 12, 31)

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_CONFIG = watershed_data.DATASETS[DATASET]
OUTPUT_DIR = SCRIPT_DIR / DATASET_CONFIG.folder
if DATASET_CONFIG.cadence_days != 1:
	raise ValueError(f"{DATASET} is not a daily dataset")

OUTPUT_PREFIX = DATASET.replace(" ", "_")
MISSING_DATES_OUTPUT = OUTPUT_DIR / f"{OUTPUT_PREFIX}_missing_dates.csv"
PER_PIXEL_OUTPUT = OUTPUT_DIR / f"{OUTPUT_PREFIX}_missing_values_per_pixel.tif"
MATRIX_OUTPUT = OUTPUT_DIR / f"{OUTPUT_PREFIX}_missing_pixel_matrices.xlsx"
WATERSHED_FILE = SCRIPT_DIR / "Updated Watershed" / "Updated_watershed.shp"

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def iter_dates(start, end):
	for offset in range((end - start).days + 1):
		yield start + dt.timedelta(days=offset)


def load_year(year):
	try:
		return watershed_data.load_data(DATASET, year)
	except FileNotFoundError:
		return None


def load_watershed_mask(series):
	with fiona.open(WATERSHED_FILE) as source:
		source_crs = source.crs_wkt or source.crs
		geometries = [
			transform_geom(source_crs, series.crs, feature["geometry"])
			for feature in source
			if feature["geometry"] is not None
		]
	if not geometries:
		raise ValueError(f"No watershed geometries found in {WATERSHED_FILE}")
	return geometry_mask(
		geometries,
		out_shape=series.values.shape[1:],
		transform=series.transform,
		invert=True,
		all_touched=False,
	)


def write_missing_date_counts(year_counts):
	with MISSING_DATES_OUTPUT.open("w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)
		writer.writerow(["year", "missing_dates", "no_observation_dates", "all_nodata_dates"])
		for year in sorted(year_counts):
			missing_dates, no_observation_dates, all_nodata_dates = year_counts[year]
			writer.writerow([year, missing_dates, no_observation_dates, all_nodata_dates])


def write_monthly_matrices(missing_pixels_by_date, years):
	workbook = Workbook(write_only=True)
	for month in range(1, 13):
		worksheet = workbook.create_sheet(title=dt.date(2000, month, 1).strftime("%B"))
		worksheet.append(["year", *range(1, 32)])
		for year in years:
			row = [year]
			for day in range(1, 32):
				try:
					current = dt.date(year, month, day)
				except ValueError:
					row.append(None)
					continue
				if START_DATE <= current <= END_DATE:
					row.append(missing_pixels_by_date[current])
				else:
					row.append(None)
			worksheet.append(row)
	workbook.save(MATRIX_OUTPUT)


def run_missing_date_checker():
	if START_DATE > END_DATE:
		raise ValueError("START_DATE must be on or before END_DATE")
	years = list(range(START_DATE.year, END_DATE.year + 1))
	first_series = None
	first_year = None
	for year in years:
		first_series = load_year(year)
		if first_series is not None:
			first_year = year
			break
	if first_series is None:
		raise FileNotFoundError(f"No watershed data found for {DATASET} in {START_DATE.year}-{END_DATE.year}")

	grid_shape = first_series.values.shape[1:]
	inside_watershed = load_watershed_mask(first_series)
	if not inside_watershed.any():
		raise ValueError(f"Watershed mask does not overlap the grid for {DATASET}")
	missing_pixels_by_date = {}
	year_counts = {}
	grid_metadata = (first_series.transform, first_series.crs)
	total_missing_value_counts = np.zeros(grid_shape, dtype=np.uint32)

	for year in years:
		series = first_series if year == first_year else load_year(year)
		if series is not None:
			if series.values.shape[1:] != grid_shape or (series.transform, series.crs) != grid_metadata:
				raise ValueError(f"Grid changed between years for {DATASET}, including {year}")
			frames_by_date = {
				observed_date: series.values[index]
				for index, observed_date in enumerate(series.dates)
			}
			if len(frames_by_date) != len(series.dates):
				raise ValueError(f"Duplicate dates found for {DATASET} in {year}")
		else:
			frames_by_date = {}

		no_observation_dates = 0
		all_nodata_dates = 0
		missing_dates = 0
		missing_value_counts = np.zeros(grid_shape, dtype=np.uint32)
		for current in iter_dates(max(START_DATE, dt.date(year, 1, 1)), min(END_DATE, dt.date(year, 12, 31))):
			frame = frames_by_date.get(current)
			if frame is None:
				missing = inside_watershed.copy()
				no_observation_dates += 1
			else:
				missing = ~np.isfinite(frame) & inside_watershed
				if missing[inside_watershed].all():
					all_nodata_dates += 1
			if missing[inside_watershed].all():
				missing_dates += 1
			missing_value_counts += missing
			missing_pixels_by_date[current] = int(missing.sum())
		year_counts[year] = (missing_dates, no_observation_dates, all_nodata_dates)
		total_missing_value_counts += missing_value_counts

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	write_missing_date_counts(year_counts)
	write_monthly_matrices(missing_pixels_by_date, years)
	total_missing_value_counts = total_missing_value_counts.astype(np.float32)
	total_missing_value_counts[~inside_watershed] = -9999
	with rasterio.open(
		PER_PIXEL_OUTPUT,
		"w",
		driver="GTiff",
		height=grid_shape[0],
		width=grid_shape[1],
		count=1,
		dtype="float32",
		crs=first_series.crs,
		transform=first_series.transform,
		nodata=-9999,
		compress="deflate",
	) as output:
		output.write(total_missing_value_counts, 1)

	print(f"Dataset: {DATASET}")
	print(f"Date range: {START_DATE} to {END_DATE}")
	print(f"Missing date counts: {MISSING_DATES_OUTPUT}")
	print(f"Missing pixels per day matrices: {MATRIX_OUTPUT}")
	print(f"Missing values per pixel: {PER_PIXEL_OUTPUT}")


if __name__ == "__main__":
	run_missing_date_checker()