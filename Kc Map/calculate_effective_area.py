import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
VERSION = "v3.0"
TIME_ORIGIN = "2015-01-01"
ANALYSIS_DAYS = 365
FALLOW_KC = 0.1
EXPECTED_CRS = "EPSG:32643"

KHARIF_RASTER = SCRIPT_DIR / "CropMap_Kharif_2024-25_new.tif"
RABI_RASTER = SCRIPT_DIR / "CropMap_Rabi_2024-25_new.tif"
CLASS_CURVES_FILE = SCRIPT_DIR / f"classwise_Kc_curves_{VERSION}.csv"
CLASS_CALENDAR_FILE = SCRIPT_DIR / f"class_calendar_{VERSION}.csv"
OUTPUT_FILE = SCRIPT_DIR / f"effective_area_{VERSION}.csv"

CLASS_PREFIX = "Class_"
INTER_CLASS = 7
MILLET_CLASS = 2
AVARE_CLASS = 4

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def read_inputs():
	class_curves = pd.read_csv(CLASS_CURVES_FILE)
	calendar = pd.read_csv(CLASS_CALENDAR_FILE)

	curves = {
		int(column.removeprefix(CLASS_PREFIX)): class_curves[column].to_numpy(
			dtype=np.float32)
		for column in class_curves.columns
		if column.startswith(CLASS_PREFIX)
	}
	if not curves:
		raise ValueError("No Class_* columns found in the classwise Kc curves")

	with rasterio.open(KHARIF_RASTER) as source:
		kharif = source.read(1, masked=True).filled(np.nan)
		transform = source.transform
		crs = source.crs
	with rasterio.open(RABI_RASTER) as source:
		rabi = source.read(1, masked=True).filled(np.nan)
		if (source.shape != kharif.shape or source.transform != transform
				or source.crs != crs):
			raise ValueError("Kharif and Rabi rasters do not share the same grid")

	if crs is None or crs.to_string() != EXPECTED_CRS:
		raise ValueError(f"Crop maps must use {EXPECTED_CRS}; found {crs}")

	pixel_area = abs(transform.a * transform.e - transform.b * transform.d)
	return calendar, curves, kharif, rabi, pixel_area


def get_calendar_mask(calendar_row, dates):
	"""Return dates inside a calendar row's recurring annual growth window."""
	growth_period = int(calendar_row["growth period (days)"])
	if growth_period <= 0:
		return np.ones(len(dates), dtype=bool)
	sowing_day, sowing_month = map(int, str(calendar_row["sowing date"]).split("-"))

	mask = np.zeros(len(dates), dtype=bool)

	for index, date in enumerate(dates):
		cycle_start = pd.Timestamp(
			year=date.year, month=sowing_month, day=sowing_day)
		if cycle_start > date:
			cycle_start -= pd.DateOffset(years=1)
		mask[index] = (date - cycle_start).days < growth_period

	return mask


def calendar_masks(calendar, dates):
	"""Union all calendar rows belonging to each class."""
	masks = {}
	for crop_class, rows in calendar.groupby("class"):
		class_mask = np.zeros(len(dates), dtype=bool)
		for _, row in rows.iterrows():
			class_mask |= get_calendar_mask(row, dates)
		masks[int(crop_class)] = class_mask
	return masks


def overlap_pair_counts(kharif, rabi, overlap):
	"""Count overlapping pixels by their two source-map class pair."""
	pairs = np.column_stack((kharif[overlap].astype(int), rabi[overlap].astype(int)))
	return np.unique(pairs, axis=0, return_counts=True)


def format_range(minimum, maximum):
	if np.isclose(minimum, maximum):
		return f"{minimum:.2f}"
	return f"{minimum:.2f}-{maximum:.2f}"


def calculate_effective_area():
	"""Calculate annual min/max effective area for each class."""
	calendar, curves, kharif, rabi, pixel_area = read_inputs()
	valid_map_values = np.concatenate((kharif[np.isfinite(kharif)], rabi[np.isfinite(rabi)]))
	lookup = np.full(
		(max(max(curves), int(valid_map_values.max())) + 1,
		 ANALYSIS_DAYS),
		np.nan,
		dtype=np.float32,
	)
	for crop_class, curve in curves.items():
		if len(curve) < ANALYSIS_DAYS:
			raise ValueError(
				f"Class_{crop_class} curve has fewer than {ANALYSIS_DAYS} days")
		lookup[crop_class, :] = curve[:ANALYSIS_DAYS]
	lookup[0, :] = FALLOW_KC
	map_classes = np.unique(valid_map_values).astype(int)
	if np.isnan(lookup[map_classes]).any():
		raise ValueError("Crop maps contain a class without a classwise Kc curve")

	dates = pd.date_range(
		TIME_ORIGIN, periods=ANALYSIS_DAYS, freq="D")
	class_masks = calendar_masks(calendar, dates)
	non_overlap = np.isfinite(kharif) & np.isfinite(rabi) & (kharif == rabi)
	kharif_only = np.isfinite(kharif) & ~np.isfinite(rabi)
	rabi_only = np.isfinite(rabi) & ~np.isfinite(kharif)
	overlap = np.isfinite(kharif) & np.isfinite(rabi) & ~non_overlap

	kharif_classes = kharif.astype(int)
	rabi_classes = rabi.astype(int)
	overlap_pairs, overlap_counts = overlap_pair_counts(kharif, rabi, overlap)
	base_area = {
		crop_class: (
			(np.count_nonzero(kharif_only & (kharif_classes == crop_class))
			+ np.count_nonzero(rabi_only & (rabi_classes == crop_class))
			+ np.count_nonzero(non_overlap & (kharif_classes == crop_class)))
			* pixel_area)
		for crop_class in class_masks
	}
	area_by_class = {crop_class: np.zeros(len(dates)) for crop_class in class_masks}

	for day_index in range(len(dates)):
		areas = base_area.copy()
		for (kharif_class, rabi_class), count in zip(
			overlap_pairs, overlap_counts):
			is_intercrop = {
				kharif_class, rabi_class
			} == {MILLET_CLASS, AVARE_CLASS}
			if is_intercrop:
				winner = INTER_CLASS
			else:
				winner = (kharif_class if lookup[kharif_class, day_index] >=
					lookup[rabi_class, day_index] else rabi_class)
			areas[winner] = areas.get(winner, 0) + count * pixel_area
		for crop_class in class_masks:
			area_by_class[crop_class][day_index] = (
				areas.get(crop_class, 0))
			if not class_masks[crop_class][day_index]:
				area_by_class[crop_class][day_index] = np.nan

	return area_by_class, class_masks, dates

# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

def write_effective_area(area_by_class, class_masks, dates,
		output_file=OUTPUT_FILE):
	"""Save one min/max effective-area record per class and year."""
	rows = []
	for crop_class, areas in area_by_class.items():
		for year in sorted(dates.year.unique()):
			year_mask = (dates.year == year) & class_masks[crop_class]
			values = areas[year_mask]
			if len(values) == 0:
				continue
			minimum, maximum = float(values.min()), float(values.max())
			rows.append({
				"class": crop_class,
				"year": year,
				"growth_days": int(year_mask.sum()),
				"min_area_m2": minimum,
				"max_area_m2": maximum,
				"area_range_m2": format_range(minimum, maximum),
			})

	pd.DataFrame(rows).to_csv(output_file, index=False)
	print(f"Wrote {len(rows)} class/year records to {output_file}")


if __name__ == "__main__":
	area_by_class, class_masks, dates = calculate_effective_area()
	write_effective_area(area_by_class, class_masks, dates)