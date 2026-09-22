import pandas as pd
from pathlib import Path
from make_cropwise_Kc_curves import get_dataset_dates, TIME_ORIGIN, DATASET_YEARS

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
METADATA_FILE = SCRIPT_DIR / "cropwise_Kc.csv"
VERSION = "v3.0"
INPUT_FILE = SCRIPT_DIR / f"cropwise_Kc_curves_{VERSION}.csv"
OUTPUT_FILE = SCRIPT_DIR / f"classwise_Kc_curves_{VERSION}.csv"

CLASS_COL = 0
CROP_COL = 2
AREA_COL = 4
TIME_COL = 0
FALLOW_COL = 1

INTER_CLASS = 7
INTER_MAIN = "Finger millet"
INTER_SIDE = 4
WEIGHT_MAIN = 7
WEIGHT_SIDE = 1

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def weighted_curve(class_crops, crop_curves):
	total_area = class_crops.iloc[:, AREA_COL].sum()
	if total_area <= 0:
		raise ValueError("The total crop area for a class must be positive")

	sum_curve = sum(
		crop_curves[crop.iloc[CROP_COL]] * crop.iloc[AREA_COL]
		for _, crop in class_crops.iterrows())
	
	return sum_curve / total_area


def build_classwise_curves(metadata_file, curves_file):
	crop_data, crop_curves = pd.read_csv(metadata_file), pd.read_csv(curves_file)
	kc_class = crop_curves.iloc[:, [TIME_COL, FALLOW_COL]].rename(
		columns={crop_curves.columns[FALLOW_COL]: "Class_0"})

	for crop_class in sorted(crop_data.iloc[:, CLASS_COL].unique()):
		class_crops = crop_data[crop_data.iloc[:, CLASS_COL] == crop_class]
		kc_class[f"Class_{int(crop_class)}"] = weighted_curve(
			class_crops, crop_curves)

	return kc_class, crop_curves


def make_inter_class(kc_class, crop_curves, inter_main, inter_side, weight_side, weight_main):

	class_numbers = [
		int(column.removeprefix("Class_")) for column in kc_class.columns 
		if column.startswith("Class_")]
	
	inter_class = max(class_numbers) + 1
	side_curve = kc_class[f"Class_{inter_side}"]

	weight_total = weight_side + weight_main
	kc_class[f"Class_{inter_class}"] = (
		weight_main * crop_curves[inter_main] + weight_side * side_curve) / weight_total
	
	return kc_class

# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

def write_classwise_curves():
	kc_class, crop_curves = build_classwise_curves(
			METADATA_FILE,
			INPUT_FILE)
		
	kc_class = make_inter_class(
		kc_class, 
		crop_curves,
		INTER_MAIN,
		INTER_SIDE,
		WEIGHT_SIDE,
		WEIGHT_MAIN)

	kc_class.to_csv(OUTPUT_FILE, index=False)

	time_origin, dataset_end, dataset_days = get_dataset_dates(TIME_ORIGIN, DATASET_YEARS)

	print(f"Saved classwise daily Kc records for {len(kc_class)} days to {OUTPUT_FILE}")
	print(f"Period: {time_origin.date()} to {dataset_end.date()} ({DATASET_YEARS} years)")

if __name__ == "__main__":
	write_classwise_curves()