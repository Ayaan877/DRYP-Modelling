import pandas as pd
from pathlib import Path

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "cropwise_Kc.csv"
VERSION = "v3.0"
OUTPUT_FILE = SCRIPT_DIR / f"cropwise_Kc_curves_{VERSION}.csv"
TIME_ORIGIN = "2015-01-01"
DATASET_YEARS = 10
FALLOW_KC = 0.1

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def interpolate(start_value, end_value, day_number, period_days):
	if period_days <= 1:
		return end_value
	
	slope = day_number / (period_days - 1)

	return start_value + slope * (end_value - start_value)


def get_dataset_dates(time_origin_text, dataset_years):
	time_origin = pd.Timestamp(time_origin_text)
	dataset_end = time_origin + pd.DateOffset(years=dataset_years)
	dataset_days = (dataset_end - time_origin).days

	return time_origin, dataset_end, dataset_days


def get_sowing_date(crop, time_origin):
	sowing_text = crop["Sowing_date"]
	if pd.isna(sowing_text):
		raise ValueError(f"Missing sowing date for {crop['Crop_type']}")

	day, month = str(sowing_text).split("-")
	sowing_date = pd.Timestamp(
		year=time_origin.year, 
		month=int(month), 
		day=int(day))
	
	if sowing_date >= time_origin:
		sowing_date -= pd.DateOffset(years=1)

	return sowing_date


def build_kc_curve(crop, dataset_days, sowing_date, time_origin):
	ini_period = int(crop["Initial_stage_(days)"])
	dev_period = int(crop["Dev_stage_(days)"])
	mid_period = int(crop["Mid_stage_(days)"])
	late_period = int(crop["Late_stage_(days)"])
	growth_period = ini_period + dev_period + mid_period + late_period

	kc_ini = float(crop["Kc_ini"])
	kc_mid = float(crop["Kc_mid"])
	kc_end = float(crop["Kc_end"])
	
	is_perennial = crop["Season"] == "Perennial"

	curve = [kc_ini] * dataset_days

	for day in range(dataset_days):
		current_date = time_origin + pd.Timedelta(days=day)

		if is_perennial:
			elapsed_day = (current_date - sowing_date).days
			growth_day = elapsed_day % growth_period
		else:
			cycle_start = pd.Timestamp(
				year=current_date.year,
				month=sowing_date.month,
				day=sowing_date.day,
			)
			if cycle_start > current_date:
				cycle_start -= pd.DateOffset(years=1)
			growth_day = (current_date - cycle_start).days
			if growth_day >= growth_period:
				continue

		if growth_day < ini_period:
			kc_value = kc_ini
		elif growth_day < ini_period + dev_period:
			dev_day = growth_day - ini_period
			kc_value = interpolate(kc_ini, kc_mid, dev_day, dev_period)
		elif growth_day < ini_period + dev_period + mid_period:
			kc_value = kc_mid
		else:
			late_day = growth_day - ini_period - dev_period - mid_period
			kc_value = interpolate(kc_mid, kc_end, late_day, late_period)

		curve[day] = kc_value

	return curve


def build_cropwise_curves(input_file, time_origin_text, dataset_years, fallow_kc):
	time_origin, dataset_end, dataset_days = get_dataset_dates(time_origin_text, dataset_years)
	crop_data = pd.read_csv(input_file)
	kc_crops = {"Time": range(dataset_days), "Fallow": [fallow_kc] * dataset_days}

	for _, crop in crop_data.iterrows():
		crop_name = crop["Crop_type"]
		sowing_date = get_sowing_date(crop, time_origin)
		kc_crops[crop_name] = build_kc_curve(
			crop, dataset_days, sowing_date, time_origin)

	return pd.DataFrame(kc_crops), time_origin, dataset_end

# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

def write_cropwise_curves():
	curves, time_origin, dataset_end = build_cropwise_curves(
		INPUT_FILE,
		TIME_ORIGIN,
		DATASET_YEARS,
		FALLOW_KC)
	
	curves.to_csv(OUTPUT_FILE, index=False)
	
	print(f"Saved cropwise daily Kc records for {len(curves)} days to {OUTPUT_FILE}")
	print(f"Period: {time_origin.date()} to {dataset_end.date()} ({DATASET_YEARS} years)")

if __name__ == "__main__":
	write_cropwise_curves()