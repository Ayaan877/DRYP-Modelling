from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
METADATA_FILE = SCRIPT_DIR / "cropwise_Kc.csv"
VERSION = "v3.0"
CLASS_CURVES_FILE = SCRIPT_DIR / f"classwise_Kc_curves_{VERSION}.csv"
OUTPUT_FILE = SCRIPT_DIR / f"class_calendar_{VERSION}.csv"

CLASS_COLUMN = "Class"
CROP_COLUMN = "Crop_type"
SOWING_COLUMN = "Sowing_date"
GROWTH_COLUMN = "Growth_period_(days)"
CLASS_PREFIX = "Class_"
FALLOW_CLASS = 0
INTER_CLASS = 7
INTER_MAIN_CROP = "Finger millet"
INTER_SIDE_CLASS = 4
PERENNIAL_CLASS = 3


def parse_day_month(value):
	day, month = (int(part) for part in str(value).split("-"))
	return month, day


def annual_day(month, day):
	return (pd.Timestamp(year=2001, month=month, day=day) - pd.Timestamp("2001-01-01")).days


def format_day(day_number):
	date = pd.Timestamp("2001-01-01") + pd.Timedelta(days=int(day_number) % 365)
	return date.strftime("%d-%m")


def crop_days(sowing_date, growth_period):
	month, day = parse_day_month(sowing_date)
	start_day = annual_day(month, day)
	return {(start_day + offset) % 365 for offset in range(int(growth_period))}


def perennial_harvest_dates(sowing_date, growth_period):
	start_day = annual_day(*parse_day_month(sowing_date)) - 365
	harvest_day = start_day + int(growth_period) - 1
	harvest_dates = []

	while harvest_day < 365:
		if harvest_day >= 0:
			harvest_dates.append(format_day(harvest_day))
		harvest_day += int(growth_period)

	return harvest_dates


def class_calendar(crops, perennial=False):
	occupied_days = set()
	harvest_days = []
	perennial_harvests = []

	for _, crop in crops.iterrows():
		month, day = parse_day_month(crop[SOWING_COLUMN])
		start_day = annual_day(month, day)
		growth_period = int(crop[GROWTH_COLUMN])
		occupied_days.update(crop_days(crop[SOWING_COLUMN], growth_period))
		if perennial:
			perennial_harvests.extend(perennial_harvest_dates(
				crop[SOWING_COLUMN], growth_period))
		else:
			harvest_days.append((start_day + growth_period - 1) % 365)

	return {
		"sowing_date": format_day(min(annual_day(*parse_day_month(value)) for value in crops[SOWING_COLUMN])),
		"harvest_date": "; ".join(perennial_harvests) if perennial else format_day(max(harvest_days)),
		"growth_period": 365 if perennial else len(occupied_days),
	}


def class_crop_names(crop_data, crop_class):
	return list(crop_data.loc[crop_data[CLASS_COLUMN] == crop_class, CROP_COLUMN])


def crop_seasons(crops):
	return "; ".join(dict.fromkeys(crops["Season"].dropna()))


def build_calendar(metadata_file, class_curves_file):
	crop_data = pd.read_csv(metadata_file)
	class_curves = pd.read_csv(class_curves_file)
	rows = []

	for column in class_curves.columns:
		if not column.startswith(CLASS_PREFIX):
			continue

		crop_class = int(column.removeprefix(CLASS_PREFIX))
		if crop_class == FALLOW_CLASS:
			crop_names = ["Fallow"]
			season = ""
			calendar_values = {
				"sowing_date": "",
				"harvest_date": "",
				"growth_period": 0,
			}
		else:
			if crop_class == INTER_CLASS:
				crop_names = [INTER_MAIN_CROP] + class_crop_names(
					crop_data, INTER_SIDE_CLASS)
				calendar_crops = pd.concat([
					crop_data[crop_data[CROP_COLUMN] == INTER_MAIN_CROP],
					crop_data[crop_data[CLASS_COLUMN] == INTER_SIDE_CLASS],
				])
			else:
				crop_names = class_crop_names(crop_data, crop_class)
				calendar_crops = crop_data[crop_data[CLASS_COLUMN] == crop_class]
			season = crop_seasons(calendar_crops)
			calendar_values = class_calendar(
				calendar_crops,
				perennial=crop_class == PERENNIAL_CLASS)
			if crop_class == PERENNIAL_CLASS:
				calendar_values["harvest_date"] = "bi-yearly"

		values = pd.to_numeric(class_curves[column], errors="coerce").dropna()
		kc_range = f"{values.min():.2f}-{values.max():.2f}"

		rows.append({
			"class": crop_class,
			"season": season,
			"crop types": "; ".join(crop_names),
			"sowing date": calendar_values["sowing_date"],
			"harvest date": calendar_values["harvest_date"],
			"growth period (days)": calendar_values["growth_period"],
			"Kc range": kc_range,
		})

	return pd.DataFrame(rows).sort_values("class")


def write_calendar():
	calendar_data = build_calendar(
		METADATA_FILE,
		CLASS_CURVES_FILE)
	calendar_data.to_csv(OUTPUT_FILE, index=False)
	print(f"Saved classwise Kc calendar to {OUTPUT_FILE}")


if __name__ == "__main__":
	write_calendar()