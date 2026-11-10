from pathlib import Path
import json
import re
from typing import Tuple
import pandas as pd
import geopandas as gpd

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
SPATIAL_BASE_GPKG = SCRIPT_DIR / "spatial_base.gpkg"
SPATIAL_BASE_LAYER = "spatial_base"
INPUT_DIR = SCRIPT_DIR / "Crop_Data_Cleaned"
TARGET_YEARS = [2022, 2023, 2024, 2025]
TARGET_SEASONS = ["kharif", "rabi", "summer"]
OUTPUT_DIR = SCRIPT_DIR / "Crop_Map_Layers"
LOG_DIR = OUTPUT_DIR / "logs"
OVERWRITE_OUTPUT = True

VILLAGE_DICT_FILE = SCRIPT_DIR / "village_code_dict.json"
CROP_DICT_FILE = SCRIPT_DIR / "crop_dict.json"

PARCEL_CATEGORY_VALUE = "parcel"
SUMMARY_LAYER = "summary_polygons"
ASSIGNMENT_LAYER = "crop_assignments"

BASE_FIELD_ALIASES = {
	"survey_num": ["survey_num"],
	"category": ["category"],
	"kgis_code": ["kgis_code"],
	"village": ["village"],
	"area_acres": ["area_acres"],
}

CROP_FIELD_ALIASES = {
	"year": ["year"],
	"season": ["season"],
	"survey_num": ["survey_num"],
	"survey_detail": ["survey_detail"],
	"area_acres": ["area_acres"],
	"crop_name": ["crop_name"],
	"crop_detail": ["crop_detail"],
	"crop_detail_kannada": ["crop_detail_kannada"],
	"crop_category": ["crop_category"],
	"kgis_code": ["kgis_code"],
	"owner_name": ["owner_name"],
	"owner_no": ["owner_no"],
	"land_code": ["land_code"],
}

SUMMARY_COLUMNS = [
	"spatial_fid",
	"season_year",
	"year",
	"season",
	"label",
	"survey_num",
	"category",
	"kgis_code",
	"village",
	"district",
	"taluk",
	"hobli",
	"survey_area_acres",
	"total_cropped_area_acres",
	"subfeature_list",
	"crop_count",
	"crop_names",
	"crop_categories",
	"crop_names_detail_en",
	"crop_names_kannada",
]

ASSIGNMENT_COLUMNS = [
	"spatial_fid",
	"season_year",
	"year",
	"season",
	"label",
	"survey_num",
	"survey_detail",
	"category",
	"kgis_code",
	"village",
	"district",
	"taluk",
	"hobli",
	"survey_area_acres",
	"cropped_area_acres",
	"crop_name",
	"crop_detail_en",
	"crop_name_kannada",
	"crop_category",
	"owner_name",
	"owner_no",
	"land_code",
]

VALID_SEASONS = {"kharif", "rabi", "summer"}
NON_AGRI_KANNADA_CANDIDATES = ["ಕೃಷಿಯೇತರ", "ಪಾಳು"]
DEFAULT_NON_AGRI_META = {
	"crop_name": "Non-Agricultural Land",
	"crop_detail_en": "Non-Agricultural Land",
	"crop_name_kannada": "ಕೃಷಿಯೇತರ",
	"crop_category": "Fallow / Non-Agricultural",
}

DEFAULT_FALLOW_META = {
	"crop_name": "Fallow Land",
	"crop_detail_en": "Fallow Land",
	"crop_name_kannada": "ಪಾಳು",
	"crop_category": "Fallow / Non-Agricultural",
}


# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def normalize_header_name(header):
	return str(header).replace("\ufeff", "").strip().lower()


def normalize_text(value):
	if value is None:
		return None
	text = str(value).strip()
	if text == "":
		return None
	return " ".join(text.split())


def normalize_text_series(series):
	return (
		series.astype("string")
		.str.replace(r"\s+", " ", regex=True)
		.str.strip()
		.replace("", pd.NA)
	)


def normalize_village_code_series(series):
	text = normalize_text_series(series)
	return text.str.replace(r"\.0$", "", regex=True)


def normalize_survey_num_series(series):
	return pd.to_numeric(series, errors="coerce").astype("Int64")


def build_match_key(village_code_series, survey_num_series):
	village = normalize_text_series(village_code_series).fillna("")
	survey = survey_num_series.astype("Int64").astype("string").fillna("")
	return village + "|" + survey


def first_non_empty(*series_list):
	if not series_list:
		raise ValueError("first_non_empty requires at least one series")
	result = normalize_text_series(series_list[0])
	for series in series_list[1:]:
		result = result.fillna(normalize_text_series(series))
	return result


def unique_join(values):
	text_values = sorted({str(value).strip() for value in values if pd.notna(value) and str(value).strip()})
	if not text_values:
		return None
	return " | ".join(text_values)


def build_label_series(survey_num_series, category_series):
	survey_text = survey_num_series.astype("Int64").astype("string")
	category_text = normalize_text_series(category_series).fillna("Unknown Category")
	return "Survey " + survey_text.fillna("NA") + " | " + category_text


def build_assignment_label_series(survey_detail_series, survey_num_series, crop_name_series):
	survey_detail_text = normalize_text_series(survey_detail_series)
	survey_num_text = survey_num_series.astype("Int64").astype("string")
	survey_text = survey_detail_text.fillna(survey_num_text)
	crop_text = normalize_text_series(crop_name_series).fillna("Unknown Crop")
	return "Survey " + survey_text.fillna("NA") + ", " + crop_text


def parse_output_key_from_file_name(input_file):
	match = re.match(r"(\d{4})_([a-z]+)_crop_data_cleaned\.csv$", input_file.name.lower())
	if not match:
		raise ValueError(
			f"Input file name does not match expected format YYYY_season_crop_data_cleaned.csv: {input_file.name}"
		)
	year = int(match.group(1))
	season = match.group(2).lower()
	return year, season, f"{year}_{season}"


def build_crop_map_output_path(output_key):
	return OUTPUT_DIR / f"{output_key}_belle_survey_crop_map.gpkg"


def get_selected_crop_files(input_dir, target_years, target_seasons):
	if not target_years:
		raise ValueError("Set TARGET_YEARS in the Configuration section before running.")
	if not target_seasons:
		raise ValueError("Set TARGET_SEASONS in the Configuration section before running.")

	year_values = [int(year_value) for year_value in target_years]
	season_values = []
	for season_value in target_seasons:
		season_text = str(season_value).strip().lower()
		if season_text not in VALID_SEASONS:
			raise ValueError(
				f"Invalid season value: {season_value!r}. Use one of: {sorted(VALID_SEASONS)}"
			)
		season_values.append(season_text)

	selected_files = []
	missing_files = []
	for year_value in year_values:
		for season_text in season_values:
			file_name = f"{year_value}_{season_text}_crop_data_cleaned.csv"
			candidate_file = input_dir / file_name
			if candidate_file.exists():
				selected_files.append(candidate_file)
			else:
				missing_files.append(str(candidate_file))

	if missing_files:
		raise FileNotFoundError(
			"Missing selected crop files:\n - " + "\n - ".join(missing_files)
		)
	return selected_files


def resolve_columns_by_alias(available_columns, alias_map):
	column_lookup = {normalize_header_name(column): column for column in available_columns}
	resolved = {}
	for canonical_name, aliases in alias_map.items():
		matched = None
		for alias in aliases:
			alias_key = normalize_header_name(alias)
			if alias_key in column_lookup:
				matched = column_lookup[alias_key]
				break
		if matched is None:
			raise ValueError(f"Missing required field '{canonical_name}'. Aliases: {aliases}")
		resolved[canonical_name] = matched
	return resolved


def load_json_dictionary(dictionary_file):
	if not dictionary_file.exists():
		raise FileNotFoundError(f"Missing dictionary file: {dictionary_file}")
	with dictionary_file.open("r", encoding="utf-8") as file:
		return json.load(file)


def build_crop_lookup(crop_dict):
	crop_lookup = {}
	for kannada_name, crop_meta in crop_dict.items():
		key = normalize_text(kannada_name)
		if key is None:
			continue
		crop_lookup[key] = {
			"crop_name": normalize_text(crop_meta.get("crop_name")),
			"crop_detail": normalize_text(crop_meta.get("crop_details")),
			"crop_category": normalize_text(crop_meta.get("crop_category")),
		}
	return crop_lookup


def build_location_maps(village_dict):
	return {
		"village": {
			str(key): normalize_text(value.get("village"))
			for key, value in village_dict.items()
		},
		"district": {
			str(key): normalize_text(value.get("district"))
			for key, value in village_dict.items()
		},
		"taluk": {
			str(key): normalize_text(value.get("taluk"))
			for key, value in village_dict.items()
		},
		"hobli": {
			str(key): normalize_text(value.get("hobli"))
			for key, value in village_dict.items()
		},
	}


def get_non_agri_metadata(crop_lookup):
	for kannada_name in NON_AGRI_KANNADA_CANDIDATES:
		meta = crop_lookup.get(kannada_name)
		if not meta:
			continue
		crop_name = normalize_text(meta.get("crop_name"))
		crop_detail = normalize_text(meta.get("crop_detail"))
		crop_category = normalize_text(meta.get("crop_category"))
		if crop_name and crop_detail and crop_category:
			return {
				"crop_name": crop_name,
				"crop_detail_en": crop_detail,
				"crop_name_kannada": kannada_name,
				"crop_category": crop_category,
			}
	return DEFAULT_NON_AGRI_META.copy()


def get_fallow_metadata(crop_lookup):
	fallow_meta = crop_lookup.get("ಪಾಳು")
	if not fallow_meta:
		return DEFAULT_FALLOW_META.copy()
	crop_name = normalize_text(fallow_meta.get("crop_name"))
	crop_detail = normalize_text(fallow_meta.get("crop_detail"))
	crop_category = normalize_text(fallow_meta.get("crop_category"))
	if crop_name and crop_detail and crop_category:
		return {
			"crop_name": crop_name,
			"crop_detail_en": crop_detail,
			"crop_name_kannada": "ಪಾಳು",
			"crop_category": crop_category,
		}
	return DEFAULT_FALLOW_META.copy()


def get_series_or_na(frame, column_name):
	if column_name is None:
		return pd.Series(pd.NA, index=frame.index, dtype="string")
	return frame[column_name]


def prepare_spatial_base(spatial_base_file, spatial_layer, location_maps):
	base = gpd.read_file(spatial_base_file, layer=spatial_layer)
	base_columns = resolve_columns_by_alias(base.columns, BASE_FIELD_ALIASES)
	base = base.rename(columns={column: key for key, column in base_columns.items()})

	if "fid" in base.columns:
		base["spatial_fid"] = pd.to_numeric(base["fid"], errors="coerce").astype("Int64")
	else:
		base["spatial_fid"] = pd.Series(base.index, index=base.index).astype("Int64")

	base["kgis_code"] = normalize_village_code_series(base["kgis_code"])
	base["survey_num"] = normalize_survey_num_series(base["survey_num"])
	base["category"] = normalize_text_series(base["category"])
	base["village"] = first_non_empty(
		base["village"],
		base["kgis_code"].map(location_maps["village"]),
	)
	base["district"] = normalize_text_series(base["kgis_code"].map(location_maps["district"]))
	base["taluk"] = normalize_text_series(base["kgis_code"].map(location_maps["taluk"]))
	base["hobli"] = normalize_text_series(base["kgis_code"].map(location_maps["hobli"]))
	base["survey_area_acres"] = pd.to_numeric(base["area_acres"], errors="coerce")
	base["base_category_lower"] = base["category"].str.lower()
	base["label"] = build_label_series(base["survey_num"], base["category"])
	base["_match_key"] = build_match_key(base["kgis_code"], base["survey_num"])

	required_output_columns = [
		"spatial_fid",
		"label",
		"survey_num",
		"category",
		"kgis_code",
		"village",
		"district",
		"taluk",
		"hobli",
		"survey_area_acres",
		"base_category_lower",
		"_match_key",
		"geometry",
	]
	return base[required_output_columns].copy()


def load_and_prepare_crop_table(input_file, crop_lookup):
	crop_raw = pd.read_csv(input_file, dtype="string")
	input_record_count = len(crop_raw)
	resolved = resolve_columns_by_alias(crop_raw.columns, CROP_FIELD_ALIASES)
	crop = crop_raw.rename(columns={column: key for key, column in resolved.items()}).copy()

	crop["kgis_code"] = normalize_village_code_series(crop["kgis_code"])
	crop["survey_num"] = normalize_survey_num_series(crop["survey_num"])
	crop["survey_detail"] = normalize_text_series(get_series_or_na(crop, "survey_detail"))
	crop["crop_name_kannada"] = normalize_text_series(crop["crop_detail_kannada"])
	crop["crop_name"] = normalize_text_series(crop["crop_name"])
	crop["crop_detail_en"] = normalize_text_series(crop["crop_detail"])
	crop["crop_category"] = normalize_text_series(crop["crop_category"])
	crop["cropped_area_acres"] = pd.to_numeric(crop["area_acres"], errors="coerce")
	crop["owner_name"] = normalize_text_series(get_series_or_na(crop, "owner_name"))
	crop["owner_no"] = normalize_text_series(get_series_or_na(crop, "owner_no")).str.replace(r"\.0$", "", regex=True)
	crop["land_code"] = normalize_text_series(get_series_or_na(crop, "land_code")).str.replace(r"\.0$", "", regex=True)

	crop_name_from_dict = crop["crop_name_kannada"].map(
		lambda value: crop_lookup.get(value, {}).get("crop_name") if pd.notna(value) else None
	)
	crop_detail_from_dict = crop["crop_name_kannada"].map(
		lambda value: crop_lookup.get(value, {}).get("crop_detail") if pd.notna(value) else None
	)
	crop_category_from_dict = crop["crop_name_kannada"].map(
		lambda value: crop_lookup.get(value, {}).get("crop_category") if pd.notna(value) else None
	)

	crop["crop_name"] = first_non_empty(crop["crop_name"], crop_name_from_dict)
	crop["crop_detail_en"] = first_non_empty(crop["crop_detail_en"], crop_detail_from_dict)
	crop["crop_category"] = first_non_empty(crop["crop_category"], crop_category_from_dict)
	crop["_match_key"] = build_match_key(crop["kgis_code"], crop["survey_num"])

	valid = crop["kgis_code"].notna() & crop["survey_num"].notna() & crop["crop_name"].notna()
	crop = crop.loc[valid].copy()
	return (
		crop[
			[
				"_match_key",
				"kgis_code",
				"survey_num",
				"survey_detail",
				"crop_name",
				"crop_detail_en",
				"crop_name_kannada",
				"crop_category",
				"cropped_area_acres",
				"owner_name",
				"owner_no",
				"land_code",
			]
		].copy(),
		input_record_count,
	)


def build_assignment_layer(
	base: gpd.GeoDataFrame,
	crop_rows: pd.DataFrame,
	year: int,
	season: str,
	output_key: str,
	non_agri_meta: dict,
	fallow_meta: dict,
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame, dict]:
	merged = base.merge(
		crop_rows,
		on="_match_key",
		how="left",
		suffixes=("_base", "_crop"),
	)
	merged["kgis_code"] = merged["kgis_code_base"]
	merged["survey_num"] = merged["survey_num_base"]

	unmatched_crop_records = crop_rows.merge(
		base[["_match_key"]].drop_duplicates(),
		on="_match_key",
		how="left",
		indicator=True,
	)
	unmatched_crop_records = unmatched_crop_records.loc[
		unmatched_crop_records["_merge"].eq("left_only")
	].drop(columns=["_merge"])

	matched = merged.loc[merged["crop_name"].notna()].copy()
	matched_assignments = matched[
		[
			"spatial_fid",
			"label",
			"survey_num",
			"survey_detail",
			"category",
			"kgis_code",
			"village",
			"district",
			"taluk",
			"hobli",
			"survey_area_acres",
			"crop_name",
			"crop_detail_en",
			"crop_name_kannada",
			"crop_category",
			"cropped_area_acres",
			"owner_name",
			"owner_no",
			"land_code",
			"geometry",
		]
	].copy()
	matched_assignments["survey_detail"] = normalize_text_series(matched_assignments["survey_detail"])
	matched_assignments["_subfeature_text"] = matched_assignments["crop_name"]

	unassigned_polygons = merged.loc[merged["crop_name"].isna()].copy()
	unassigned_polygons = (
		unassigned_polygons[
			[
				"spatial_fid",
				"label",
				"survey_num",
				"survey_detail",
				"category",
				"kgis_code",
				"village",
				"district",
				"taluk",
				"hobli",
				"survey_area_acres",
				"owner_name",
				"owner_no",
				"land_code",
				"geometry",
			]
		]
		.drop_duplicates(subset=["spatial_fid"])
		.copy()
	)
	unassigned_polygons["crop_name"] = non_agri_meta["crop_name"]
	unassigned_polygons["crop_detail_en"] = non_agri_meta["crop_detail_en"]
	unassigned_polygons["crop_name_kannada"] = non_agri_meta["crop_name_kannada"]
	unassigned_polygons["crop_category"] = non_agri_meta["crop_category"]
	unassigned_polygons["cropped_area_acres"] = unassigned_polygons["survey_area_acres"]

	unassigned_is_parcel = (
		unassigned_polygons["category"]
		.astype("string")
		.str.lower()
		.fillna("")
		.eq(PARCEL_CATEGORY_VALUE)
	)
	unassigned_parcel = unassigned_polygons.loc[unassigned_is_parcel].copy()
	unassigned_non_parcel = unassigned_polygons.loc[~unassigned_is_parcel].copy()

	unassigned_parcel["crop_name"] = fallow_meta["crop_name"]
	unassigned_parcel["crop_detail_en"] = fallow_meta["crop_detail_en"]
	unassigned_parcel["crop_name_kannada"] = fallow_meta["crop_name_kannada"]
	unassigned_parcel["crop_category"] = fallow_meta["crop_category"]
	unassigned_non_parcel["_subfeature_text"] = unassigned_non_parcel["crop_name"]
	unassigned_parcel["_subfeature_text"] = unassigned_parcel["crop_name"]

	assignment_work = pd.concat(
		[matched_assignments, unassigned_non_parcel, unassigned_parcel],
		ignore_index=True,
	)
	assignment_work["label"] = build_assignment_label_series(
		assignment_work["survey_detail"],
		assignment_work["survey_num"],
		assignment_work["crop_name"],
	)
	assignment_work["season_year"] = output_key
	assignment_work["year"] = year
	assignment_work["season"] = season
	assignment = assignment_work[ASSIGNMENT_COLUMNS + ["geometry"]].copy()

	summary_subfeatures = assignment_work[
		["spatial_fid", "_subfeature_text"]
	].drop_duplicates()
	assignment_stats = {
		"assigned_polygons": int(matched["spatial_fid"].nunique(dropna=True)),
		"unassigned_fallow_polygons": int(unassigned_parcel["spatial_fid"].nunique(dropna=True)),
		"unassigned_non_agri_polygons": int(unassigned_non_parcel["spatial_fid"].nunique(dropna=True)),
	}
	return assignment, unmatched_crop_records, summary_subfeatures, assignment_stats


def build_summary_layer(base, assignment, summary_subfeatures, year: int, season: str, output_key: str):
	summary_stats = (
		assignment.groupby(["spatial_fid"], dropna=False)
		.agg(
			crop_count=("crop_name", lambda values: pd.Series(values).dropna().nunique()),
			crop_names=("crop_name", unique_join),
			crop_categories=("crop_category", unique_join),
			crop_names_detail_en=("crop_detail_en", unique_join),
			crop_names_kannada=("crop_name_kannada", unique_join),
			total_cropped_area_acres=("cropped_area_acres", "sum"),
		)
		.reset_index()
	)

	summary_subfeature_stats = (
		summary_subfeatures.groupby(["spatial_fid"], dropna=False)
		.agg(subfeature_list=("_subfeature_text", unique_join))
		.reset_index()
	)
	summary_stats = summary_stats.merge(
		summary_subfeature_stats,
		on=["spatial_fid"],
		how="left",
	)

	summary = base.merge(
		summary_stats,
		on=["spatial_fid"],
		how="left",
	)
	summary["season_year"] = output_key
	summary["year"] = year
	summary["season"] = season
	summary["label"] = build_label_series(summary["survey_num"], summary["category"])
	summary["crop_count"] = pd.to_numeric(summary["crop_count"], errors="coerce").fillna(0).astype("Int64")
	summary["total_cropped_area_acres"] = pd.to_numeric(summary["total_cropped_area_acres"], errors="coerce")
	return summary[SUMMARY_COLUMNS + ["geometry"]].copy()


def remove_existing_output(file_path):
	if OVERWRITE_OUTPUT and file_path.exists():
		file_path.unlink()


def write_logs(output_key, unmatched_crop_records):
	unmatched_log = LOG_DIR / f"{output_key}_unmatched_crop_records.csv"
	if unmatched_crop_records.empty:
		if unmatched_log.exists():
			unmatched_log.unlink()
		return None

	LOG_DIR.mkdir(parents=True, exist_ok=True)
	unmatched_crop_records.to_csv(unmatched_log, index=False, encoding="utf-8-sig")
	return unmatched_log


def build_gpkg_for_crop_file(crop_file, base, crop_lookup, non_agri_meta, fallow_meta):
	year, season, output_key = parse_output_key_from_file_name(crop_file)
	print()
	print(f"Processing {year} {season.title()} file: {crop_file.name}")
	print("Step 1/5: Loading and preparing crop records...")
	crop_rows, input_record_count = load_and_prepare_crop_table(crop_file, crop_lookup)

	print("Step 2/5: Matching crop records and assigning unassigned polygons...")
	assignment, unmatched_crop_records, summary_subfeatures, assignment_stats = build_assignment_layer(
		base, crop_rows, year, season, output_key, non_agri_meta, fallow_meta
	)

	print("Step 3/5: Building summary-layer records...")
	summary = build_summary_layer(base, assignment, summary_subfeatures, year, season, output_key)
	print("Step 4/5: Writing unmatched crop-record log...")
	unmatched_log = write_logs(output_key, unmatched_crop_records)
	if unmatched_log is None:
		print("  No unmatched crop records; no log file created.")
	else:
		print(f"  Wrote unmatched-record log: {unmatched_log}")

	return {
		"output_key": output_key,
		"target_year": year,
		"target_season": season,
		"summary": summary,
		"assignment": assignment,
		"input_crop_record_count": input_record_count,
		"unmatched_crop_count": len(unmatched_crop_records),
		"unmatched_log": unmatched_log,
		"assignment_stats": assignment_stats,
		"excluded_land_crop_names": {
			name.casefold()
			for name in (non_agri_meta["crop_name"], fallow_meta["crop_name"])
			if name
		},
	}


def print_season_summary(result, base):
	assignment = result["assignment"]
	assignment_stats = result["assignment_stats"]
	polygon_count = len(base)
	assigned_polygon_count = assignment_stats["assigned_polygons"]
	unassigned_fallow_count = assignment_stats["unassigned_fallow_polygons"]
	unassigned_non_agri_count = assignment_stats["unassigned_non_agri_polygons"]
	unassigned_polygon_count = unassigned_fallow_count + unassigned_non_agri_count
	feature_counts = assignment.groupby("spatial_fid", dropna=False).size()
	mean_features = feature_counts.mean() if not feature_counts.empty else 0.0
	std_features = feature_counts.std(ddof=0) if not feature_counts.empty else 0.0

	cultivated_crop_rows = assignment.loc[assignment["cropped_area_acres"].gt(0)].copy()
	excluded_land_crop_names = result["excluded_land_crop_names"]
	cultivated_crop_rows = cultivated_crop_rows.loc[
		~cultivated_crop_rows["crop_name"]
		.astype("string")
		.str.strip()
		.str.casefold()
		.isin(excluded_land_crop_names)
	]
	crop_acreage = (
		cultivated_crop_rows.groupby("crop_name")["cropped_area_acres"]
		.sum()
		.sort_values(ascending=False)
		.head(5)
	)
	total_cultivated_area = cultivated_crop_rows["cropped_area_acres"].sum()

	print()
	print("=" * 72)
	print(f"Season processing summary: {result['target_year']} {result['target_season'].title()}")
	print("-" * 72)
	print(f"Input crop records: {result['input_crop_record_count']:,}")
	print(f"Unmatched crop records: {result['unmatched_crop_count']:,}")
	if result["unmatched_log"] is not None:
		print(f"Unmatched-record log: {result['unmatched_log']}")
	print(f"Polygons: {polygon_count:,}")
	print(f"Assigned polygons: {assigned_polygon_count:,}")
	if unassigned_polygon_count:
		fallow_proportion = unassigned_fallow_count / unassigned_polygon_count
		non_agri_proportion = unassigned_non_agri_count / unassigned_polygon_count
	else:
		fallow_proportion = 0.0
		non_agri_proportion = 0.0
	print(
		f"Unassigned polygons: {unassigned_polygon_count:,} "
		f"(fallow: {unassigned_fallow_count:,}, {fallow_proportion:.1%}; "
		f"non-agri: {unassigned_non_agri_count:,}, {non_agri_proportion:.1%})"
	)
	print(f"Unique crop names: {assignment['crop_name'].nunique():,}")
	print(f"Crop features generated: {len(assignment):,}")
	print(f"Features per polygon: {mean_features:.2f} +/- {std_features:.2f}")
	print("Top 5 crops by acreage:")
	if crop_acreage.empty:
		print("  None")
	else:
		for crop_name, acreage in crop_acreage.items():
			if pd.notna(total_cultivated_area) and total_cultivated_area > 0:
				area_proportion = acreage / total_cultivated_area
				area_text = f"{area_proportion:.1%} of cultivated area"
			else:
				area_text = "percentage unavailable (cultivated area is zero)"
			print(f"  {crop_name} - {acreage:,.2f} acres ({area_text})")
	print("=" * 72)
	print()


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	village_dict = load_json_dictionary(VILLAGE_DICT_FILE)
	crop_dict = load_json_dictionary(CROP_DICT_FILE)
	location_maps = build_location_maps(village_dict)
	crop_lookup = build_crop_lookup(crop_dict)
	non_agri_meta = get_non_agri_metadata(crop_lookup)
	fallow_meta = get_fallow_metadata(crop_lookup)
	base = prepare_spatial_base(SPATIAL_BASE_GPKG, SPATIAL_BASE_LAYER, location_maps)
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	crop_files = get_selected_crop_files(INPUT_DIR, TARGET_YEARS, TARGET_SEASONS)
	selected_years = sorted({int(year_value) for year_value in TARGET_YEARS})
	year_start = selected_years[0]
	year_end = selected_years[-1]
	summary_output_gpkg = OUTPUT_DIR / f"{year_start}-{year_end}_belle_survey_summary.gpkg"
	remove_existing_output(summary_output_gpkg)
	for year_value in selected_years:
		for season_value in TARGET_SEASONS:
			output_key = f"{year_value}_{str(season_value).strip().lower()}"
			remove_existing_output(build_crop_map_output_path(output_key))

	summary_frames = []
	for crop_file in crop_files:
		result = build_gpkg_for_crop_file(crop_file, base, crop_lookup, non_agri_meta, fallow_meta)
		summary_frames.append(result["summary"])
		crop_map_output_gpkg = build_crop_map_output_path(result["output_key"])
		print("Step 5/5: Writing crop assignment layer...")
		result["assignment"].to_file(
			crop_map_output_gpkg,
			layer=ASSIGNMENT_LAYER,
			driver="GPKG",
		)
		print(f"Wrote crop assignment layer: {crop_map_output_gpkg.name}")
		print_season_summary(result, base)

	if summary_frames:
		print("Writing combined summary layer...")
		combined_summary = gpd.GeoDataFrame(
			pd.concat(summary_frames, ignore_index=True),
			geometry="geometry",
			crs=base.crs,
		)
		combined_summary.to_file(summary_output_gpkg, layer=SUMMARY_LAYER, driver="GPKG")
		print(f"Wrote summary layer: {summary_output_gpkg.name}")
