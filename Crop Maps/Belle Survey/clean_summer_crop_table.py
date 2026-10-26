from pathlib import Path
import json
import re
import pandas as pd

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "Crop_Data_Summer"
INPUT_PATTERN = "*_summer_crop_data.csv"
OUTPUT_DIR = SCRIPT_DIR / "Crop_Data_Cleaned"
OUTPUT_TEMPLATE = "{year}_summer_crop_data_cleaned.csv"
VILLAGE_DICT_FILE = SCRIPT_DIR / "village_id_dict.json"
CROP_DICT_FILE = SCRIPT_DIR / "crop_dict.json"
CHUNK_SIZE = 200_000
OVERWRITE_OUTPUT = True
OUTPUT_ENCODING = "utf-8-sig"

ACRE_TO_HA = 0.40468564224

ALIAS_MAP = {
	"survey_no_text": ["Hissa Survey No", "Survey No"],
	"area_acre": ["acreage"],
	"crop_name": ["Crop Name (Standardised)"],
	"crop_detail_english": ["Crop Detail (English)"],
	"crop_detail_local": ["Crop Detail", "Crop"],
	"crop_category": ["Crop Category"],
	"kgis_village_id": ["KGISCode"],
	"source_village_id": ["site_code"],
}

REQUIRED_FIELDS = {
	"survey_no_text",
	"area_acre",
	"kgis_village_id",
}

OUTPUT_COLUMNS = [
	"year",
	"season",
	"survey_num",
	"survey_detail",
	"area_ha",
	"crop_name",
	"crop_detail",
	"crop_detail_kannada",
	"crop_category",
	"kgis_village_id",
	"source_village_id",
	"kgis_village_name",
	"district",
	"taluk",
	"hobli",
]


# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def normalize_header_name(header):
	return str(header).replace("\ufeff", "").strip().lower()


def normalize_text_series(series):
	return (
		series.astype("string")
		.str.replace(r"\s+", " ", regex=True)
		.str.strip()
		.replace("", pd.NA)
	)


def normalize_mixed_case_text(series):
	text = normalize_text_series(series)
	english_mask = text.str.match(r"^[A-Za-z][A-Za-z\s\.\-'/&()]*$", na=False)
	text.loc[english_mask] = (
		text.loc[english_mask]
		.str.lower()
		.str.title()
	)
	return text


def parse_parent_survey_number_from_survey_detail(series):
	survey_detail = normalize_text_series(series)
	return pd.to_numeric(
		survey_detail.str.extract(r"^\s*(\d+)")[0],
		errors="coerce",
	).astype("Int64")


def first_non_empty(*series_list):
	if not series_list:
		raise ValueError("first_non_empty requires at least one series")

	result = normalize_text_series(series_list[0])
	for series in series_list[1:]:
		result = result.fillna(normalize_text_series(series))
	return result


def get_column_lookup(input_file):
	header = pd.read_csv(input_file, nrows=0)
	return {normalize_header_name(column): column for column in header.columns}


def resolve_columns(column_lookup):
	resolved_columns = {}
	for canonical_name, aliases in ALIAS_MAP.items():
		matched_column = None
		for alias in aliases:
			lookup_key = normalize_header_name(alias)
			if lookup_key in column_lookup:
				matched_column = column_lookup[lookup_key]
				break

		if matched_column is None and canonical_name in REQUIRED_FIELDS:
			raise ValueError(
				f"Missing required column for '{canonical_name}'. Tried aliases: {aliases}"
			)

		resolved_columns[canonical_name] = matched_column

	return resolved_columns


def extract_year_from_file_name(input_file):
	match = re.search(r"(20\d{2})", input_file.name)
	if not match:
		raise ValueError(f"Could not parse year from file name: {input_file.name}")
	return int(match.group(1))


def get_series_or_na(chunk, column_name):
	if column_name is None:
		return pd.Series(pd.NA, index=chunk.index, dtype="string")
	return chunk[column_name]


def load_json_dictionary(dictionary_file):
	if not dictionary_file.exists():
		raise FileNotFoundError(f"Missing dictionary file: {dictionary_file}")
	with dictionary_file.open("r", encoding="utf-8") as file:
		return json.load(file)


def build_location_maps(village_dict):
	return {
		"kgis_village_name": {
			key: value.get("village")
			for key, value in village_dict.items()
		},
		"district": {
			key: value.get("district")
			for key, value in village_dict.items()
		},
		"taluk": {
			key: value.get("taluk")
			for key, value in village_dict.items()
		},
		"hobli": {
			key: value.get("hobli")
			for key, value in village_dict.items()
		},
	}


def build_crop_maps(crop_dict):
	return {
		"crop_detail": {
			key: value.get("crop_details")
			for key, value in crop_dict.items()
		},
		"crop_name": {
			key: value.get("crop_name")
			for key, value in crop_dict.items()
		},
		"crop_category": {
			key: value.get("crop_category")
			for key, value in crop_dict.items()
		},
	}


def clean_chunk(chunk, resolved_columns, year, location_maps, crop_maps):
	cleaned_chunk = pd.DataFrame(index=chunk.index)

	survey_detail = normalize_text_series(
		chunk[resolved_columns["survey_no_text"]]
	)
	crop_detail_kannada_source = normalize_text_series(
		get_series_or_na(chunk, resolved_columns["crop_detail_local"])
	)
	crop_detail_english = normalize_text_series(
		get_series_or_na(chunk, resolved_columns["crop_detail_english"])
	)
	crop_category = normalize_text_series(
		get_series_or_na(chunk, resolved_columns["crop_category"])
	)
	fallow_local_mask = crop_detail_kannada_source.str.lower().eq("fallow land")
	missing_english_meta_mask = crop_detail_english.isna() & crop_category.isna()
	summer_fallow_exception_mask = fallow_local_mask & missing_english_meta_mask

	crop_detail_kannada = crop_detail_kannada_source.mask(
		summer_fallow_exception_mask,
		"ಪಾಳು",
	)
	crop_name = get_series_or_na(chunk, resolved_columns["crop_name"])
	crop_detail_from_dict = crop_detail_kannada.map(crop_maps["crop_detail"])
	crop_name_from_dict = crop_detail_kannada.map(crop_maps["crop_name"])
	crop_category_from_dict = crop_detail_kannada.map(crop_maps["crop_category"])
	fallow_crop_detail = pd.Series(pd.NA, index=chunk.index, dtype="string")
	fallow_crop_name = pd.Series(pd.NA, index=chunk.index, dtype="string")
	fallow_crop_category = pd.Series(pd.NA, index=chunk.index, dtype="string")
	fallow_crop_detail.loc[summer_fallow_exception_mask] = "Fallow Land"
	fallow_crop_name.loc[summer_fallow_exception_mask] = "Fallow Land"
	fallow_crop_category.loc[summer_fallow_exception_mask] = (
		"Fallow / Non-Agricultural"
	)

	cleaned_chunk["year"] = year
	cleaned_chunk["season"] = "Summer"
	cleaned_chunk["survey_num"] = parse_parent_survey_number_from_survey_detail(
		survey_detail
	)
	cleaned_chunk["survey_detail"] = survey_detail
	cleaned_chunk["area_ha"] = (
		pd.to_numeric(chunk[resolved_columns["area_acre"]], errors="coerce")
		* ACRE_TO_HA
	)
	cleaned_chunk["crop_detail_kannada"] = crop_detail_kannada
	cleaned_chunk["crop_name"] = first_non_empty(
		crop_name,
		crop_name_from_dict,
		fallow_crop_name,
	)
	cleaned_chunk["crop_detail"] = first_non_empty(
		crop_detail_english,
		crop_detail_from_dict,
		fallow_crop_detail,
	)
	cleaned_chunk["crop_category"] = first_non_empty(
		crop_category,
		crop_category_from_dict,
		fallow_crop_category,
	)
	cleaned_chunk["kgis_village_id"] = normalize_text_series(
		chunk[resolved_columns["kgis_village_id"]]
	).str.replace(r"\.0$", "", regex=True)
	cleaned_chunk["source_village_id"] = first_non_empty(
		get_series_or_na(chunk, resolved_columns["source_village_id"]),
		cleaned_chunk["kgis_village_id"],
	).str.replace(r"\.0$", "", regex=True)

	kgis_village_ids = cleaned_chunk["kgis_village_id"]
	cleaned_chunk["kgis_village_name"] = normalize_mixed_case_text(
		kgis_village_ids.map(location_maps["kgis_village_name"])
	)
	cleaned_chunk["district"] = normalize_mixed_case_text(
		kgis_village_ids.map(location_maps["district"])
	)
	cleaned_chunk["taluk"] = normalize_mixed_case_text(
		kgis_village_ids.map(location_maps["taluk"])
	)
	cleaned_chunk["hobli"] = normalize_mixed_case_text(
		kgis_village_ids.map(location_maps["hobli"])
	)

	return cleaned_chunk[OUTPUT_COLUMNS]


def process_input_file(input_file, location_maps, crop_maps):
	year = extract_year_from_file_name(input_file)
	output_file = OUTPUT_DIR / OUTPUT_TEMPLATE.format(year=year)

	if OVERWRITE_OUTPUT and output_file.exists():
		output_file.unlink()

	column_lookup = get_column_lookup(input_file)
	resolved_columns = resolve_columns(column_lookup)
	selected_columns = [
		column
		for column in dict.fromkeys(resolved_columns.values())
		if column is not None
	]

	total_rows_written = 0
	write_header = True
	chunk_count = 0
	reader = pd.read_csv(
		input_file,
		usecols=selected_columns,
		dtype="string",
		chunksize=CHUNK_SIZE,
	)

	print(f"Processing {input_file.name} -> {output_file.name}...")
	for chunk in reader:
		chunk_count += 1
		cleaned_chunk = clean_chunk(
			chunk,
			resolved_columns,
			year,
			location_maps,
			crop_maps,
		)
		cleaned_chunk.to_csv(
			output_file,
			mode="a",
			index=False,
			header=write_header,
			encoding=OUTPUT_ENCODING,
		)
		write_header = False
		total_rows_written += len(cleaned_chunk)
		print(
			f"Processed chunk {chunk_count}: "
			f"{total_rows_written:,} rows written"
		)

	return output_file, total_rows_written


def build_summer_crop_reference_tables():
	input_files = sorted(INPUT_DIR.glob(INPUT_PATTERN))
	if not input_files:
		raise FileNotFoundError(
			f"No input files matched '{INPUT_PATTERN}' in {INPUT_DIR}"
		)
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	village_dict = load_json_dictionary(VILLAGE_DICT_FILE)
	crop_dict = load_json_dictionary(CROP_DICT_FILE)
	location_maps = build_location_maps(village_dict)
	crop_maps = build_crop_maps(crop_dict)
	written_outputs = []
	for input_file in input_files:
		output_file, rows_written = process_input_file(
			input_file,
			location_maps,
			crop_maps,
		)
		written_outputs.append((output_file, rows_written))

	return written_outputs


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	results = build_summer_crop_reference_tables()
	for output_file, rows_written in results:
		print(
			f"Wrote {rows_written:,} cleaned rows to {output_file.name}"
		)