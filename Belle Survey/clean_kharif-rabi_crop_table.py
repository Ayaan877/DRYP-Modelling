from pathlib import Path
import json
import pandas as pd

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "Crop_Data_Kharif-Rabi"
INPUT_PATTERN = "*_kharif-rabi_crop_data.csv"
OUTPUT_DIR = SCRIPT_DIR / "Crop_Data_Cleaned"
OUTPUT_TEMPLATE = "{year}_{season}_crop_data_cleaned.csv"
VILLAGE_DICT_FILE = SCRIPT_DIR / "village_id_dict.json"
CROP_DICT_FILE = SCRIPT_DIR / "crop_dict.json"
CHUNK_SIZE = 200_000
OVERWRITE_OUTPUT = True
OUTPUT_ENCODING = "utf-8-sig"

ACRE_TO_HA = 0.40468564224

ALIAS_MAP = {
	"season_key": ["season_key"],
	"season_raw": ["season"],
	"survey_no_text": ["Survey No"],
	"extent": ["Extent"],
	"crop_detail_kannada": ["Crop Detail"],
	"kgis_village_id": ["KGISCode"],
	"source_village_id": ["site_code"],
	"owner_name": ["Owner"],
	"owner_no": ["Owner No"],
	"land_code": ["Land Code"],
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
	"owner_name",
	"owner_no",
	"land_code",
]

SEASON_VALUES = {
	"kharif": "Kharif",
	"rabi": "Rabi",
	"summer": "Summer",
}


# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def normalize_header_name(header):
	return str(header).replace("\ufeff", "").strip().lower()


def get_column_lookup(input_file):
	header = pd.read_csv(input_file, nrows=0)
	return {normalize_header_name(column): column for column in header.columns}


def resolve_required_columns(column_lookup, ALIAS_MAP):
	resolved_columns = {}
	for canonical_name, aliases in ALIAS_MAP.items():
		matched_column = None
		for alias in aliases:
			lookup_key = normalize_header_name(alias)
			if lookup_key in column_lookup:
				matched_column = column_lookup[lookup_key]
				break

		if matched_column is None:
			raise ValueError(
				f"Missing required column for '{canonical_name}'. Tried aliases: {aliases}"
			)

		resolved_columns[canonical_name] = matched_column

	return resolved_columns


def normalize_text_series(series):
	return (
		series.astype("string")
		.str.replace(r"\s+", " ", regex=True)
		.str.strip()
		.replace("", pd.NA)
	)


def get_series_or_na(chunk, column_name):
	if column_name is None:
		return pd.Series(pd.NA, index=chunk.index, dtype="string")
	return chunk[column_name]


def parse_year_from_season_key(series):
	year_text = normalize_text_series(series)
	return pd.to_numeric(
		year_text.str.extract(r"^\s*(\d{4})")[0],
		errors="coerce",
	).astype("Int64")


def parse_season(series):
	season_text = normalize_text_series(series).str.lower()
	season_key = season_text.str.extract(r"(kharif|rabi|summer)")[0]
	return season_key.map(SEASON_VALUES).astype("string")


def parse_parent_survey_number_from_survey_detail(series):
	survey_detail = normalize_text_series(series)
	return pd.to_numeric(
		survey_detail.str.extract(r"^\s*(\d+)")[0],
		errors="coerce",
	).astype("Int64")


def parse_extent_to_hectares(series):
	extent = normalize_text_series(series)
	parts = extent.str.extract(
		r"^\s*(\d+)(?:-(\d+))?(?:-(\d+(?:\.\d+)?))?\s*$"
	)
	acres = pd.to_numeric(parts[0], errors="coerce")
	gunthas = pd.to_numeric(parts[1], errors="coerce").fillna(0)
	annas = pd.to_numeric(parts[2], errors="coerce").fillna(0)
	total_acres = acres + (gunthas / 40) + (annas / 1600)
	total_acres = total_acres.where(acres.notna(), pd.NA)
	return total_acres * ACRE_TO_HA


def normalize_output_year(year_value):
	year_number = pd.to_numeric(str(year_value).strip(), errors="coerce")
	if pd.isna(year_number):
		raise ValueError(f"Invalid year value for output file name: {year_value!r}")
	return int(year_number)


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


def clean_chunk(chunk, resolved_columns, location_maps, crop_maps):
	cleaned_chunk = pd.DataFrame()
	survey_detail = normalize_text_series(chunk[resolved_columns["survey_no_text"]])
	crop_detail_kannada = normalize_text_series(chunk[resolved_columns["crop_detail_kannada"]])
	crop_detail_from_dict = crop_detail_kannada.map(crop_maps["crop_detail"])
	crop_name_from_dict = crop_detail_kannada.map(crop_maps["crop_name"])
	crop_category_from_dict = crop_detail_kannada.map(crop_maps["crop_category"])

	cleaned_chunk["year"] = parse_year_from_season_key(
		chunk[resolved_columns["season_key"]]
	)
	cleaned_chunk["season"] = parse_season(chunk[resolved_columns["season_raw"]])
	cleaned_chunk["survey_num"] = parse_parent_survey_number_from_survey_detail(
		survey_detail
	)
	cleaned_chunk["survey_detail"] = survey_detail
	cleaned_chunk["area_ha"] = parse_extent_to_hectares(
		chunk[resolved_columns["extent"]]
	)
	cleaned_chunk["crop_detail_kannada"] = crop_detail_kannada
	cleaned_chunk["crop_detail"] = normalize_text_series(crop_detail_from_dict)
	cleaned_chunk["crop_name"] = normalize_text_series(crop_name_from_dict)
	cleaned_chunk["crop_category"] = normalize_text_series(crop_category_from_dict)
	cleaned_chunk["kgis_village_id"] = normalize_text_series(
		chunk[resolved_columns["kgis_village_id"]]
	).str.replace(r"\.0$", "", regex=True)
	cleaned_chunk["source_village_id"] = normalize_text_series(
		get_series_or_na(chunk, resolved_columns["source_village_id"])
	).str.replace(r"\.0$", "", regex=True)

	kgis_village_ids = cleaned_chunk["kgis_village_id"]
	cleaned_chunk["kgis_village_name"] = normalize_text_series(
		kgis_village_ids.map(location_maps["kgis_village_name"])
	)
	cleaned_chunk["district"] = normalize_text_series(
		kgis_village_ids.map(location_maps["district"])
	)
	cleaned_chunk["taluk"] = normalize_text_series(
		kgis_village_ids.map(location_maps["taluk"])
	)
	cleaned_chunk["hobli"] = normalize_text_series(
		kgis_village_ids.map(location_maps["hobli"])
	)

	cleaned_chunk["owner_name"] = (
		chunk[resolved_columns["owner_name"]]
		.astype("string")
		.str.strip()
		.replace("", pd.NA)
	)
	cleaned_chunk["owner_no"] = normalize_text_series(
		chunk[resolved_columns["owner_no"]]
	).str.replace(r"\.0$", "", regex=True)
	cleaned_chunk["land_code"] = normalize_text_series(
		chunk[resolved_columns["land_code"]]
	).str.replace(r"\.0$", "", regex=True)

	return cleaned_chunk[OUTPUT_COLUMNS]


def count_in_scope_rows(input_file, resolved_columns):
	# Rows that should reach the output: parseable year and Kharif/Rabi season.
	season_columns = list(
		dict.fromkeys(
			[resolved_columns["season_key"], resolved_columns["season_raw"]]
		)
	)
	in_scope_rows = 0
	for chunk in pd.read_csv(
		input_file,
		usecols=season_columns,
		dtype="string",
		chunksize=CHUNK_SIZE,
	):
		year = parse_year_from_season_key(chunk[resolved_columns["season_key"]])
		season = parse_season(chunk[resolved_columns["season_raw"]])
		in_scope_rows += int((year.notna() & season.isin(["Kharif", "Rabi"])).sum())
	return in_scope_rows


def deduplicate_output_file(output_file):
	cleaned = pd.read_csv(
		output_file,
		dtype="string",
		keep_default_na=False,
		encoding=OUTPUT_ENCODING,
	)
	before = len(cleaned)
	deduped = cleaned.drop_duplicates()
	removed = before - len(deduped)
	if removed:
		deduped.to_csv(output_file, index=False, encoding=OUTPUT_ENCODING)
	return removed


def process_input_file(
	input_file,
	resolved_columns,
	location_maps,
	crop_maps,
	created_output_files,
):
	rows_written_by_output = {}
	write_header_by_output = {}
	chunk_count = 0
	selected_columns = list(dict.fromkeys(resolved_columns.values()))
	reader = pd.read_csv(
		input_file,
		usecols=selected_columns,
		dtype="string",
		chunksize=CHUNK_SIZE,
	)

	print(f"Processing {input_file.name}...")
	for chunk in reader:
		chunk_count += 1
		cleaned_chunk = clean_chunk(
			chunk,
			resolved_columns,
			location_maps,
			crop_maps,
		)

		chunk_output = cleaned_chunk.dropna(subset=["year", "season"])
		chunk_output = chunk_output[
			chunk_output["season"].isin(["Kharif", "Rabi"])
		]

		for (year_value, season_value), season_chunk in chunk_output.groupby(
			["year", "season"],
			dropna=False,
		):
			season_key = str(season_value).strip().lower()
			output_year = normalize_output_year(year_value)
			output_file = OUTPUT_DIR / OUTPUT_TEMPLATE.format(
				year=output_year,
				season=season_key,
			)

			if OVERWRITE_OUTPUT and output_file not in created_output_files:
				if output_file.exists():
					output_file.unlink()
				created_output_files.add(output_file)

			if output_file not in write_header_by_output:
				write_header_by_output[output_file] = True
				rows_written_by_output[output_file] = 0

			season_chunk.to_csv(
				output_file,
				mode="a",
				index=False,
				header=write_header_by_output[output_file],
				encoding=OUTPUT_ENCODING,
			)
			write_header_by_output[output_file] = False
			rows_written_by_output[output_file] += len(season_chunk)

		print(
			f"Processed chunk {chunk_count}: "
			f"{sum(rows_written_by_output.values()):,} rows written "
		)

	return rows_written_by_output


def build_crop_reference_tables():
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
	created_output_files = set()
	rows_written_by_output = {}

	for input_file in input_files:
		column_lookup = get_column_lookup(input_file)
		resolved_columns = resolve_required_columns(column_lookup, ALIAS_MAP)
		file_rows_written = process_input_file(
			input_file,
			resolved_columns,
			location_maps,
			crop_maps,
			created_output_files,
		)

		expected_rows = count_in_scope_rows(input_file, resolved_columns)
		written_rows = sum(file_rows_written.values())
		if written_rows != expected_rows:
			raise RuntimeError(
				f"Row count mismatch for {input_file.name}: "
				f"{expected_rows:,} Kharif/Rabi input rows vs {written_rows:,} written"
			)

		if not file_rows_written:
			print(
				f"No Kharif/Rabi rows found in {input_file.name}; "
				"no output files were written."
			)
			continue

		for output_file, rows_written in file_rows_written.items():
			rows_written_by_output[output_file] = (
				rows_written_by_output.get(output_file, 0) + rows_written
			)

	duplicates_removed_by_output = {}
	for output_file in rows_written_by_output:
		duplicates_removed_by_output[output_file] = deduplicate_output_file(output_file)

	return rows_written_by_output, duplicates_removed_by_output


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	rows_written_by_output, duplicates_removed_by_output = build_crop_reference_tables()

	total_rows = 0
	total_duplicates_removed = 0
	for output_file in sorted(rows_written_by_output.keys()):
		rows_written = rows_written_by_output[output_file]
		total_rows += rows_written
		removed = duplicates_removed_by_output.get(output_file, 0)
		total_duplicates_removed += removed
		print(f"Wrote {rows_written:,} cleaned rows to {output_file.name}")
		print(f"Removed {removed:,} duplicate rows from {output_file.name}")

	print(f"Total rows written: {total_rows:,}")
	print(f"Total duplicate rows removed: {total_duplicates_removed:,}")
