from contextlib import closing
from pathlib import Path
import json
import re
import sqlite3
import pandas as pd

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_SOURCES = [
	(SCRIPT_DIR / "Crop_Data_Kharif-Rabi", "*_kharif-rabi_crop_data.csv")
]
SPATIAL_BASE_GPKG = SCRIPT_DIR / "spatial_base.gpkg"
SPATIAL_BASE_LAYER = "spatial_base"
CHUNK_SIZE = 200_000
OUTPUT_ENCODING = "utf-8-sig"

OUTPUT_SOURCE = SCRIPT_DIR / "source_village_lookup.csv"
OUTPUT_SPATIAL = SCRIPT_DIR / "spatial_village_lookup.csv"
OUTPUT_DICT = SCRIPT_DIR / "village_code_dict.json"
OUTPUT_CANONICAL = SCRIPT_DIR / "village_dict_canonical.csv"
OUTPUT_DICT_CONFLICTS = SCRIPT_DIR / "village_dict_conflicts.csv"

NAME_FIELDS = ["village", "district", "taluk", "hobli"]
SOURCE_COLUMNS = ["season", "kgis_code", "site_code"] + NAME_FIELDS
DICT_CONFLICT_COLUMNS = [
	"spatial_code", "kgis_code", "site_code", "season"
] + ["spatial_village"] + NAME_FIELDS + [
	"conflict_type", "field", "canonical_value", "variant"
]

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def normalize_text(value):
	if value is None or pd.isna(value):
		return None
	text = re.sub(r"\s+", " ", str(value).replace("\ufeff", "")).strip()
	return text or None


def normalize_code(value):
	text = normalize_text(value)
	if text is not None and text.endswith(".0"):
		return text[:-2]
	return text


def normalize_name(value):
	text = normalize_text(value)
	if text is None:
		return None
	return re.sub(r"\s+", "", text).casefold()


def read_input_files():
	input_files = sorted(
		path
		for directory, pattern in INPUT_SOURCES
		for path in directory.glob(pattern)
	)
	if not input_files:
		raise FileNotFoundError(f"No input files matched the configured sources: {INPUT_SOURCES}")
	return input_files


def source_file_columns(input_file):
	columns = pd.read_csv(input_file, nrows=0, encoding=OUTPUT_ENCODING).columns
	required = ["KGISCode", "site_code", "District", "Taluk", "Hobli"]
	missing = [column for column in required if column not in columns]
	village_columns = [
		column for column in ("KGIS_village_name", "Village") if column in columns
	]
	if not village_columns:
		missing.append("KGIS_village_name or Village")

	summer_match = re.fullmatch(r"(\d{4})_summer_crop_data\.csv", input_file.name, re.I)
	if summer_match:
		file_season = f"summer_{summer_match.group(1)}"
	else:
		file_season = None
		for column in ("season", "season_key"):
			if column not in columns:
				missing.append(column)
			else:
				required.append(column)
	if missing:
		raise ValueError(f"Missing required columns in {input_file.name}: {missing}")
	return list(dict.fromkeys(required + village_columns)), village_columns, file_season


def parse_season(season, season_key, input_file):
	season_text = normalize_text(season)
	key_text = normalize_text(season_key)
	season_match = re.search(r"\b(kharif|rabi|summer)\b", season_text or "", re.I)
	year_match = re.match(r"^(\d{4})(?:_|$)", key_text or "")
	if season_match is None or year_match is None:
		raise ValueError(
			f"Invalid season or season_key in {input_file.name}: "
			f"{season_text!r}, {key_text!r}"
		)
	return f"{season_match.group(1).lower()}_{year_match.group(1)}"


def build_source_lookup(input_files):
	first_rows = {}
	variant_rows = {}
	for input_file in input_files:
		usecols, village_columns, file_season = source_file_columns(input_file)
		print(f"Processing {input_file.name}...")
		reader = pd.read_csv(
			input_file, usecols=usecols, dtype="string",
			chunksize=CHUNK_SIZE, encoding=OUTPUT_ENCODING,
		)
		for chunk in reader:
			for values in chunk.drop_duplicates().itertuples(index=False, name=None):
				row = dict(zip(chunk.columns, values))
				season = file_season or parse_season(
					row["season"], row["season_key"], input_file
				)
				village = next((
					normalize_text(row[column])
					for column in village_columns
					if normalize_text(row[column]) is not None
				), None)
				entry = {
					"season": season,
					"kgis_code": normalize_code(row["KGISCode"]),
					"site_code": normalize_code(row["site_code"]),
					"village": village,
					"district": normalize_text(row["District"]),
					"taluk": normalize_text(row["Taluk"]),
					"hobli": normalize_text(row["Hobli"]),
				}
				identity = (season, entry["site_code"], entry["kgis_code"])
				first_rows.setdefault(identity, entry)
				signature = tuple(entry[field] for field in NAME_FIELDS)
				variant_rows.setdefault(identity, {}).setdefault(signature, entry)

	return list(first_rows.values()), variant_rows


def build_spatial_lookup(spatial_file, layer):
	if not spatial_file.exists():
		raise FileNotFoundError(f"Spatial base GeoPackage not found: {spatial_file}")
	uri = f"{spatial_file.resolve().as_uri()}?mode=ro"
	with closing(sqlite3.connect(uri, uri=True)) as connection:
		columns = {
			row[1] for row in connection.execute(f'PRAGMA table_info("{layer}")')
		}
		if not {"kgis_code", "village"}.issubset(columns):
			raise ValueError(f"Layer {layer!r} must contain kgis_code and village")
		rows = connection.execute(
			f'SELECT DISTINCT kgis_code, village FROM "{layer}"'
		)
		unique = {}
		for code, village in rows:
			entry = (normalize_code(code), normalize_text(village))
			unique.setdefault(entry, None)
	return [
		{"kgis_code": code, "village": village}
		for code, village in unique
	]


SEASON_ORDER = {"kharif": 0, "rabi": 1, "summer": 2}


def season_sort_key(season):
	name, year = season.split("_")
	return int(year), SEASON_ORDER[name]


def format_name(value):
	text = normalize_text(value)
	return text.lower().capitalize() if text is not None else None


def build_canonical_lookup(source_rows, variant_rows, spatial_rows):
	map_codes = {
		row["kgis_code"]
		for row in spatial_rows if row["kgis_code"] is not None
	}
	spatial_names_by_code = {}
	for spatial_row in spatial_rows:
		code = spatial_row["kgis_code"]
		name = spatial_row["village"]
		normalized = normalize_name(name)
		if code is None or normalized is None:
			continue
		spatial_names_by_code.setdefault(code, {}).setdefault(normalized, name)

	matches = []
	conflicts = []
	for sequence, row in enumerate(source_rows):
		code = row["kgis_code"]
		if code not in map_codes:
			conflicts.append({
				**row, "spatial_code": None, "spatial_village": None,
				"conflict_type": "no KGIS code match",
				"field": "kgis_code", "canonical_value": None, "variant": code,
			})
			continue
		matches.append((sequence, row, code))

	latest = {}
	for sequence, row, code in matches:
		identity = (row["season"], row["site_code"], row["kgis_code"])
		for variant in variant_rows[identity].values():
			order = (season_sort_key(row["season"]), sequence)
			for field in NAME_FIELDS:
				if variant[field] is None:
					continue
				best = latest.setdefault(code, {}).get(field)
				if best is None or order >= best[0]:
					latest[code][field] = (order, format_name(variant[field]))
	names_by_code = {
		code: {field: fields.get(field, (None, None))[1] for field in NAME_FIELDS}
		for code, fields in latest.items()
	}

	dictionary = {}
	seen_variants = set()
	for sequence, row, code in matches:
		names = names_by_code[code]
		dictionary[code] = names
		identity = (row["season"], row["site_code"], code)
		for variant in variant_rows[identity].values():
			for field in NAME_FIELDS:
				value = variant[field]
				if value is None or normalize_name(value) == normalize_name(names[field]):
					continue
				key = (code, field, normalize_name(value))
				if key in seen_variants:
					continue
				seen_variants.add(key)
				conflicts.append({
					**variant, "spatial_code": code,
					"spatial_village": "; ".join(
						spatial_names_by_code.get(code, {}).values()
					) or None,
					"conflict_type": "location name variant",
					"field": field,
					"canonical_value": names[field], "variant": value,
				})
	return dictionary, conflicts


def write_csv(path, rows, columns):
	pd.DataFrame(rows, columns=columns).to_csv(
		path, index=False, encoding=OUTPUT_ENCODING
	)


def write_outputs(source_rows, spatial_rows, dictionary, dict_conflicts):
	write_csv(OUTPUT_SOURCE, source_rows, SOURCE_COLUMNS)
	write_csv(OUTPUT_SPATIAL, spatial_rows, ["kgis_code", "village"])
	with OUTPUT_DICT.open("w", encoding="utf-8") as file:
		json.dump(dictionary, file, ensure_ascii=False, indent=2)
	canonical_rows = [
		{"kgis_code": kgis_code, **values}
		for kgis_code, values in dictionary.items()
	]
	write_csv(
		OUTPUT_CANONICAL, canonical_rows,
		["kgis_code"] + NAME_FIELDS,
	)
	write_csv(OUTPUT_DICT_CONFLICTS, dict_conflicts, DICT_CONFLICT_COLUMNS)

# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	input_files = read_input_files()
	source_rows, variant_rows = build_source_lookup(input_files)
	spatial_rows = build_spatial_lookup(SPATIAL_BASE_GPKG, SPATIAL_BASE_LAYER)
	dictionary, dict_conflicts = build_canonical_lookup(
		source_rows, variant_rows, spatial_rows
	)
	write_outputs(source_rows, spatial_rows, dictionary, dict_conflicts)
	print(f"Input files: {len(input_files):,}")
	print(f"Source lookup rows: {len(source_rows):,}")
	print(f"Spatial lookup rows: {len(spatial_rows):,}")
	print(f"Dictionary source codes: {len(dictionary):,}")
	print(f"Dictionary conflicts: {len(dict_conflicts):,}")
