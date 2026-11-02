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
INPUT_DIR = SCRIPT_DIR / "Crop_Data_Kharif-Rabi"
INPUT_PATTERN = "*_kharif-rabi_crop_data.csv"
CHUNK_SIZE = 200_000
OUTPUT_ENCODING = "utf-8-sig"

KGIS_CODE_COLUMN = "KGISCode"
SITE_CODE_COLUMN = "site_code"
VILLAGE_COLUMN = "KGIS_village_name"
NAME_FIELDS = ["village", "district", "taluk", "hobli"]
FIELD_TO_SOURCE_COLUMN = {
	"village": VILLAGE_COLUMN,
	"district": "District",
	"taluk": "Taluk",
	"hobli": "Hobli",
}

SPATIAL_BASE_GPKG = SCRIPT_DIR / "spatial_base.gpkg"
SPATIAL_BASE_LAYER = "spatial_base"
SPATIAL_BASE_CODE_COLUMN = "kgis_code"

OUTPUT_ATTRIBUTES = SCRIPT_DIR / "spatial_base_attributes.csv"
OUTPUT_DICT = SCRIPT_DIR / "village_code_dict.json"
OUTPUT_KEY = SCRIPT_DIR / "village_code_canonical.csv"
OUTPUT_CONFLICTS = SCRIPT_DIR / "village_code_conflicts.csv"

CONFLICT_COLUMNS = [
	"map_code",
	"kgis_code",
	"site_code",
	"conflict_type",
	"field",
	"canonical_value",
	"variants",
]

CONFLICT_VARIANTS = "multiple variants"
CONFLICT_DISTINCT_GEOMETRIES = "kgis and site codes match different map codes"
CONFLICT_NO_MATCH = "no map code match"

ENGLISH_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9\s\.\-'/&,()]*$")

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def normalize_text(value):
	if value is None or pd.isna(value):
		return None

	text = str(value).replace("\ufeff", "")
	text = re.sub(r"\s+", " ", text).strip()
	if text == "":
		return None
	return text


def normalize_code(value):
	text = normalize_text(value)
	if text is not None and text.endswith(".0"):
		text = text[:-2]
	return text


def normalize_for_match(value):
	text = normalize_text(value)
	if text is None:
		return None
	return text.casefold()


def is_english_name(value):
	text = normalize_text(value)
	if text is None:
		return False
	return bool(ENGLISH_NAME_PATTERN.fullmatch(text))


def read_input_files():
	input_files = sorted(INPUT_DIR.glob(INPUT_PATTERN))
	if not input_files:
		raise FileNotFoundError(
			f"No files matched {INPUT_PATTERN!r} in {INPUT_DIR}"
		)
	return input_files


def export_spatial_base_attributes():
	if not SPATIAL_BASE_GPKG.exists():
		raise FileNotFoundError(
			f"Spatial base GeoPackage not found: {SPATIAL_BASE_GPKG}"
		)

	uri = f"{SPATIAL_BASE_GPKG.resolve().as_uri()}?mode=ro"
	with closing(sqlite3.connect(uri, uri=True)) as connection:
		geometry_row = connection.execute(
			"SELECT column_name FROM gpkg_geometry_columns WHERE table_name = ?",
			(SPATIAL_BASE_LAYER,),
		).fetchone()
		geometry_column = geometry_row[0] if geometry_row else None

		columns = [
			row[1]
			for row in connection.execute(
				f'PRAGMA table_info("{SPATIAL_BASE_LAYER}")'
			)
			if row[1] != geometry_column
		]
		select_list = ", ".join(f'"{column}"' for column in columns)
		query = f'SELECT {select_list} FROM "{SPATIAL_BASE_LAYER}"'

		feature_count = 0
		header = True
		for chunk in pd.read_sql_query(query, connection, chunksize=CHUNK_SIZE):
			chunk.to_csv(
				OUTPUT_ATTRIBUTES,
				mode="w" if header else "a",
				header=header,
				index=False,
				encoding=OUTPUT_ENCODING if header else "utf-8",
			)
			header = False
			feature_count += len(chunk)

	return feature_count


def read_map_codes():
	attributes = pd.read_csv(
		OUTPUT_ATTRIBUTES,
		usecols=[SPATIAL_BASE_CODE_COLUMN],
		dtype="string",
		encoding=OUTPUT_ENCODING,
	)
	codes = {
		normalize_code(value)
		for value in attributes[SPATIAL_BASE_CODE_COLUMN].dropna().unique()
	}
	codes.discard(None)
	return codes


def build_canonical_lookup(input_files, map_codes):
	canonical: dict[str, dict[str, str | None]] = {
		code: {field: None for field in NAME_FIELDS} for code in map_codes
	}
	# casefolded spelling -> first spelling encountered, per map code and field
	variants = {
		code: {field: {} for field in NAME_FIELDS} for code in map_codes
	}
	source_codes = {code: {"kgis": set(), "site": set()} for code in map_codes}
	# (kgis_code, site_code) -> village names seen, in encounter order
	pair_villages = {}

	use_columns = list(dict.fromkeys(
		[KGIS_CODE_COLUMN, SITE_CODE_COLUMN] + list(FIELD_TO_SOURCE_COLUMN.values())
	))

	for input_file in input_files:
		reader = pd.read_csv(
			input_file,
			usecols=use_columns,
			dtype="string",
			chunksize=CHUNK_SIZE,
			encoding=OUTPUT_ENCODING,
		)
		print(f"Processing {input_file.name}...")

		for chunk in reader:
			for row in chunk.drop_duplicates().itertuples(index=False, name=None):
				row_data = dict(zip(chunk.columns, row))
				kgis_code = normalize_code(row_data[KGIS_CODE_COLUMN])
				site_code = normalize_code(row_data[SITE_CODE_COLUMN])
				if kgis_code is None and site_code is None:
					continue

				field_values = {
					field: normalize_text(row_data[column])
					for field, column in FIELD_TO_SOURCE_COLUMN.items()
				}

				kgis_matches = kgis_code in map_codes
				site_matches = site_code in map_codes

				if (
					kgis_matches
					and site_matches
					and kgis_code != site_code
				) or not (kgis_matches or site_matches):
					village_names = pair_villages.setdefault(
						(kgis_code, site_code), {}
					)
					village = field_values["village"]
					if is_english_name(village):
						village_names.setdefault(normalize_for_match(village), village)

				matched_codes = {
					code
					for code in (kgis_code, site_code)
					if code is not None and code in map_codes
				}
				for map_code in matched_codes:
					if kgis_code is not None:
						source_codes[map_code]["kgis"].add(kgis_code)
					if site_code is not None:
						source_codes[map_code]["site"].add(site_code)

					for field_name, field_value in field_values.items():
						if not is_english_name(field_value):
							continue

						if canonical[map_code][field_name] is None:
							canonical[map_code][field_name] = field_value

						variants[map_code][field_name].setdefault(
							normalize_for_match(field_value), field_value
						)

	return canonical, variants, source_codes, pair_villages


def build_conflict_report(
	canonical, variants, source_codes, pair_villages, map_codes
):
	conflicts = []

	for map_code in sorted(canonical):
		for field_name in NAME_FIELDS:
			spellings = list(variants[map_code][field_name].values())
			if len(spellings) <= 1:
				continue

			conflicts.append({
				"map_code": map_code,
				"kgis_code": ", ".join(sorted(source_codes[map_code]["kgis"])),
				"site_code": ", ".join(sorted(source_codes[map_code]["site"])),
				"conflict_type": CONFLICT_VARIANTS,
				"field": field_name,
				"canonical_value": canonical[map_code][field_name],
				"variants": " | ".join(spellings),
			})

	for kgis_code, site_code in sorted(
		pair_villages, key=lambda pair: (pair[0] or "", pair[1] or "")
	):
		village_names = list(pair_villages[(kgis_code, site_code)].values())
		kgis_matches = kgis_code in map_codes
		site_matches = site_code in map_codes

		if kgis_matches and site_matches:
			matched = [kgis_code, site_code]
			conflict_type = CONFLICT_DISTINCT_GEOMETRIES
			map_code = ", ".join(matched)
			canonical_value = " | ".join(
				f"{code}: {canonical[code]['village']}" for code in matched
			)
			difference = [
				f"kgis_code {kgis_code} and site_code {site_code} "
				f"each match a different map code"
			] + [f"source village: {name}" for name in village_names]
		else:
			conflict_type = CONFLICT_NO_MATCH
			map_code = None
			canonical_value = None
			difference = [
				f"neither kgis_code {kgis_code} nor site_code {site_code} "
				f"matches a map code"
			] + [f"source village: {name}" for name in village_names]

		conflicts.append({
			"map_code": map_code,
			"kgis_code": kgis_code,
			"site_code": site_code,
			"conflict_type": conflict_type,
			"field": "village",
			"canonical_value": canonical_value,
			"variants": " | ".join(difference),
		})

	return conflicts


def write_outputs(canonical, conflicts):
	saved = {
		code: names
		for code, names in canonical.items()
		if any(names[field] is not None for field in NAME_FIELDS)
	}

	with OUTPUT_DICT.open("w", encoding="utf-8") as file:
		json.dump(saved, file, ensure_ascii=False, indent=2)

	canonical_rows = []
	for map_code in sorted(saved):
		row = {"map_code": map_code}
		for field in NAME_FIELDS:
			row[field] = saved[map_code][field]
		canonical_rows.append(row)

	pd.DataFrame(canonical_rows, columns=["map_code"] + NAME_FIELDS).to_csv(
		OUTPUT_KEY,
		index=False,
		encoding=OUTPUT_ENCODING,
	)

	pd.DataFrame(conflicts, columns=CONFLICT_COLUMNS).to_csv(
		OUTPUT_CONFLICTS,
		index=False,
		encoding=OUTPUT_ENCODING,
	)

	return saved


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	input_files = read_input_files()

	feature_count = export_spatial_base_attributes()
	print(f"Features in spatial map: {feature_count:,}")
	print(f"Attribute table saved: {OUTPUT_ATTRIBUTES.name}")

	map_codes = read_map_codes()
	print(f"Unique map codes: {len(map_codes):,}")

	canonical, variants, source_codes, pair_villages = build_canonical_lookup(
		input_files, map_codes
	)
	conflicts = build_conflict_report(
		canonical, variants, source_codes, pair_villages, map_codes
	)
	saved = write_outputs(canonical, conflicts)

	print(f"Input files: {len(input_files)}")
	print(f"Canonical IDs saved: {len(saved):,}")
	print(f"Map codes with no source match: {len(map_codes) - len(saved):,}")
	print(f"Conflicts saved: {len(conflicts):,}")
	print(f"Canonical dictionary: {OUTPUT_DICT.name}")
	print(f"Canonical table: {OUTPUT_KEY.name}")
	print(f"Conflict report: {OUTPUT_CONFLICTS.name}")
