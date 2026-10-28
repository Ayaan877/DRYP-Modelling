from collections import OrderedDict
from pathlib import Path
import fiona
from tqdm import tqdm

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_GPKG = SCRIPT_DIR / "Chikkaballapura_surveys.gpkg"
INPUT_LAYER = "cadastral_Chikkaballapura_district"
OUTPUT_GPKG = SCRIPT_DIR / "spatial_base.gpkg"
OUTPUT_LAYER = "spatial_base"
OVERWRITE_OUTPUT = True

MAP_FIELDS = {
	"feature_id": "fid",
	"survey_num": "surveynumberi",
	"category": "Category",
	"area_sq_m": "SHAPE.STArea()",
	"kgis_village_code": "KGISVillageCode",
	"kgis_village_name": "village",
}

OUTPUT_COLUMNS = [
	"fid",
	"survey_num",
	"category",
	"area_ha",
	"kgis_village_code",
	"kgis_village_name",
]

SQUARE_METERS_PER_HECTARE = 10_000

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def normalize_text(value):
	if value is None:
		return None
	text = str(value).strip()
	if text == "":
		return None
	return " ".join(text.split())


def normalize_village_code(value):
	text = normalize_text(value)
	if text is None:
		return None
	if text.endswith(".0"):
		text = text[:-2]
	return text


def parse_survey_number(value):
	if value is None:
		return None
	text = normalize_text(value)
	if text is None:
		return None
	digits = []
	for character in text:
		if character.isdigit():
			digits.append(character)
		else:
			break
	if not digits:
		return None
	return int("".join(digits))


def parse_area_sq_m(value):
	if value is None:
		return 0.0
	try:
		return float(value)
	except (TypeError, ValueError):
		text = normalize_text(value)
		if text is None:
			return 0.0
		return float(text.replace(",", ""))


def get_map_field(field_key):
	if field_key not in MAP_FIELDS:
		raise ValueError(f"Missing map field configuration for key: {field_key}")
	return MAP_FIELDS[field_key]


def get_required_fields():
	feature_id_field = get_map_field("feature_id")
	required_fields = [
		get_map_field("survey_num"),
		get_map_field("category"),
		get_map_field("area_sq_m"),
		get_map_field("kgis_village_code"),
		get_map_field("kgis_village_name"),
	]
	if feature_id_field.lower() != "fid":
		required_fields.append(feature_id_field)
	return required_fields


def validate_input_layer(source, required_fields):
	source_fields = list(source.schema["properties"].keys())
	missing_fields = [field for field in required_fields if field not in source_fields]
	if missing_fields:
		raise ValueError(
			f"Missing required fields in layer '{INPUT_LAYER}': {missing_fields}. "
			f"Available fields: {source_fields}"
		)


def get_output_schema(source_geometry_type):
	return {
		"geometry": source_geometry_type,
		"properties": OrderedDict([
			("fid", "int"),
			("survey_num", "int"),
			("category", "str:80"),
			("area_ha", "float"),
			("kgis_village_code", "str:32"),
			("kgis_village_name", "str:120"),
		]),
	}


def get_source_feature_id(source_feature, source_properties):
	feature_id_field = get_map_field("feature_id")
	if feature_id_field in source_properties:
		return int(source_properties[feature_id_field])

	if feature_id_field.lower() == "fid":
		source_id = source_feature.get("id")
		if source_id is None:
			raise ValueError("Source feature has no provider-level 'id' value for fid")
		return int(source_id)

	raise ValueError(
		f"Feature ID field '{feature_id_field}' not found in source properties"
	)


def build_output_feature(source_feature):
	source_properties = source_feature["properties"]
	area_square_meters = parse_area_sq_m(
		source_properties[get_map_field("area_sq_m")]
	)

	output_properties = {
		"fid": get_source_feature_id(source_feature, source_properties),
		"survey_num": parse_survey_number(
			source_properties[get_map_field("survey_num")]
		),
		"category": normalize_text(source_properties[get_map_field("category")]),
		"area_ha": area_square_meters / SQUARE_METERS_PER_HECTARE,
		"kgis_village_code": normalize_village_code(
			source_properties[get_map_field("kgis_village_code")]
		),
		"kgis_village_name": normalize_text(
			source_properties[get_map_field("kgis_village_name")]
		),
	}

	return {
		"geometry": source_feature["geometry"],
		"properties": {column: output_properties[column] for column in OUTPUT_COLUMNS},
	}


def remove_existing_output(output_file):
	if OVERWRITE_OUTPUT and output_file.exists():
		output_file.unlink()


def write_spatial_base(input_file, input_layer, output_file, output_layer):
	if not input_file.exists():
		raise FileNotFoundError(f"Input GPKG not found: {input_file}")

	remove_existing_output(output_file)
	required_fields = get_required_fields()

	with fiona.open(input_file, layer=input_layer) as source:
		if source.schema is None:
			raise ValueError(f"Layer '{input_layer}' has no schema definition")

		validate_input_layer(source, required_fields)
		output_schema = get_output_schema(source.schema["geometry"])
		total_features = len(source)

		with fiona.open(
			output_file,
			mode="w",
			driver="GPKG",
			layer=output_layer,
			schema=output_schema,
			crs=source.crs,
			crs_wkt=source.crs_wkt,
		) as sink:
			written_count = 0
			with tqdm(
				total=total_features,
				desc=f"Processing {input_file.name}",
				unit="feature",
			) as progress:
				for source_feature in source:
					sink.write(build_output_feature(source_feature))
					written_count += 1
					progress.update(1)

	return written_count


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
	written_count = write_spatial_base(
			INPUT_GPKG,
			INPUT_LAYER,
			OUTPUT_GPKG,
			OUTPUT_LAYER,
		)
	
	print(
		f"Wrote {written_count:,} features to "
		f"{OUTPUT_GPKG.name}:{OUTPUT_LAYER}"
	)