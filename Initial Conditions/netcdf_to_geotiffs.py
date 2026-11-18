from pathlib import Path
from typing import Any, Optional

import numpy as np
import rasterio
from netCDF4 import Dataset, num2date
from rasterio.crs import CRS
from rasterio.errors import CRSError
from rasterio.transform import from_origin
from tqdm import tqdm

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILES = [
    (SCRIPT_DIR / "OBF_hourly_grid_R3.nc", 3)
]
OUTPUT_DIRECTORY = SCRIPT_DIR
TIME_VARIABLE = "time"
X_COORDINATE_VARIABLE: Optional[str] = None
Y_COORDINATE_VARIABLE: Optional[str] = None
SOURCE_CRS: Optional[str] = "EPSG:32643"
VARIABLES: Optional[list[str]] = None
DEFAULT_NODATA = -9999.0

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------


def coordinate_axis(name: str, variable: Any) -> Optional[str]:
    axis = str(getattr(variable, "axis", "")).upper()
    if axis in ("X", "Y"):
        return axis

    standard_name = str(getattr(variable, "standard_name", "")).lower()
    lowered_name = name.lower()
    if standard_name in ("projection_x_coordinate", "longitude", "grid_longitude"):
        return "X"
    if standard_name in ("projection_y_coordinate", "latitude", "grid_latitude"):
        return "Y"
    if lowered_name in ("x", "lon", "longitude", "easting", "eastings"):
        return "X"
    if lowered_name in ("y", "lat", "latitude", "northing", "northings"):
        return "Y"
    return None


def coordinate_variable_name(
    dataset: Dataset, dimension: str, configured_name: Optional[str]
) -> str:
    if configured_name is not None:
        if configured_name not in dataset.variables:
            raise ValueError(f"Coordinate variable {configured_name!r} was not found.")
        variable = dataset.variables[configured_name]
        if variable.dimensions != (dimension,):
            raise ValueError(
                f"Coordinate variable {configured_name!r} must use only "
                f"the {dimension!r} dimension."
            )
        return configured_name

    if dimension in dataset.variables:
        variable = dataset.variables[dimension]
        if variable.dimensions == (dimension,):
            return dimension

    candidates = [
        name
        for name, variable in dataset.variables.items()
        if variable.dimensions == (dimension,)
    ]
    for candidate in candidates:
        if coordinate_axis(candidate, dataset.variables[candidate]) is not None:
            return candidate
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Could not identify a 1D coordinate variable for {dimension!r}; "
        "set X_COORDINATE_VARIABLE or Y_COORDINATE_VARIABLE in the config."
    )


def spatial_dimensions(
    dataset: Dataset, variable: Any, time_dimension: str
) -> tuple[str, str]:
    dimensions = [dimension for dimension in variable.dimensions if dimension != time_dimension]
    if len(dimensions) != 2:
        raise ValueError("A data layer must have exactly two non-time dimensions.")

    x_dimension = None
    y_dimension = None
    for dimension in dimensions:
        coordinate_name = coordinate_variable_name(dataset, dimension, None)
        axis = coordinate_axis(coordinate_name, dataset.variables[coordinate_name])
        if axis == "X":
            x_dimension = dimension
        elif axis == "Y":
            y_dimension = dimension

    if x_dimension is None and X_COORDINATE_VARIABLE is not None:
        x_dimension = dataset.variables[X_COORDINATE_VARIABLE].dimensions[0]
    if y_dimension is None and Y_COORDINATE_VARIABLE is not None:
        y_dimension = dataset.variables[Y_COORDINATE_VARIABLE].dimensions[0]

    if x_dimension is None and y_dimension is None:
        y_dimension, x_dimension = dimensions
    elif x_dimension is None:
        x_dimension = next(dimension for dimension in dimensions if dimension != y_dimension)
    elif y_dimension is None:
        y_dimension = next(dimension for dimension in dimensions if dimension != x_dimension)

    if (
        x_dimension is None
        or y_dimension is None
        or x_dimension == y_dimension
        or x_dimension not in dimensions
        or y_dimension not in dimensions
    ):
        raise ValueError("Could not identify distinct X and Y dimensions for a data layer.")
    return y_dimension, x_dimension


def resolve_crs(
    dataset: Dataset,
    data_variable: Any,
    x_coordinate: Any,
    y_coordinate: Any,
) -> CRS:
    if SOURCE_CRS is not None:
        return CRS.from_user_input(SOURCE_CRS)

    candidates = []
    grid_mapping = getattr(data_variable, "grid_mapping", None)
    if grid_mapping in dataset.variables:
        candidates.append(dataset.variables[grid_mapping])
    candidates.extend((x_coordinate, y_coordinate))
    candidates.extend(dataset.variables.values())

    crs_attribute_names = ("spatial_ref", "crs_wkt", "epsg_code", "crs")
    for variable in candidates:
        for attribute_name in crs_attribute_names:
            value = getattr(variable, attribute_name, None)
            if value is None:
                continue
            try:
                return CRS.from_user_input(value)
            except (CRSError, TypeError, ValueError):
                continue

    for attribute_name in crs_attribute_names:
        value = getattr(dataset, attribute_name, None)
        if value is not None:
            try:
                return CRS.from_user_input(value)
            except (CRSError, TypeError, ValueError):
                continue

    units = " ".join(
        str(getattr(coordinate, "units", "")).lower()
        for coordinate in (x_coordinate, y_coordinate)
    )
    if "degree" in units:
        return CRS.from_epsg(4326)
    raise ValueError(
        "Could not determine the input CRS from NetCDF metadata; "
        "set SOURCE_CRS in the config."
    )


def make_grid(
    x_values: np.ndarray, y_values: np.ndarray
) -> tuple[Any, bool, bool]:
    if x_values.size < 2 or y_values.size < 2:
        raise ValueError("At least two X and Y coordinate values are required.")

    x_differences = np.diff(x_values)
    y_differences = np.diff(y_values)
    x_step = float(np.median(x_differences))
    y_step = float(np.median(y_differences))
    if x_step == 0 or y_step == 0:
        raise ValueError("Coordinate values must be strictly monotonic.")
    if not np.allclose(x_differences, x_step) or not np.allclose(y_differences, y_step):
        raise ValueError("GeoTIFF output requires regularly spaced coordinates.")

    transform = from_origin(
        float(np.min(x_values)) - abs(x_step) / 2,
        float(np.max(y_values)) + abs(y_step) / 2,
        abs(x_step),
        abs(y_step),
    )
    return transform, x_step < 0, y_step > 0


def data_layer_names(
    dataset: Dataset, time_dimension: str, selected_variables: Optional[list[str]]
) -> list[str]:
    if selected_variables is not None:
        missing = [name for name in selected_variables if name not in dataset.variables]
        if missing:
            raise ValueError(f"Requested data variables were not found: {missing}")
        candidates = selected_variables
    else:
        candidates = list(dataset.variables)

    return [
        name
        for name in candidates
        if time_dimension in dataset.variables[name].dimensions
        and len(dataset.variables[name].dimensions) == 3
    ]


def convert_netcdf_to_geotiffs(
    input_file: Path, run_number: int, output_directory: Path
) -> None:
    with Dataset(input_file, "r") as source:
        if TIME_VARIABLE not in source.variables:
            raise ValueError(f"Time variable {TIME_VARIABLE!r} was not found in {input_file}.")

        time_variable = source.variables[TIME_VARIABLE]
        time_units = getattr(time_variable, "units", None)
        if not time_units:
            raise ValueError(f"Time variable {TIME_VARIABLE!r} has no units attribute.")
        calendar = getattr(time_variable, "calendar", "standard")
        timestamps: list[Any] = np.asarray(
            num2date(
                time_variable[:],
                time_units,
                calendar=calendar,
                only_use_cftime_datetimes=False,
            ),
            dtype=object,
        ).reshape(-1).tolist()
        layer_names = data_layer_names(source, TIME_VARIABLE, VARIABLES)
        if not layer_names:
            raise ValueError(f"No time-dependent 2D data layers were found in {input_file}.")

        for layer_name in layer_names:
            data_variable = source.variables[layer_name]
            y_dimension, x_dimension = spatial_dimensions(
                source, data_variable, TIME_VARIABLE
            )
            x_name = coordinate_variable_name(
                source, x_dimension, X_COORDINATE_VARIABLE
            )
            y_name = coordinate_variable_name(
                source, y_dimension, Y_COORDINATE_VARIABLE
            )
            x_values = np.asarray(source.variables[x_name][:], dtype=np.float64)
            y_values = np.asarray(source.variables[y_name][:], dtype=np.float64)
            transform, reverse_x, reverse_y = make_grid(x_values, y_values)
            crs = resolve_crs(
                source,
                data_variable,
                source.variables[x_name],
                source.variables[y_name],
            )

            layer_output_directory = output_directory / f"{layer_name}_run{run_number}"
            layer_output_directory.mkdir(parents=True, exist_ok=True)
            remaining_dimensions = [
                dimension
                for dimension in data_variable.dimensions
                if dimension != TIME_VARIABLE
            ]
            transpose_order = (
                remaining_dimensions.index(y_dimension),
                remaining_dimensions.index(x_dimension),
            )
            fill_value = getattr(data_variable, "_FillValue", None)

            for time_index in tqdm(
                range(len(timestamps)),
                desc=f"{layer_name} run{run_number}",
                unit="frame",
            ):
                indexers = tuple(
                    time_index if dimension == TIME_VARIABLE else slice(None)
                    for dimension in data_variable.dimensions
                )
                frame = np.ma.asarray(data_variable[indexers])
                frame = np.ma.transpose(frame, axes=transpose_order)
                if reverse_y:
                    frame = frame[::-1, :]
                if reverse_x:
                    frame = frame[:, ::-1]

                output_nodata = fill_value
                if np.ma.is_masked(frame) and np.ma.getmaskarray(frame).any():
                    if output_nodata is None:
                        frame = frame.astype(np.float32)
                        output_nodata = DEFAULT_NODATA
                    output_data = np.ma.filled(frame, output_nodata)
                else:
                    output_data = np.asarray(frame)

                timestamp = timestamps[time_index].strftime("%Y-%m-%d_%H-%M-%S")
                output_file = layer_output_directory / (
                    f"{layer_name}_run{run_number}_{timestamp}.tif"
                )
                profile = {
                    "driver": "GTiff",
                    "height": output_data.shape[0],
                    "width": output_data.shape[1],
                    "count": 1,
                    "dtype": output_data.dtype,
                    "crs": crs,
                    "transform": transform,
                    "compress": "deflate",
                }
                if output_nodata is not None:
                    profile["nodata"] = output_nodata

                with rasterio.open(output_file, "w", **profile) as destination:
                    destination.write(output_data, 1)
                    destination.update_tags(
                        variable=layer_name,
                        run_number=str(run_number),
                        timestamp=timestamps[time_index].isoformat(),
                        source_file=input_file.name,
                        time_units=time_units,
                    )


def main() -> None:
    for input_file, run_number in INPUT_FILES:
        convert_netcdf_to_geotiffs(input_file, run_number, OUTPUT_DIRECTORY)


if __name__ == "__main__":
    main()