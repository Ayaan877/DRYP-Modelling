from datetime import date
from pathlib import Path

import fiona
import numpy as np
import rasterio.features
from netCDF4 import Dataset, date2num, num2date
from rasterio.transform import from_origin
from rasterio.warp import transform_geom

from reproject_netcdf import reproject_netcdf

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "RF_predcited_RZSM_1981_2024.nc"
WATERSHED_FILE = SCRIPT_DIR / "Updated Watershed" / ("Updated_watershed.shp")
OUTPUT_FILE = SCRIPT_DIR / "RZSM_2014-2024.nc"
START_YEAR = 2014
END_YEAR = 2024
NETCDF_FORMAT = "NETCDF4"
INPUT_CRS = "EPSG:4326"
OUTPUT_CRS = "EPSG:32643"
COMPRESSION_LEVEL = 4
TIME_CHUNK_SIZE = 31

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------


def read_watershed(path):
    with fiona.open(path) as source:
        geometries = [feature["geometry"] for feature in source]
        source_crs = source.crs_wkt or source.crs

    if not geometries:
        raise ValueError(f"No geometries were found in {path}")
    if not source_crs:
        raise ValueError(f"The watershed CRS is missing from {path}")
    return geometries, source_crs


def coordinate_indices(coordinates, lower, upper, cell_padding=0.0):
    indices = np.flatnonzero(
        (coordinates >= lower - cell_padding)
        & (coordinates <= upper + cell_padding)
    )
    if indices.size == 0:
        raise ValueError("The watershed does not overlap the input grid.")
    return indices


def make_watershed_mask(geometries, source_crs, lon, lat, input_crs):
    transformed = [
        transform_geom(source_crs, input_crs, geometry)
        for geometry in geometries
    ]
    subset_lon = lon
    subset_lat = lat
    x_resolution = float(np.median(np.diff(subset_lon)))
    y_resolution = float(np.median(np.diff(subset_lat)))
    transform = from_origin(
        float(subset_lon[0]) - x_resolution / 2,
        float(subset_lat[-1]) + y_resolution / 2,
        x_resolution,
        y_resolution,
    )
    mask_descending_lat = rasterio.features.geometry_mask(
        transformed,
        out_shape=(len(subset_lat), len(subset_lon)),
        transform=transform,
        invert=True,
        all_touched=True,
    )
    return mask_descending_lat[::-1, :]


def extract_soil_moisture(
    input_file, watershed_file, output_file, start_year, end_year, input_crs
):
    geometries, watershed_crs = read_watershed(watershed_file)
    requested_start = date(start_year, 1, 1)
    requested_end = date(end_year, 12, 31)
    if requested_start > requested_end:
        raise ValueError("START_YEAR must not be greater than END_YEAR")

    with Dataset(input_file, "r") as source:
        time_variable = source.variables["time"]
        source_dates = np.array(
            num2date(
                time_variable[:],
                time_variable.units,
                getattr(time_variable, "calendar", "standard"),
                only_use_cftime_datetimes=False,
            ),
            dtype=object,
            ndmin=1,
        )
        source_start = source_dates[0].date()
        source_end = source_dates[-1].date()
        selection_start = max(requested_start, source_start)
        selection_end = min(requested_end, source_end)
        if (selection_start, selection_end) != (requested_start, requested_end):
            print(
                f"Requested period {requested_start} through {requested_end} "
                f"extends beyond the input period {source_start} through "
                f"{source_end}; extracting {selection_start} through "
                f"{selection_end}."
            )
        selected_time = np.array(
            [
                selection_start <= current.date() <= selection_end
                for current in source_dates
            ],
            dtype=bool,
        )
        time_indices = np.flatnonzero(selected_time)
        if time_indices.size == 0:
            raise ValueError("The requested dates do not overlap the input.")

        lon = np.asarray(source.variables["lon"][:])
        lat = np.asarray(source.variables["lat"][:])
        transformed = [
            transform_geom(watershed_crs, "EPSG:4326", geometry)
            for geometry in geometries
        ]
        from rasterio.features import bounds as geometry_bounds

        min_lon, min_lat, max_lon, max_lat = geometry_bounds(transformed[0])
        for geometry in transformed[1:]:
            g_min_lon, g_min_lat, g_max_lon, g_max_lat = geometry_bounds(geometry)
            min_lon = min(min_lon, g_min_lon)
            min_lat = min(min_lat, g_min_lat)
            max_lon = max(max_lon, g_max_lon)
            max_lat = max(max_lat, g_max_lat)
        lon_resolution = float(np.median(np.diff(lon)))
        lat_resolution = float(np.median(np.diff(lat)))
        lon_indices = coordinate_indices(
            lon,
            min_lon,
            max_lon,
            cell_padding=abs(lon_resolution) / 2,
        )
        lat_indices = coordinate_indices(
            lat,
            min_lat,
            max_lat,
            cell_padding=abs(lat_resolution) / 2,
        )
        mask = make_watershed_mask(
            geometries,
            watershed_crs,
            lon[lon_indices],
            lat[lat_indices],
            input_crs=input_crs,
        )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        source_data = source.variables["RZSM"]
        fill_value = getattr(source_data, "_FillValue", -9999.0)
        with Dataset(output_file, "w", format=NETCDF_FORMAT) as target:
            target.createDimension("time", len(time_indices))
            target.createDimension("lat", len(lat_indices))
            target.createDimension("lon", len(lon_indices))
            target.setncattr(
                "title",
                "Root-zone soil moisture within the updated watershed boundary",
            )
            target.setncattr("source_file", str(input_file))
            target.setncattr("watershed_file", str(watershed_file))
            target.setncattr(
                "extracted_period",
                f"{selection_start.isoformat()} through {selection_end.isoformat()}",
            )

            out_lon = target.createVariable("lon", lon.dtype, ("lon",))
            out_lat = target.createVariable("lat", lat.dtype, ("lat",))
            out_time = target.createVariable("time", time_variable.dtype, ("time",))
            out_rzsm = target.createVariable(
                "RZSM",
                source_data.dtype,
                ("time", "lat", "lon"),
                zlib=True,
                complevel=COMPRESSION_LEVEL,
                fill_value=fill_value,
                chunksizes=(
                    min(TIME_CHUNK_SIZE, len(time_indices)),
                    len(lat_indices),
                    len(lon_indices),
                ),
            )
            out_lon[:] = lon[lon_indices]
            out_lat[:] = lat[lat_indices]
            out_time[:] = date2num(
                [source_dates[i] for i in time_indices],
                time_variable.units,
                getattr(time_variable, "calendar", "standard"),
            )
            out_lon.setncatts(lon_variable_attrs(source))
            out_lat.setncatts(lat_variable_attrs(source))
            out_time.setncatts(
                {
                    "units": time_variable.units,
                    "calendar": getattr(time_variable, "calendar", "standard"),
                }
            )
            out_rzsm.setncatts(
                {
                    name: value
                    for name, value in source_data.__dict__.items()
                    if name != "_FillValue"
                }
            )

            for output_start in range(0, len(time_indices), TIME_CHUNK_SIZE):
                output_stop = min(
                    output_start + TIME_CHUNK_SIZE, len(time_indices)
                )
                input_indices = time_indices[output_start:output_stop]
                data = np.ma.asarray(
                    source_data[
                        input_indices.min() : input_indices.max() + 1,
                        lat_indices.min() : lat_indices.max() + 1,
                        lon_indices.min() : lon_indices.max() + 1,
                    ]
                )
                data = data[input_indices - input_indices.min(), :, :]
                data = np.ma.masked_where(
                    np.broadcast_to(~mask, data.shape), data
                )
                out_rzsm[output_start:output_stop, :, :] = data

    reproject_netcdf(
        output_file,
        output_file,
        INPUT_CRS,
        OUTPUT_CRS,
        watershed_file=watershed_file,
    )
    print(f"Wrote {len(time_indices)} time steps to {output_file}")


def lon_variable_attrs(source):
    return dict(source.variables["lon"].__dict__)


def lat_variable_attrs(source):
    return dict(source.variables["lat"].__dict__)


# -----------------------------------------------------------------------
# Run Script
# -----------------------------------------------------------------------

if __name__ == "__main__":
    extract_soil_moisture(
        INPUT_FILE,
        WATERSHED_FILE,
        OUTPUT_FILE,
        START_YEAR,
        END_YEAR,
        INPUT_CRS
    )