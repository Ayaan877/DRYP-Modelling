import numpy as np
import fiona
from netCDF4 import Dataset
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_geom

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

TARGET_NCOLS = 156
TARGET_NROWS = 105
TARGET_XLLCORNER = 798778.801047321060
TARGET_YLLCORNER = 1484563.296254347777
TARGET_CELLSIZE = 250.0
TARGET_TRANSFORM = from_origin(
    TARGET_XLLCORNER,
    TARGET_YLLCORNER + TARGET_NROWS * TARGET_CELLSIZE,
    TARGET_CELLSIZE,
    TARGET_CELLSIZE,
)

# -----------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------

def copy_attributes(source, target):
    for name in source.ncattrs():
        if name != "_FillValue":
            target.setncattr(name, source.getncattr(name))


def get_watershed_mask(coordinate_x, coordinate_y, watershed_file, target_crs):
    with fiona.open(watershed_file) as watershed:
        source_crs = watershed.crs_wkt or watershed.crs
        if not source_crs:
            raise ValueError("The watershed shapefile has no CRS information.")
        geometries = [
            transform_geom(source_crs, target_crs, feature["geometry"])
            for feature in watershed
        ]

    if not geometries:
        raise ValueError("The watershed shapefile contains no geometries.")

    x_resolution = abs(float(np.median(np.diff(coordinate_x))))
    y_resolution = abs(float(np.median(np.diff(coordinate_y))))
    transform = from_origin(
        float(coordinate_x[0]) - x_resolution / 2,
        float(coordinate_y[0]) + y_resolution / 2,
        x_resolution,
        y_resolution,
    )
    mask = geometry_mask(
        geometries,
        out_shape=(len(coordinate_y), len(coordinate_x)),
        transform=transform,
        invert=True,
        all_touched=True,
    )
    if not np.any(mask):
        raise ValueError("The watershed does not overlap the projected grid.")
    return mask


def coordinate_transform(x_coordinates, y_coordinates):
    x_step = float(np.median(np.diff(x_coordinates)))
    y_step = float(np.median(np.diff(y_coordinates)))
    if not np.isclose(np.diff(x_coordinates), x_step, rtol=1e-4, atol=1e-5).all():
        raise ValueError("The source X coordinate is not regularly spaced.")
    if not np.isclose(np.diff(y_coordinates), y_step, rtol=1e-4, atol=1e-5).all():
        raise ValueError("The source Y coordinate is not regularly spaced.")
    if x_step <= 0 or y_step == 0:
        raise ValueError("Source coordinates must increase in X and be nonzero in Y.")

    top = float(y_coordinates.max() + abs(y_step) / 2)
    transform = from_origin(
        float(x_coordinates.min() - x_step / 2),
        top,
        x_step,
        abs(y_step),
    )
    return transform, y_step > 0


def reproject_netcdf(
    input_file,
    output_file,
    source_crs,
    target_crs,
    data_variable="RZSM",
    time_dimension="time",
    y_dimension="lat",
    x_dimension="lon",
    watershed_file=None,
):
    """Reproject a ``(time, y, x)`` NetCDF variable with nearest neighbour."""
    with Dataset(input_file, "r") as source:
        data_source = source.variables[data_variable]
        x_source = np.asarray(source.variables[x_dimension][:], dtype=np.float64)
        y_source = np.asarray(source.variables[y_dimension][:], dtype=np.float64)
        source_transform, y_ascending = coordinate_transform(x_source, y_source)
        source_crs_object = CRS.from_user_input(source_crs)
        target_crs_object = CRS.from_user_input(target_crs)
        fill_value = getattr(data_source, "_FillValue", -9999.0)
        time_values = np.asarray(source.variables[time_dimension][:])
        time_attributes = {
            name: source.variables[time_dimension].getncattr(name)
            for name in source.variables[time_dimension].ncattrs()
        }
        source_data = np.ma.asarray(data_source[:])
        data_attributes = {
            name: data_source.getncattr(name)
            for name in data_source.ncattrs()
            if name != "_FillValue"
        }
        source_attributes = {
            name: source.getncattr(name)
            for name in ("source_file", "watershed_file", "extracted_period")
            if hasattr(source, name)
        }
        source_title = getattr(
            source,
            "title",
            f"{data_variable} reprojected to {target_crs}",
        )

    coordinate_x = TARGET_TRANSFORM.c + (
        np.arange(TARGET_NCOLS) + 0.5
    ) * TARGET_TRANSFORM.a
    coordinate_y = TARGET_TRANSFORM.f - (
        np.arange(TARGET_NROWS) + 0.5
    ) * abs(TARGET_TRANSFORM.e)
    if watershed_file is None:
        watershed_mask = np.ones(
            (TARGET_NROWS, TARGET_NCOLS),
            dtype=bool,
        )
    else:
        watershed_mask = get_watershed_mask(
            coordinate_x, coordinate_y, watershed_file, target_crs
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(output_file, "w", format="NETCDF4") as target:
        target.setncattr("Conventions", "CF-1.8")
        target.setncattr("title", source_title)
        target.setncattr(
            "history",
            "Reprojected to EPSG:32643 using nearest-neighbour resampling",
        )
        for name, value in source_attributes.items():
            target.setncattr(name, value)

        target.createDimension(time_dimension, len(time_values))
        target.createDimension(y_dimension, TARGET_NROWS)
        target.createDimension(x_dimension, TARGET_NCOLS)

        time_target = target.createVariable(
            time_dimension,
            time_values.dtype,
            (time_dimension,),
        )
        time_target.setncatts(time_attributes)
        time_target[:] = time_values

        y_target = target.createVariable(y_dimension, "f4", (y_dimension,))
        x_target = target.createVariable(x_dimension, "f4", (x_dimension,))
        x_target.units = "m"
        y_target.units = "m"
        x_target.long_name = "UTM easting pixel center"
        y_target.long_name = "UTM northing pixel center"
        x_target.standard_name = "projection_x_coordinate"
        y_target.standard_name = "projection_y_coordinate"
        x_target.axis = "X"
        y_target.axis = "Y"
        x_target[:] = coordinate_x
        y_target[:] = coordinate_y

        crs_target = target.createVariable("crs", "i4")
        crs_target.long_name = "CRS definition"
        crs_target.grid_mapping_name = "universal_transverse_mercator"
        crs_target.spatial_ref = target_crs_object.to_wkt()
        crs_target.crs_wkt = target_crs_object.to_wkt()
        crs_target.epsg_code = f"EPSG:{target_crs_object.to_epsg()}"

        data_target = target.createVariable(
            data_variable,
            source_data.dtype,
            (time_dimension, y_dimension, x_dimension),
            zlib=True,
            complevel=4,
            chunksizes=(1, min(256, TARGET_NROWS), min(256, TARGET_NCOLS)),
            fill_value=fill_value,
        )
        data_target.setncatts(data_attributes)
        data_target.coordinates = f"{y_dimension} {x_dimension}"
        data_target.grid_mapping = "crs"

        for index in range(len(time_values)):
            source_slice = source_data[index, :, :]
            if y_ascending:
                source_slice = np.flipud(source_slice)
            source_array = source_slice.filled(fill_value)
            projected = np.full(
                (TARGET_NROWS, TARGET_NCOLS),
                fill_value,
                dtype=source_data.dtype,
            )
            reproject(
                source=np.asarray(source_array),
                destination=projected,
                src_transform=source_transform,
                src_crs=source_crs_object,
                src_nodata=fill_value,
                dst_transform=TARGET_TRANSFORM,
                dst_crs=target_crs_object,
                dst_nodata=fill_value,
                resampling=Resampling.nearest,
            )
            projected[np.logical_not(watershed_mask)] = fill_value
            data_target[index, :, :] = projected

    print(f"Wrote reprojected NetCDF: {output_file}")
