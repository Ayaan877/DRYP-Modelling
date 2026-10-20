import fiona
import numpy as np
from netCDF4 import Dataset
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import Affine, from_origin
from rasterio.warp import reproject, transform_geom

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

    transform = from_origin(
        float(x_coordinates.min() - x_step / 2),
        float(y_coordinates.max() + abs(y_step) / 2),
        x_step,
        abs(y_step),
    )
    return transform, y_step > 0


def read_source_data(
    input_file,
    data_variable,
    time_dimension,
    y_dimension,
    x_dimension,
    source_crs,
    target_crs,
):
    with Dataset(input_file, "r") as source:
        data_source = source.variables[data_variable]
        x_source = np.asarray(source.variables[x_dimension][:], dtype=np.float64)
        y_source = np.asarray(source.variables[y_dimension][:], dtype=np.float64)
        source_transform, y_ascending = coordinate_transform(x_source, y_source)
        source_crs_object = CRS.from_user_input(source_crs)
        target_crs_object = CRS.from_user_input(target_crs)
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
        fill_value = getattr(data_source, "_FillValue", -9999.0)

    return (
        source_data,
        x_source,
        y_source,
        source_transform,
        y_ascending,
        source_crs_object,
        target_crs_object,
        time_values,
        time_attributes,
        data_attributes,
        source_attributes,
        source_title,
        fill_value,
    )


def projected_source_grid(
    source_transform,
    x_source,
    y_source,
    source_crs_object,
    target_crs_object):

    source_corners_x = [
        source_transform.c,
        source_transform.c + source_transform.a * len(x_source),
    ]
    source_corners_y = [
        source_transform.f,
        source_transform.f - abs(source_transform.e) * len(y_source),
    ]
    source_extent = {
        "type": "Polygon",
        "coordinates": [[
            [source_corners_x[0], source_corners_y[0]],
            [source_corners_x[1], source_corners_y[0]],
            [source_corners_x[1], source_corners_y[1]],
            [source_corners_x[0], source_corners_y[1]],
            [source_corners_x[0], source_corners_y[0]],
        ]],
    }
    projected_extent = transform_geom(
        source_crs_object,
        target_crs_object,
        source_extent,
    )["coordinates"][0]
    projected_x = np.asarray([point[0] for point in projected_extent])
    projected_y = np.asarray([point[1] for point in projected_extent])
    output_ncols = len(x_source)
    output_nrows = len(y_source)
    target_transform = from_origin(
        projected_x.min(),
        projected_y.max(),
        (projected_x.max() - projected_x.min()) / output_ncols,
        (projected_y.max() - projected_y.min()) / output_nrows,
    )
    return target_transform, output_ncols, output_nrows


def create_output_data(
    output_file,
    data_variable,
    time_dimension,
    y_dimension,
    x_dimension,
    time_values,
    time_attributes,
    data_attributes,
    source_attributes,
    source_title,
    target_crs_object,
    coordinate_x,
    coordinate_y,
    output_ncols,
    output_nrows,
    source_data,
    fill_value,
    target_crs):

    output_file.parent.mkdir(parents=True, exist_ok=True)
    target = Dataset(output_file, "w", format="NETCDF4")
    target.setncattr("Conventions", "CF-1.8")
    target.setncattr("title", source_title)
    target.setncattr(
        "history",
        f"Reprojected to {target_crs} using nearest-neighbour resampling",
    )
    for name, value in source_attributes.items():
        target.setncattr(name, value)

    target.createDimension(time_dimension, len(time_values))
    target.createDimension(y_dimension, output_nrows)
    target.createDimension(x_dimension, output_ncols)
    time_target = target.createVariable(
        time_dimension, time_values.dtype, (time_dimension,)
    )
    time_target.setncatts(time_attributes)
    time_target[:] = time_values

    y_target = target.createVariable(y_dimension, "f4", (y_dimension,))
    x_target = target.createVariable(x_dimension, "f4", (x_dimension,))
    x_target.setncatts({
        "units": "m",
        "long_name": "UTM easting pixel center",
        "standard_name": "projection_x_coordinate",
        "axis": "X",
    })
    y_target.setncatts({
        "units": "m",
        "long_name": "UTM northing pixel center",
        "standard_name": "projection_y_coordinate",
        "axis": "Y",
    })
    x_target[:] = coordinate_x
    y_target[:] = coordinate_y

    crs_target = target.createVariable("crs", "i4")
    crs_target.setncatts({
        "long_name": "CRS definition",
        "grid_mapping_name": "universal_transverse_mercator",
        "spatial_ref": target_crs_object.to_wkt(),
        "crs_wkt": target_crs_object.to_wkt(),
        "epsg_code": f"EPSG:{target_crs_object.to_epsg()}",
    })
    data_target = target.createVariable(
        data_variable,
        source_data.dtype,
        (time_dimension, y_dimension, x_dimension),
        zlib=True,
        complevel=4,
        chunksizes=(1, min(256, output_nrows), min(256, output_ncols)),
        fill_value=fill_value,
    )
    data_target.setncatts(data_attributes)
    data_target.coordinates = f"{y_dimension} {x_dimension}"
    data_target.grid_mapping = "crs"
    return target, data_target


def reproject_time_steps(
    data_target,
    source_data,
    y_ascending,
    fill_value,
    source_transform,
    source_crs_object,
    target_transform,
    target_crs_object,
    watershed_mask):

    for index in range(source_data.shape[0]):
        source_slice = source_data[index, :, :]
        if y_ascending:
            source_slice = np.flipud(source_slice)
        projected = np.full(
            watershed_mask.shape,
            fill_value,
            dtype=source_data.dtype,
        )
        reproject(
            source=np.asarray(source_slice.filled(fill_value)),
            destination=projected,
            src_transform=source_transform,
            src_crs=source_crs_object,
            src_nodata=fill_value,
            dst_transform=target_transform,
            dst_crs=target_crs_object,
            dst_nodata=fill_value,
            resampling=Resampling.nearest,
        )
        projected[np.logical_not(watershed_mask)] = fill_value
        data_target[index, :, :] = projected


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

    (
        source_data,
        x_source,
        y_source,
        source_transform,
        y_ascending,
        source_crs_object,
        target_crs_object,
        time_values,
        time_attributes,
        data_attributes,
        source_attributes,
        source_title,
        fill_value,
    ) = read_source_data(
        input_file,
        data_variable,
        time_dimension,
        y_dimension,
        x_dimension,
        source_crs,
        target_crs,
    )
    target_transform, output_ncols, output_nrows = projected_source_grid(
        source_transform,
        x_source,
        y_source,
        source_crs_object,
        target_crs_object,
    )

    coordinate_x = target_transform.c + (
        np.arange(output_ncols) + 0.5
    ) * target_transform.a
    coordinate_y = target_transform.f - (
        np.arange(output_nrows) + 0.5
    ) * abs(target_transform.e)
    if watershed_file is None:
        watershed_mask = np.ones(
            (output_nrows, output_ncols), dtype=bool
        )
    else:
        watershed_mask = get_watershed_mask(
            coordinate_x, coordinate_y, watershed_file, target_crs
        )

    target, data_target = create_output_data(
        output_file,
        data_variable,
        time_dimension,
        y_dimension,
        x_dimension,
        time_values,
        time_attributes,
        data_attributes,
        source_attributes,
        source_title,
        target_crs_object,
        coordinate_x,
        coordinate_y,
        output_ncols,
        output_nrows,
        source_data,
        fill_value,
        target_crs,
    )
    try:
        reproject_time_steps(
            data_target,
            source_data,
            y_ascending,
            fill_value,
            source_transform,
            source_crs_object,
            target_transform,
            target_crs_object,
            watershed_mask,
        )
    finally:
        target.close()
    return target_transform
