import numpy as np
from netCDF4 import Dataset
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import reproject

from reproject_netcdf import (
    create_output_data,
    get_watershed_mask,
    read_source_data,
    reproject_time_steps,
)


TARGET_GRID = (
    156,
    105,
    798778.801047321060,
    1484563.296254347777,
    250.0,
)


def downscale_netcdf(
    input_file,
    output_file,
    target_crs,
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
        "RZSM",
        "time",
        "lat",
        "lon",
        target_crs,
        target_crs,
    )
    ncols, nrows, xllcorner, yllcorner, cellsize = TARGET_GRID
    target_transform = from_origin(
        xllcorner,
        yllcorner + nrows * cellsize,
        cellsize,
        cellsize,
    )
    coordinate_x = target_transform.c + (
        np.arange(ncols) + 0.5
    ) * target_transform.a
    coordinate_y = target_transform.f - (
        np.arange(nrows) + 0.5
    ) * abs(target_transform.e)
    if watershed_file is None:
        watershed_mask = np.ones((nrows, ncols), dtype=bool)
    else:
        watershed_mask = get_watershed_mask(
            coordinate_x,
            coordinate_y,
            watershed_file,
            target_crs,
        )

    target, data_target = create_output_data(
        output_file,
        "RZSM",
        "time",
        "lat",
        "lon",
        time_values,
        time_attributes,
        data_attributes,
        source_attributes,
        source_title,
        target_crs_object,
        coordinate_x,
        coordinate_y,
        ncols,
        nrows,
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
