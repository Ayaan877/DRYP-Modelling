# Kc Map

This folder prepares crop coefficients (Kc) for the Chikkaballapur watershed and produces daily spatial Kc datasets for DRYP.

## Scripts

- `clean_Kc_metadata.py`: cleans the FAO-56 crop metadata into `cropwise_Kc.csv`.
- `make_cropwise_Kc_curves.py`: converts crop metadata into multi-year daily crop-wise Kc curves.
- `make_classwise_Kc_curves.py`: area-weights crop curves by crop class and creates class-wise curves.
- `classwise_calendar.py`: creates a class-wise calendar with crop composition, union growth periods, and annual Kc ranges.
- `make_spatiotemporal_Kc.py`: combines Kharif and Rabi crop rasters, class-wise curves, and the watershed boundary into a daily 10 m NetCDF Kc dataset.
- `downscale_kc_netcdf.py`: regrids the 10 m NetCDF Kc dataset to a fixed 250 m grid.
- `plot_Kc.py`: plots a crop Kc curve with seasonal bands for visual checking.
- `check_data.py`: plots and checks crop-wise Kc curve data.

## Main inputs

FAO-56/crop coefficient CSV files, Kharif and Rabi crop-map rasters, the `Updated Watershed` shapefile, and supporting map metadata.

## Main outputs

Cleaned and interpolated CSV curves, daily `Kc_spatiotemporal_*.nc` datasets at 10 m and 250 m, plots, and QGIS visualization projects. Generated products are ignored by `.gitignore`.
