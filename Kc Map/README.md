# Kc Map

These files produce daily crop coefficient (Kc) datasets and a spatiotemporal crop coefficient (Kc) netCDF file for the Chikkaballapur watershed.

## Input Data

- `Kc_FAO56.csv` - Crop-wise metadata needed to prepare Kc curves, including: 
    - Classification
    - Season
    - Crop Name
    - FAO Proxy Name
    - Sowing Period
    - Seasonal Area (for Kharif, Rabi and Summer)
    - Growth Period (min and max)
    - Growth Stage Durations (ini, dev, mid and late)
    - Crop Coefficient (Kc_ini, Kc_mid, Kc_end)
- **Crop Maps** - `CropMap_Kharif_2024-25_new.tif` and `CropMap_Rabi_2024-25_new.tif` contain the spatial distribution of class-wise crops for the Kharif and Rabi seasons, respectively.
- **Updated Watershed** - Shapefile for watershed area boundary. 
- `Soil_Texture.tif` - Soil texture GEOtiff needed for excluding habitation, rocky outcrops, water bodies and dykes and ridges

## Scripts

- `clean_Kc_metadata.py`: Cleans the FAO-56 crop metadata into `cropwise_Kc.csv`.
- `make_cropwise_Kc_curves.py`: Converts crop metadata into multi-year crop-wise Kc curves with daily temporal resolution.
- `make_classwise_Kc_curves.py`: Area-weights crop curves by crop class and creates class-wise Kc curves.
- `classwise_calendar.py`: Creates a class-wise calendar with crop composition, union growth periods, and annual Kc ranges.
- `make_spatiotemporal_Kc.py`: combines Kharif and Rabi crop rasters, class-wise curves, and the watershed boundary into a daily 10 m NetCDF Kc dataset.
- `downscale_kc_netcdf.py`: regrids the 10 m NetCDF Kc dataset to a fixed 250 m grid.
- `plot_Kc.py`: plots a crop Kc curve with seasonal bands for visual checking.
- `check_data.py`: plots and checks crop-wise Kc curve data.

## Main inputs

FAO-56/crop coefficient CSV files, Kharif and Rabi crop-map rasters, the `Updated Watershed` shapefile, and supporting map metadata.

## Main outputs

Cleaned and interpolated CSV curves, daily `Kc_spatiotemporal_*.nc` datasets at 10 m and 250 m, plots, and QGIS visualization projects. Generated products are ignored by `.gitignore`.
