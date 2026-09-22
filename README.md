# DRYP Modelling

This project prepares spatial and time-varying inputs for DRYP modelling of the Chikkaballapur (ChikB) watershed in Karnataka, India. The datasets describe crop classes, crop coefficients (Kc), rooting depth, well depth, elevation, and the watershed boundary.

## DRYP

DRYP is a distributed rainfall-runoff hydrological model. It represents water movement across a landscape using spatial grids and time series, so the model inputs here are prepared as georeferenced rasters, NetCDF datasets, tables, and vector boundaries.

## Folders

- `Crop Maps`: crop polygon layers for Eucalyptus and Mango.
- `Kc Map`: crop-coefficient tables, scripts, maps, and NetCDF products.
- `Rooting Depth`: rooting-depth tables, scripts, and gridded products.
- `Well Depth Map`: well-depth observations, elevation data, and QGIS outputs.

## Environment

Create or activate the project environment and install dependencies with:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Generated GIS and scientific-data products are excluded by `.gitignore`; source scripts and intended input data remain available for reproducibility.
