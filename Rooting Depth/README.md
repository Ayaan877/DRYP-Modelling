# Rooting Depth

This folder creates crop rooting-depth maps for the Chikkaballapur watershed and converts them between 10 m and 250 m grids for hydrological modelling.

## Scripts

- `make_rooting_depth.py`: reads the rooting-depth lookup table, Kharif and Rabi crop rasters, and the watershed boundary; writes a 10 m ESRI ASCII rooting-depth grid.
- `downscale_rooting_depth.py`: regrids the 10 m rooting-depth grid to the fixed 250 m target grid.

## Inputs

`Rooting Depth FAO56.csv`, Kharif and Rabi crop-map rasters, the nested `Updated Watershed` shapefile, and the 10 m rooting-depth grid for downscaling.

## Outputs

`Rooting_Depth_*_10m.asc` and `Rooting_Depth_*_250m.asc` grids, plus the QGIS project and supporting map metadata.