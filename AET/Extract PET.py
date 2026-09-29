#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from netCDF4 import Dataset
import sys
import datetime as dt


# In[3]:


get_ipython().system('pip install numpy netcdf4')


# In[12]:


## ***** MAKE THE NECESSARY CHANGES TO THE FUNCTION main() ***********##
## ***** AND RUN THIS PYTHON SCRIPT TO DOWNLOAD hPET and dPET ********##
def main():
    # example (please change these values to your specification)
    # input arguments
    # what do you want to extract single point(0) or a region(1)?
    spatial_res = 1

    # what is the period?
    startyear = 2025
    endyear = 2025

    # what is the extent of the region?
    # this only be used if spatial_res = 1
    latmin_ws = 13.41489
    latmax =  13.64971
    lonmin = 77.76006
    lonmax = 78.11752

    # ✅ FIX: Add 0.1° buffer (= 1 grid cell) on all sides
    grid_res = 0.1
    latmin = latmin_ws - 2*grid_res
    #latmax = latmax_ws + grid_res
    #lonmin = lonmin_ws - grid_res
    #lonmax = lonmax_ws + grid_res

    # what is the location of your single point?
    # this only be used if spatial_res = 0
    latval = 16.21 
    lonval = 76.71

    # What is the specific naming you want to use for the area?
    regionname= 'Badenhalli_subwatershed'

    # what is the time resolution hPET you want hourly or daily? 
    t_resolution ='hourly'

    # what is the directory where you download and put the hPET data?
    # download the data from the Birstol repository manually and save it 
    # in your local machine.
    data_path = 'E:/Well Labs/PET_hourly/'

    # where do you want the extracted subset of the data to be?
    output_path = 'E:/Well Labs/Extracted PET_hourly/'

    # run script
    if spatial_res == 0:
        wrapper_singlepoint(startyear,endyear,latval,lonval,regionname,t_resolution,data_path,output_path)
    elif spatial_res == 1:
        wrapper_region(startyear,endyear,latmin,latmax,lonmin,lonmax,regionname,t_resolution,data_path,output_path)
    else:
        raise ValueError('spatial_res only takes 0 or 1 please check!!')


## ********** NO CHANGE ON THE CODE BELOW THIS **********************************************##
def wrapper_region(startyear,endyear,latmin,latmax,lonmin,lonmax,regionname,t_resolution,data_path,output_path):
    """
    This is a wrapper function to run for downloading hPET and dPET data.
    All the arguments need to be given at the end of the script before running
    the script.

    :param startyear: the begging year to start the data download (min = 1981, max = 2019)
    :param endyear: the last year of data to be  downloaded (min = 1981, max = 2019)
    :param latmin: the minimum latitude value of the region (float)
    :param latmax: the maximum latitude value of the region (float)
    :param lonmin: the minimum longitude value of the region (float)
    :param lonmax: the maximum longitude value of the region (float)
    :param regionname: name of the region (it could be any name the user wants) (string)
    :param t_resolution: the time resolution to be downloaded (daily or hourly)
    :param data_path:  the file path that contain the downloaded data (string)
    :param output_path:  the file path to store the extracted subset of the data (string)
    :return:
    """

    if t_resolution == 'daily':
        datapath = data_path 
    elif t_resolution == 'hourly':
        datapath = data_path
    else:
        raise ValueError("t_resolution is wrong please write 'daily' or 'hourly'")

    # set up the year array loop through each year to download the data
    years = np.arange(startyear,endyear+1)
    for y in range(0,len(years)):
        year=int(years[y])
        region_extract(datapath,year,latmin,latmax,lonmin,lonmax,regionname,t_resolution,output_path)
        print(year)


def wrapper_singlepoint(startyear,endyear,latval,lonval,regionname,t_resolution,data_path,output_path):
    """
    This is a wrapper function to run for downloading hPET and dPET data.
    All the arguments need to be given at the end of the script before running
    the script.

    :param startyear: the begging year to start the data download (min = 1981, max = 2019)
    :param endyear: the last year of data to be  downloaded (min = 1981, max = 2019)
    :param latmin: the minimum latitude value of the region (float)
    :param latmax: the maximum latitude value of the region (float)
    :param lonmin: the minimum longitude value of the region (float)
    :param lonmax: the maximum longitude value of the region (float)
    :param regionname: name of the region (it could be any name the user wants) (string)
    :param t_resolution: the time resolution to be downloaded (daily or hourly)
    :param data_path:  the file path that contain the downloaded data (string)
    :param output_path:  the file path to store the extracted subset of the data (string)
    :return:
    """

    if t_resolution == 'daily':
        datapath = data_path
    elif t_resolution == 'hourly':
        datapath = data_path
    else:
        raise ValueError("t_resolution is wrong please write 'daily' or 'hourly'")

    # set up the year array loop through each year to download the data
    years = np.arange(startyear,endyear+1)
    for y in range(0,len(years)):
        year=int(years[y])
        singlepoint_extract(datapath,year,latval,lonval,regionname,t_resolution,output_path)
        print(year)


def region_extract(datapath,year,latmin,latmax,lonmin,lonmax,regionname,t_resolution,output_path):
    """
    This function extract the data from the global hPET and dPET file and write a new
    netCDF file with a file name <year>_<t_resolution>_pet_<regionname>.nc in the output_path
    provided.

    :param datapath: the file path where the hPET data is stored (url)
    :param year: the year for which data is going to be downloaded (integer)
    :param latmin: the minimum latitude value (float)
    :param latmax: the maximum latitude value (float)
    :param lonmin: the minimum longitude value (float)
    :param lonmax: the maximum longitude value (float)
    :param regionname: name of the region (it could be any name the user wants) (string)
    :param t_resolution: the time resolution to be downloaded (daily or hourly)
    :param output_path:  the file path to store the downloaded data (string)
    :return: hPET or dPET data in a netCDF file
    """

    if t_resolution == 'daily':
        fname = '_daily_pet.nc'
        tunits='days since '+str(year)+'-01-01' # time unit for the new netcdf file
    elif t_resolution == 'hourly':
        fname = '_hourly_pet.nc'
        tunits='hours since '+str(year)+'-01-01 00:00:00'
    else:
        raise ValueError("t_resolution is wrong please write 'daily' or 'hourly'")

    pet_hr = Dataset(datapath + str(year) + fname)
    lats = pet_hr.variables['latitude'][:]
    lons = pet_hr.variables['longitude'][:]

    # extract the min and max index
    latminind, lonminind = nearest_point(latmin, lonmin, lats, lons)
    latmaxind, lonmaxind = nearest_point(latmax, lonmax, lats, lons)

    # read the data pet
    reg_data=pet_hr.variables['pet'][:, latmaxind:latminind, lonminind:lonmaxind]  
    # read the new latitude and longitude
    newlats=lats[latmaxind:latminind]
    newlons=lons[lonminind:lonmaxind]
    # get a filename and variable name (here it is called pet)
    filename=output_path+str(year)+'_'+t_resolution+'_pet_'+regionname+'.nc'
    varname='pet'
    # write the new data on a netcdf file
    nc_write(reg_data, newlats, newlons, varname, tunits, filename)

    # Extract the timezone values for the grid
    nc_offset=Dataset(datapath + 'timezone_offset.nc')
    offset=nc_offset.variables['offset'][latmaxind:latminind, lonminind:lonmaxind]
    # write the new data on a netcdf file
    filename = output_path+'timezone_offset_'+regionname+'.nc'
    tunits = 'days since 1981-01-01'
    nc_write(offset, newlats, newlons, 'offset', tunits, filename)

    return None


def singlepoint_extract(datapath,year,latval,lonval,regionname,t_resolution,output_path):
    """
    This function extract the data from the global hPET and dPET file and write a new
    netCDF file with a file name <year>_<t_resolution>_pet_<regionname>.nc in the output_path
    provided.

    :param datapath: the file path where the hPET data is stored (url)
    :param year: the year for which data is going to be downloaded (integer)
    :param latmin: the minimum latitude value (float)
    :param latmax: the maximum latitude value (float)
    :param lonmin: the minimum longitude value (float)
    :param lonmax: the maximum longitude value (float)
    :param regionname: name of the region (it could be any name the user wants) (string)
    :param t_resolution: the time resolution to be downloaded (daily or hourly)
    :param output_path:  the file path to store the downloaded data (string)
    :return: hPET or dPET data in a netCDF file
    """

    if t_resolution == 'daily':
        fname = '_daily_pet.nc'
        tunits='days since '+str(year)+'-01-01' # time unit for the new netcdf file

    elif t_resolution == 'hourly':
        fname = '_hourly_pet.nc'
        tunits='hours since '+str(year)+'-01-01 00:00:00'

    else:
        raise ValueError("t_resolution is wrong please write 'daily' or 'hourly'")

    pet_hr = Dataset(datapath + str(year) + fname)
    lats = pet_hr.variables['latitude'][:]
    lons = pet_hr.variables['longitude'][:]

    # extract the min and max index
    latind, lonind = nearest_point(latval, lonval, lats, lons)

    print(latind)
    print(lonind)
    # read the data pet
    point_data=pet_hr.variables['pet'][:, latind, lonind]  

    # Extract the timezone values for the grid
    nc_offset=Dataset(datapath + 'timezone_offset.nc')
    offset=nc_offset.variables['offset'][latind, lonind]
    if t_resolution == 'daily':
        filename='dPET_'+str(latval)+'_'+str(lonval)+'_'+str(year)+'.txt'
    elif t_resolution == 'hourly':
        filename='hPET_'+str(latval)+'_'+str(lonval)+'_'+str(offset)+'_'+str(year)+'.txt'
    else:
        raise ValueError("t_resolution is wrong please write 'daily' or 'hourly'")
    # save the data in a text file
    np.savetxt(output_path+filename,point_data,fmt='%0.5f')

    return None   


def nearest_point(lat_var, lon_var, lats, lons):
    """
    This function identify the nearest grid location index for a specific lat-lon
    point.
    :param lat_var: the latitude
    :param lon_var: the longitude
    :param lats: all available latitude locations in the data
    :param lons: all available longitude locations in the data
    :return: the lat_index and lon_index
    """
    # this part is to handle if lons are givn 0-360 or -180-180
    if any(lons > 180.0) and (lon_var < 0.0):
        lon_var = lon_var + 360.0
    else:
        lon_var = lon_var

    lat = lats
    lon = lons

    if lat.ndim == 2:
        lat = lat[:, 0]
    else:
        pass
    if lon.ndim == 2:
        lon = lon[0, :]
    else:
        pass

    index_a = np.where(lat >= lat_var)[0][-1]
    index_b = np.where(lat <= lat_var)[0][-1]

    if abs(lat[index_a] - lat_var) >= abs(lat[index_b] - lat_var):
        index_lat = index_b
    else:
        index_lat = index_a

    index_a = np.where(lon >= lon_var)[0][0]
    index_b = np.where(lon <= lon_var)[0][0]
    if abs(lon[index_a] - lon_var) >= abs(lon[index_b] - lon_var):
        index_lon = index_b
    else:
        index_lon = index_a

    return index_lat, index_lon


def nc_write(data, lat, lon, varname, tunits, filename):
    """
    this function write the PET on a netCDF file.

    :param: data: data to be written (time,lat,lon)
    :param: lat: latitude
    :param: lon: longitude
    :param: varname: name of the variable to be written (e.g. 'pet')
    :param: tunits: time units for the data (e.g. 'days since 1981-01-01')
    :param:filename: the file name to write the values with .nc extension

    :return:  produce a netCDF file in the same directory.
    """

    ds = Dataset(filename, mode='w', format='NETCDF4_CLASSIC')

    time = ds.createDimension('time', None)
    latitude = ds.createDimension('latitude', len(lat))
    longitude = ds.createDimension('longitude', len(lon))

    time = ds.createVariable('time', np.float32, ('time',))
    latitude = ds.createVariable('latitude', np.float32, ('latitude',))
    longitude = ds.createVariable('longitude', np.float32, ('longitude',))

    # check if the data is 2d or 3d
    if len(data.shape) == 3: # 3D array
        pet_val = ds.createVariable(varname, 'f4', ('time','latitude','longitude'), zlib=True)
        time.units = tunits  
        time.calendar = 'proleptic_gregorian'
        time[:] = np.arange(data.shape[0])
        latitude[:] = lat
        longitude [:] = lon
        pet_val[:,:,:] = data
    # this is only to write the time offsets
    elif len(data.shape) == 2: # 2D array
        pet_val = ds.createVariable(varname, 'i', ('latitude','longitude'), zlib=True)
        latitude[:] = lat
        longitude [:] = lon
        pet_val[:,:] = data
    else:
        raise ValueError('the function can only write a 2D or 3D array data!')

    ds.close()

    return None    

##********************************************************************************##
if __name__ == '__main__':
    start = dt.datetime.now()
    main()
    end = dt.datetime.now()
    print(end-start)


# In[14]:


import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

ds   = Dataset(r"C:\Users\welll\Desktop\Sidlaghatta_DRYP\source_sinks\PET\2024_daily_pet_Badenhalli_subwatershed.nc")
pet  = ds.variables['pet'][:]
lat  = ds.variables['latitude'][:]
lon  = ds.variables['longitude'][:]

# Plot NaN locations across ALL time steps
nan_count_map = np.isnan(pet).sum(axis=0)   # how many days is each cell NaN

plt.figure(figsize=(7, 5))
plt.title("NaN Count per Grid Cell (across all time steps)")
plt.imshow(nan_count_map, origin='lower',
           extent=[lon.min(), lon.max(), lat.min(), lat.max()],
           cmap='Reds')
plt.colorbar(label='Number of NaN days')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig("nan_locations.png", dpi=150)
plt.show()
print("Saved nan_locations.png")


# In[ ]:




