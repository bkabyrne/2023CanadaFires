from mpl_toolkits.basemap import Basemap, cm
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import datetime
from netCDF4 import Dataset
import glob, os
from scipy import stats
import numpy.ma as ma
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Polygon
from math import pi, cos, radians
import numpy.matlib
from pylab import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
from scipy.interpolate import griddata

# #####################################################          
#  Re-grid the GFAS CO emissions and write the data in           
#  so that it can be read by CMS-Flux                            
# ##################################################### 


def regrid_and_write_GFAS(lat_grid_size,lon_grid_size,lon_2x25,lat_2x25,area_2x25,lon,lat,area_GFAS,CO_Flux_kgCkm2s):
    #
    #for ddt in range(31+30+31): #range(days_in_year):
    #    dd = ddt+182 # looping over limited time
    #    
    #    month_in=int(month_arr[dd])
    #    day_in=int(day_arr[dd])
    #
    #    file_nc = '/u/bbyrne1/BiomassBurning_datasets/GFAS/'+str(year_in).zfill(4)+'/'+str(month_in).zfill(2)+'/'+str(year_in).zfill(4)+str(month_in).zfill(2)+str(day_in).zfill(2)+'.nc'
    #    print(file_nc)
    #    f=Dataset(file_nc,mode='r')
    #    #co2fire = f.variables['co2fire'] # kg m**-2 s**-1   Wildfire flux of Carbon Dioxide
    #    cofire = f.variables['cofire'][0,:,:] # kg m**-2 s**-1   Wildfire flux of Carbon Monoxide
    #    #injh = f.variables['injh'] # m   Injection height (from IS4FIRES)
    #    f.close()
    #
    #    CO_Flux_kgCkm2s_temp = np.zeros((1800,3600))
    #    CO_Flux_kgCkm2s_temp[:,0:1800] = cofire[:,1800:3600] * 1000. * 1000. * 12 / (12. + 16.)
    #    CO_Flux_kgCkm2s_temp[:,1800:3600] = cofire[:,0:1800] * 1000. * 1000. * 12 / (12. + 16.)
    #
    #    CO_Flux_kgCkm2s = np.flip(CO_Flux_kgCkm2s_temp,0)
    #        
    #    #                                                                                                                     
    CO_Flux_kgCkm2s_regrid = np.zeros((np.size(lat_2x25),np.size(lon_2x25)))
    #    #
    #

    lat_inventory_edges = np.append(lat-lat_grid_size/2.,lat[-1]+lat_grid_size/2.)
    lon_inventory_edges = np.append(lon-lon_grid_size/2.,lon[-1]+lon_grid_size/2.)
    
    for ii in range(np.size(lat_2x25)):
        for jj in range(np.size(lon_2x25)):
            # MERRA-2 grid midpoints:                                                                                                                                   
            lat_MERRA2_midpoint = lat_2x25[ii]
            lon_MERRA2_midpoint = lon_2x25[jj]
            #                                                                                                                                                           
            # Calculate bounding box of MERRA-2 grid cell                                                                                                               
            MERRA2_top_grid = lat_MERRA2_midpoint + 2.0/2.
            MERRA2_bottom_grid = lat_MERRA2_midpoint - 2.0/2.
            MERRA2_left_grid = lon_MERRA2_midpoint - 2.5/2.
            MERRA2_right_grid = lon_MERRA2_midpoint + 2.5/2.
            #                                                                                                                                                           
            # Find inventory edges in cell. Note need to include lower index minus 1 since lowed boundary can be outside cell                                           
            lat_edges_in_cell = np.where( np.logical_and( lat_inventory_edges >= lat_MERRA2_midpoint-2.0/2. , lat_inventory_edges <= lat_MERRA2_midpoint+2.0/2. ) )
            lon_edges_in_cell = np.where( np.logical_and( lon_inventory_edges >= lon_MERRA2_midpoint-2.5/2. , lon_inventory_edges <= lon_MERRA2_midpoint+2.5/2. ) )
            #                                                                                                                                                           
            Temp_total_flux = 0.
            Temp_total_area = 0.
            for iit in np.arange(np.min(lat_edges_in_cell)-1,np.max(lat_edges_in_cell)+1):
                for jjt in np.arange(np.min(lon_edges_in_cell)-1,np.max(lon_edges_in_cell)+1):
                    if iit < np.size(lat):
                        if jjt < np.size(lon):
                            inventory_lat_grid_center = lat[iit]
                            inventory_lon_grid_center = lon[jjt]
                            #                                                                                                                                                   
                            # Calculate bounding box of inventory grid cell                                                                                                     
                            inventory_top_grid = inventory_lat_grid_center + lat_grid_size/2.
                            inventory_bottom_grid = inventory_lat_grid_center - lat_grid_size/2.
                            inventory_left_grid = inventory_lon_grid_center - lon_grid_size/2.
                            inventory_right_grid = inventory_lon_grid_center + lon_grid_size/2.
                            #                                                                                                                                                   
                            # Find bounding box on intersect of both grids                                                                                                      
                            bottom_box = np.max([inventory_bottom_grid,MERRA2_bottom_grid])
                            top_box = np.min([inventory_top_grid,MERRA2_top_grid])
                            left_box = np.max([inventory_left_grid,MERRA2_left_grid])
                            right_box = np.min([inventory_right_grid,MERRA2_right_grid])
                            #                                                                                                                                                   
                            # Estimate area of intersect and inventory cells in degrees squared (rough estimate but good aproximation for small areas)                          
                            rough_area_small_box = (top_box - bottom_box) * (right_box - left_box)
                            rough_area_inventory = (inventory_top_grid - inventory_bottom_grid) * (inventory_right_grid - inventory_left_grid)
                            #                                                                                                                                                   
                            # Total flux                                                                                                                                        
                            Temp_total_flux += CO_Flux_kgCkm2s[iit,jjt] * area_GFAS[iit,jjt] * (rough_area_small_box/rough_area_inventory)
                            Temp_total_area += area_GFAS[iit,jjt] * (rough_area_small_box/rough_area_inventory)
            # mean flux over MERRA-2 gridcell                                                                                                                           
            CO_Flux_kgCkm2s_regrid[ii,jj] = Temp_total_flux / Temp_total_area
    # =========
    
    print('Check total emissions')
    print( np.sum(CO_Flux_kgCkm2s * area_GFAS) )
    print( np.sum(CO_Flux_kgCkm2s_regrid * area_2x25) )
    print( (np.sum(CO_Flux_kgCkm2s * area_GFAS)) / (np.sum(CO_Flux_kgCkm2s_regrid * area_2x25)) )

    return CO_Flux_kgCkm2s_regrid


def read_GFAS_data(year_in,month_in,day_in):
    
    file_nc = '/u/bbyrne1/BiomassBurning_datasets/GFAS/'+str(year_in).zfill(4)+'/'+str(month_in).zfill(2)+'/'+str(year_in).zfill(4)+str(month_in).zfill(2)+str(day_in).zfill(2)+'.nc'
    print(file_nc)
    f=Dataset(file_nc,mode='r')
    cofire = f.variables['cofire'][0,:,:] # kg m**-2 s**-1   Wildfire flux of Carbon Monoxide
    f.close()
    CO_Flux_kgCkm2s_temp = np.zeros((1800,3600))
    CO_Flux_kgCkm2s_temp[:,0:1800] = cofire[:,1800:3600] * 1000. * 1000. * 12 / (12. + 16.)
    CO_Flux_kgCkm2s_temp[:,1800:3600] = cofire[:,0:1800] * 1000. * 1000. * 12 / (12. + 16.)
    CO_Flux_kgCkm2s = np.flip(CO_Flux_kgCkm2s_temp,0)

    return CO_Flux_kgCkm2s


def write_regrided_GFAS(lat_2x25, lon_2x25, CO_regrided, year_in, month_in, day_in):
    # Write out fluxes
    nc_out = '/nobackup/bbyrne1/Flux_2x25_CO/BiomassBurn/GFAS/'+str(year_in).zfill(4)+'/'+str(month_in).zfill(2)+'/'+str(day_in).zfill(2)+'.nc'
    print(nc_out)
    #
    dataset = Dataset(nc_out,'w')
    #
    lats = dataset.createDimension('lat',np.size(lat_2x25))
    lons = dataset.createDimension('lon',np.size(lon_2x25))
    #
    postBBs = dataset.createVariable('CO_Flux', np.float64, ('lat','lon'))
    postBBs[:,:] = CO_regrided
    postBBs.units = 'kgC/km2/s'
    #
    dataset.close()
    # =============================================
    print('=========================')

    
year_in = 2023
days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
#days_in_month = np.array([0, 0, 0, 0, 0, 0, 31, 31, 30, 31, 30, 31])
days_in_year = 365

n=0
month_arr = np.zeros(days_in_year)
day_arr = np.zeros(days_in_year)
for i in range(12):
    for j in range(days_in_month[i]):
        month_arr[n] = int(i+1)
        day_arr[n] = int(j+1) 
        n=n+1


# --- Read constants ---                                                                
file_nc = '/u/bbyrne1/BiomassBurning_datasets/GFAS/2022/01/20220101.nc'
f=Dataset(file_nc,mode='r')
lon_temp = f.variables['longitude'][:]
lat_temp = f.variables['latitude'][:]
f.close()
lat_grid_size = 0.1
lon_grid_size = 0.1
lon = np.zeros(3600)
lon[0:1800] = lon_temp[1800:3600]-360.
lon[1800:3600] = lon_temp[0:1800]
lat = np.flip(lat_temp,0)
#lat_inventory_edges = np.append(lat-lat_grid_size/2.,lat[-1]+lat_grid_size/2.)
#lon_inventory_edges = np.append(lon-lon_grid_size/2.,lon[-1]+lon_grid_size/2.)
#
nc_out = '/nobackup/bbyrne1/MERRA2/2x2.5/2023/05/MERRA2.20230503.I3.2x25.nc4'
f = Dataset(nc_out,'r')
lat_2x25 = f.variables['lat'][:]
lon_2x25 = f.variables['lon'][:]
f.close()
# ------

# --- Calculate area of grids ---
earth_radius = 6371009 # in meters
lat_dist0 = pi * earth_radius / 180.0
y = lon_2x25*0. + 2.0 * lat_dist0
x= lat_2x25*0.
for i in range(np.size(lat_2x25)):
    x[i]= 2.5 * lat_dist0 * cos(radians(lat_2x25[i]))
area_2x25 = np.zeros((np.size(x),np.size(y)))
for i in range(np.size(y)):
    for j in range(np.size(x)):
        area_2x25[j,i] = np.abs(x[j]*y[i])
#
lat_dist0 = pi * earth_radius / 180.0
y = lon*0. + lat_grid_size * lat_dist0
x= lat*0.
for i in range(np.size(lat)):
    x[i]= lon_grid_size * lat_dist0 * cos(radians(lat[i]))
area_GFAS = np.zeros((np.size(x),np.size(y)))
for i in range(np.size(y)):
    for j in range(np.size(x)):
        area_GFAS[j,i] = np.abs(x[j]*y[i])
# ------       


# ===============
year = 2019
month=8
day =30
CO_Flux_kgCkm2s =  read_GFAS_data(year,month,day)
CO_regrided = regrid_and_write_GFAS(lat_grid_size,lon_grid_size,lon_2x25,lat_2x25,area_2x25,lon,lat,area_GFAS,CO_Flux_kgCkm2s)
write_regrided_GFAS(lat_2x25,lon_2x25,CO_regrided,year,month,day)
# ===============
year = 2019
month=8
day =31
CO_Flux_kgCkm2s =  read_GFAS_data(year,month,day)
CO_regrided = regrid_and_write_GFAS(lat_grid_size,lon_grid_size,lon_2x25,lat_2x25,area_2x25,lon,lat,area_GFAS,CO_Flux_kgCkm2s)
write_regrided_GFAS(lat_2x25,lon_2x25,CO_regrided,year,month,day)
# ===============
year = 2019
month=9
day =1
CO_Flux_kgCkm2s =  read_GFAS_data(year,month,day)
CO_regrided = regrid_and_write_GFAS(lat_grid_size,lon_grid_size,lon_2x25,lat_2x25,area_2x25,lon,lat,area_GFAS,CO_Flux_kgCkm2s)
write_regrided_GFAS(lat_2x25,lon_2x25,CO_regrided,year,month,day)
# ===============
year = 2019
month=9
day =2
CO_Flux_kgCkm2s =  read_GFAS_data(year,month,day)
CO_regrided = regrid_and_write_GFAS(lat_grid_size,lon_grid_size,lon_2x25,lat_2x25,area_2x25,lon,lat,area_GFAS,CO_Flux_kgCkm2s)
write_regrided_GFAS(lat_2x25,lon_2x25,CO_regrided,year,month,day)
# ===============
year = 2019
month=9
day =4
CO_Flux_kgCkm2s =  read_GFAS_data(year,month,day)
CO_regrided = regrid_and_write_GFAS(lat_grid_size,lon_grid_size,lon_2x25,lat_2x25,area_2x25,lon,lat,area_GFAS,CO_Flux_kgCkm2s)
write_regrided_GFAS(lat_2x25,lon_2x25,CO_regrided,year,month,day)

# ===============
year = 2020
month=5
day =26
CO_Flux_kgCkm2s =  read_GFAS_data(year,month,day)
CO_regrided = regrid_and_write_GFAS(lat_grid_size,lon_grid_size,lon_2x25,lat_2x25,area_2x25,lon,lat,area_GFAS,CO_Flux_kgCkm2s)
write_regrided_GFAS(lat_2x25,lon_2x25,CO_regrided,year,month,day)
# ===============
year = 2021
month=4
day =21
CO_Flux_kgCkm2s =  read_GFAS_data(year,month,day)
CO_regrided = regrid_and_write_GFAS(lat_grid_size,lon_grid_size,lon_2x25,lat_2x25,area_2x25,lon,lat,area_GFAS,CO_Flux_kgCkm2s)
write_regrided_GFAS(lat_2x25,lon_2x25,CO_regrided,year,month,day)
# ===============
year = 2021
month=5
day =4
CO_Flux_kgCkm2s =  read_GFAS_data(year,month,day)
CO_regrided = regrid_and_write_GFAS(lat_grid_size,lon_grid_size,lon_2x25,lat_2x25,area_2x25,lon,lat,area_GFAS,CO_Flux_kgCkm2s)
write_regrided_GFAS(lat_2x25,lon_2x25,CO_regrided,year,month,day)
# ===============
year = 2021
month=5
day =5
CO_Flux_kgCkm2s =  read_GFAS_data(year,month,day)
CO_regrided = regrid_and_write_GFAS(lat_grid_size,lon_grid_size,lon_2x25,lat_2x25,area_2x25,lon,lat,area_GFAS,CO_Flux_kgCkm2s)
write_regrided_GFAS(lat_2x25,lon_2x25,CO_regrided,year,month,day)
# ===============
year = 2022
month=8
day =22
CO_Flux_kgCkm2s =  read_GFAS_data(year,month,day)
CO_regrided = regrid_and_write_GFAS(lat_grid_size,lon_grid_size,lon_2x25,lat_2x25,area_2x25,lon,lat,area_GFAS,CO_Flux_kgCkm2s)
write_regrided_GFAS(lat_2x25,lon_2x25,CO_regrided,year,month,day)
# ===============
year = 2022
month=9
day =1
CO_Flux_kgCkm2s =  read_GFAS_data(year,month,day)
CO_regrided = regrid_and_write_GFAS(lat_grid_size,lon_grid_size,lon_2x25,lat_2x25,area_2x25,lon,lat,area_GFAS,CO_Flux_kgCkm2s)
write_regrided_GFAS(lat_2x25,lon_2x25,CO_regrided,year,month,day)
# ===============
year = 2022
month=9
day =2
CO_Flux_kgCkm2s =  read_GFAS_data(year,month,day)
CO_regrided = regrid_and_write_GFAS(lat_grid_size,lon_grid_size,lon_2x25,lat_2x25,area_2x25,lon,lat,area_GFAS,CO_Flux_kgCkm2s)
write_regrided_GFAS(lat_2x25,lon_2x25,CO_regrided,year,month,day)
# ===============
year = 2022
month=9
day =4
CO_Flux_kgCkm2s =  read_GFAS_data(year,month,day)
CO_regrided = regrid_and_write_GFAS(lat_grid_size,lon_grid_size,lon_2x25,lat_2x25,area_2x25,lon,lat,area_GFAS,CO_Flux_kgCkm2s)
write_regrided_GFAS(lat_2x25,lon_2x25,CO_regrided,year,month,day)



#year_arr = np.arange(2018-2009)+2009
#for year in year_arr:
#    # -----
#    if (year % 4) == 0:
#        days_in_months = np.array([31,29,31,30,31,30,31,31,30,31,30,31])
#    else:
#        days_in_months = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
#    # -----
#    for month in range(12):
#        for day in range(days_in_month[month]):
#            # Read GFAS CO emissions
#            CO_Flux_kgCkm2s =  read_GFAS_data(year,month+1,day+1)
#            # Regrid to 2x2.5
#            CO_regrided = regrid_and_write_GFAS(lat_grid_size,lon_grid_size,lon_2x25,lat_2x25,area_2x25,lon,lat,area_GFAS,CO_Flux_kgCkm2s)
#            # Write the regrided CO fluxes
#            write_regrided_GFAS(lat_2x25,lon_2x25,CO_regrided,year,month+1,day+1)
