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
#  Re-grid the QFED CO emissions and write the data in   
#  so that it can be read by CMS-Flux                     
# #####################################################


# --- Read constants ---                                                              
file_nc = '/u/bbyrne1/BiomassBurning_datasets/QFED_v2.6r1_0.25res/Y2019/M01/qfed2.emis_co.061.20190101.nc4'
f=Dataset(file_nc,mode='r')
lon = f.variables['lon'][:]
lat = f.variables['lat'][:]
f.close()
lat_grid_size = 0.25
lon_grid_size = 0.3125
lat_inventory_edges = np.append(lat-lat_grid_size/2.,lat[-1]+lat_grid_size/2.)
lon_inventory_edges = np.append(lon-lon_grid_size/2.,lon[-1]+lon_grid_size/2.)
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
area_QFED = np.zeros((np.size(x),np.size(y)))
for i in range(np.size(y)):
    for j in range(np.size(x)):
        area_QFED[j,i] = np.abs(x[j]*y[i])
# ------
        

        
def regrid_QFED_2x25(year_in,days_in_month,days_in_year,lat_2x25,lon_2x25,area_2x25,lat,lon,area_QFED,lat_grid_size,lon_grid_size):

    # This function reads in daily QFED fluxes, performs regriding and writes the regrided data   

    # Create arrays for day of year, month of year 
    
    n=0
    month_arr = np.zeros(days_in_year)
    day_arr = np.zeros(days_in_year)
    for i in range(12):
        for j in range(days_in_month[i]):
            month_arr[n] = int(i+1)
            day_arr[n] = int(j+1) 
            n=n+1

    # Loop over days
    for dd in range(days_in_year):
        month_in=int(month_arr[dd])
        day_in=int(day_arr[dd])

        # ---- Read QFED data 
        #    Source: NASA/GSFC/GMAO GEOS-5 Aerosol Group
        #    Title: QFED Level3b v2.5 (@CVSTAG) Gridded Emission Estimates
        #    Contact: arlindo.dasilva@nasa.gov
        #    History: File written by GFIO v1.0.8
        #    dimensions(sizes): lon(1152), lat(721), time(1)
        file_nc = '/u/bbyrne1/BiomassBurning_datasets/QFED_v2.6r1_0.25res/Y'+str(year_in).zfill(4)+'/M'+str(month_in).zfill(2)+'/qfed2.emis_co.061.'+str(year_in).zfill(4)+str(month_in).zfill(2)+str(day_in).zfill(2)+'.nc4'
        print(file_nc)
        f=Dataset(file_nc,mode='r')
        biomass = np.squeeze(f.variables['biomass'][:]) # CO Biomass Emissions # units: kg s-1 m-2 ---- Seems like kgCO
        f.close()

        # Convert to CMS-Flux units
        CO_Flux_kgCkm2s = biomass * 1000. * 1000. * (12./28.)
        
        # Write out fluxes                                                                                                   \
        CO_Flux_kgCkm2s_regrid = np.zeros((np.size(lat_2x25),np.size(lon_2x25)))
        for ii in range(np.size(lat_2x25)):
            for jj in range(np.size(lon_2x25)):
                # MERRA-2 grid midpoints:                                                                                                          
                lat_MERRA2_midpoint = lat_2x25[ii]
                lon_MERRA2_midpoint = lon_2x25[jj]
                #                                                                                                                                  
                # Calculate bounding box of MERRA-2 grid cell                                                                                      
                MERRA2_top_grid = lat_MERRA2_midpoint + 2./2.
                MERRA2_bottom_grid = lat_MERRA2_midpoint - 2./2.
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
                                Temp_total_flux += CO_Flux_kgCkm2s[iit,jjt] * area_QFED[iit,jjt] * (rough_area_small_box/rough_area_inventory)
                                Temp_total_area += area_QFED[iit,jjt] * (rough_area_small_box/rough_area_inventory)
                # mean flux over MERRA-2 gridcell                                                                                                  
                CO_Flux_kgCkm2s_regrid[ii,jj] = Temp_total_flux / Temp_total_area


        # =========                                                                                                                                
        print('Check total emissions')
        print( np.sum(CO_Flux_kgCkm2s * area_QFED) )  # lat[191:546] # lon[463:786]                                                                
        print( np.sum(CO_Flux_kgCkm2s_regrid * area_2x25) )
        print( (np.sum(CO_Flux_kgCkm2s * area_QFED)) / (np.sum(CO_Flux_kgCkm2s_regrid * area_2x25)) )
        #                                                                                                                                                                                 
        # Write out fluxes                                                                                                                                                                
        nc_out = '/nobackup/bbyrne1/Flux_2x25_CO/BiomassBurn/QFED/'+str(year_in).zfill(4)+'/'+str(month_in).zfill(2)+'/'+str(day_in).zfill(2)+'.nc'
        print(nc_out)
        #                                                                                                                                                                                 
        dataset = Dataset(nc_out,'w')
        #                                                                                                                                                                                 
        lats = dataset.createDimension('lat',np.size(lat_2x25))
        lons = dataset.createDimension('lon',np.size(lon_2x25))
        #                                                                                                                                                                                 
        postBBs = dataset.createVariable('CO_Flux', np.float64, ('lat','lon'))
        postBBs[:,:] = CO_Flux_kgCkm2s_regrid
        postBBs.units = 'kgC/km2/s'
        #                                                                                                                                                                                 
        dataset.close()
    # =============================================                                                                                                                                   
# ---------- Function calls 


#year_in = 2002
#days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
#days_in_year = 365
#regrid_QFED_2x25(year_in,days_in_month,days_in_year,lat_2x25,lon_2x25,area_2x25,lat,lon,area_QFED,lat_grid_size,lon_grid_size)

#year_in = 2003
#days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
#days_in_year = 365
#regrid_QFED_2x25(year_in,days_in_month,days_in_year,lat_2x25,lon_2x25,area_2x25,lat,lon,area_QFED,lat_grid_size,lon_grid_size)

#year_in = 2004 # -- LEAP YEAR
#days_in_month = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
#days_in_year = 366
#regrid_QFED_2x25(year_in,days_in_month,days_in_year,lat_2x25,lon_2x25,area_2x25,lat,lon,area_QFED,lat_grid_size,lon_grid_size)

#year_in = 2005
#days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
#days_in_year = 365
#regrid_QFED_2x25(year_in,days_in_month,days_in_year,lat_2x25,lon_2x25,area_2x25,lat,lon,area_QFED,lat_grid_size,lon_grid_size)

#year_in = 2006
#days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
#days_in_year = 365
#regrid_QFED_2x25(year_in,days_in_month,days_in_year,lat_2x25,lon_2x25,area_2x25,lat,lon,area_QFED,lat_grid_size,lon_grid_size)

#year_in = 2007
#days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
#days_in_year = 365
#regrid_QFED_2x25(year_in,days_in_month,days_in_year,lat_2x25,lon_2x25,area_2x25,lat,lon,area_QFED,lat_grid_size,lon_grid_size)

#year_in = 2008 # -- LEAP YEAR
#days_in_month = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
#days_in_year = 366
#regrid_QFED_2x25(year_in,days_in_month,days_in_year,lat_2x25,lon_2x25,area_2x25,lat,lon,area_QFED,lat_grid_size,lon_grid_size)

#year_in = 2009
#days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
#days_in_year = 365
#regrid_QFED_2x25(year_in,days_in_month,days_in_year,lat_2x25,lon_2x25,area_2x25,lat,lon,area_QFED,lat_grid_size,lon_grid_size)

#year_in = 2010
#days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
#days_in_year = 365
#regrid_QFED_2x25(year_in,days_in_month,days_in_year,lat_2x25,lon_2x25,area_2x25,lat,lon,area_QFED,lat_grid_size,lon_grid_size)

#year_in = 2011
#days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
#days_in_year = 365
#regrid_QFED_2x25(year_in,days_in_month,days_in_year,lat_2x25,lon_2x25,area_2x25,lat,lon,area_QFED,lat_grid_size,lon_grid_size)

#year_in = 2012 # -- LEAP YEAR
#days_in_month = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
#days_in_year = 366
#regrid_QFED_2x25(year_in,days_in_month,days_in_year,lat_2x25,lon_2x25,area_2x25,lat,lon,area_QFED,lat_grid_size,lon_grid_size)

#year_in = 2013
#days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
#days_in_year = 365
#regrid_QFED_2x25(year_in,days_in_month,days_in_year,lat_2x25,lon_2x25,area_2x25,lat,lon,area_QFED,lat_grid_size,lon_grid_size)

#year_in = 2014
#days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
#days_in_year = 365
#regrid_QFED_2x25(year_in,days_in_month,days_in_year,lat_2x25,lon_2x25,area_2x25,lat,lon,area_QFED,lat_grid_size,lon_grid_size)

#year_in = 2015
#days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
#days_in_year = 365
#regrid_QFED_2x25(year_in,days_in_month,days_in_year,lat_2x25,lon_2x25,area_2x25,lat,lon,area_QFED,lat_grid_size,lon_grid_size)

#year_in = 2016 # -- LEAP YEAR
#days_in_month = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
#days_in_year = 366
#regrid_QFED_2x25(year_in,days_in_month,days_in_year,lat_2x25,lon_2x25,area_2x25,lat,lon,area_QFED,lat_grid_size,lon_grid_size)

#year_in = 2017
#days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
#days_in_year = 365
#regrid_QFED_2x25(year_in,days_in_month,days_in_year,lat_2x25,lon_2x25,area_2x25,lat,lon,area_QFED,lat_grid_size,lon_grid_size)



#year_in = 2018
#days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
#days_in_year = 365
#regrid_QFED_2x25(year_in,days_in_month,days_in_year,lat_2x25,lon_2x25,area_2x25,lat,lon,area_QFED,lat_grid_size,lon_grid_size)
    
#year_in = 2019
#days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
#days_in_year = 365
#regrid_QFED_2x25(year_in,days_in_month,days_in_year,lat_2x25,lon_2x25,area_2x25,lat,lon,area_QFED,lat_grid_size,lon_grid_size)

#year_in = 2020   # -- LEAP YEAR
#days_in_month = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
#days_in_year = 366
#regrid_QFED_2x25(year_in,days_in_month,days_in_year,lat_2x25,lon_2x25,area_2x25,lat,lon,area_QFED,lat_grid_size,lon_grid_size)

#year_in = 2021
#days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
#days_in_year = 365
#regrid_QFED_2x25(year_in,days_in_month,days_in_year,lat_2x25,lon_2x25,area_2x25,lat,lon,area_QFED,lat_grid_size,lon_grid_size)

#year_in = 2022
#days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
#days_in_year = 365
#regrid_QFED_2x25(year_in,days_in_month,days_in_year,lat_2x25,lon_2x25,area_2x25,lat,lon,area_QFED,lat_grid_size,lon_grid_size)

year_in = 2023
days_in_month = np.array([0, 0, 0, 0, 0, 0, 0, 31, 30, 31, 30, 31])
days_in_year = 365
regrid_QFED_2x25(year_in,days_in_month,days_in_year,lat_2x25,lon_2x25,area_2x25,lat,lon,area_QFED,lat_grid_size,lon_grid_size)
