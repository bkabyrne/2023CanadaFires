# --- import modules ---                    
from mpl_toolkits.basemap import Basemap, cm
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import datetime
from netCDF4 import Dataset
import glob, os
from scipy import stats
from datetime import date
import numpy.ma as ma
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Polygon
from math import pi, cos, radians
import numpy.matlib
from pylab import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.patches as mpatches

'''
------ write_daily_YHx_prior.py

 This program reads the TROPOMI co-samples and writes them to annual files.
 The output of this file is intended to be used to generate the Heald uncertainty
 estimates on the obs.

'''


def read_obs(nc_file_TROPOMI,nc_file_GFED,nc_file_GFAS,nc_file_QFED,doy):
    #
    if os.path.isfile(nc_file_GFED):
        #
        f=Dataset(nc_file_TROPOMI,mode='r')
        longitudev=f.variables['longitude'][:]
        latitudev=f.variables['latitude'][:]
        xCO_uncertaintyv=f.variables['xCO-uncertainty'][:]
        f.close()
        
        # ===================================
        if os.path.isfile(nc_file_GFED):
            f=Dataset(nc_file_GFED,mode='r')
            Y_GFEDv=np.squeeze(f.variables['Y'][:]*1e9)
            Hx_GFEDv=np.squeeze(f.variables['HX'][:]*1e9)
            f.close()
        else:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        #
        if os.path.isfile(nc_file_GFAS):
            f=Dataset(nc_file_GFAS,mode='r')
            Hx_GFASv=np.squeeze(f.variables['HX'][:]*1e9)
            f.close()
        else:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        #
        if os.path.isfile(nc_file_QFED):
            f=Dataset(nc_file_QFED,mode='r')
            Hx_QFEDv=np.squeeze(f.variables['HX'][:]*1e9)
            f.close()
        else:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        # ===================================
        
        doy_arrv = Hx_QFEDv*0.+doy

        if np.size(Hx_QFEDv) != np.size(latitudev):
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            return longitudev, latitudev, xCO_uncertaintyv, Y_GFEDv, Hx_GFEDv, Hx_GFASv, Hx_QFEDv, doy_arrv
    return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def combine_annual_obs(year,month_arr,day_arr,doy_to_include,OH_fields):
    #
    for ii in doy_to_include:
        doy = ii
        if OH_fields == 'GC':
            nc_file_GFED = '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-2023/Prior_OH_COinv_GFED_'+str(year).zfill(4)+'/OBSF/TROPOMI_XCO_2x25/'+str(year).zfill(4)+'/'+str(int(month_arr[ii])).zfill(2)+'/'+str(int(day_arr[ii])).zfill(2)+'.nc'
            nc_file_GFAS = '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-2023/Prior_OH_COinv_GFAS_'+str(year).zfill(4)+'/OBSF/TROPOMI_XCO_2x25/'+str(year).zfill(4)+'/'+str(int(month_arr[ii])).zfill(2)+'/'+str(int(day_arr[ii])).zfill(2)+'.nc'
            nc_file_QFED = '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-2023/Prior_OH_COinv_QFED_'+str(year).zfill(4)+'/OBSF/TROPOMI_XCO_2x25/'+str(year).zfill(4)+'/'+str(int(month_arr[ii])).zfill(2)+'/'+str(int(day_arr[ii])).zfill(2)+'.nc'
            
        elif OH_fields == 'Kazu':
            nc_file_GFED = '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-2023/FWD_CO_GFED_'+str(year).zfill(4)+'/OBSF/TROPOMI_XCO_2x25/'+str(year).zfill(4)+'/'+str(int(month_arr[ii])).zfill(2)+'/'+str(int(day_arr[ii])).zfill(2)+'.nc'
            nc_file_GFAS = '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-2023/FWD_CO_GFAS_'+str(year).zfill(4)+'/OBSF/TROPOMI_XCO_2x25/'+str(year).zfill(4)+'/'+str(int(month_arr[ii])).zfill(2)+'/'+str(int(day_arr[ii])).zfill(2)+'.nc'
            nc_file_QFED = '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-2023/FWD_CO_QFED_'+str(year).zfill(4)+'/OBSF/TROPOMI_XCO_2x25/'+str(year).zfill(4)+'/'+str(int(month_arr[ii])).zfill(2)+'/'+str(int(day_arr[ii])).zfill(2)+'.nc'
        #
        nc_file_TROPOMI = '/nobackup/bbyrne1/TROPOMI_XCO_2x25/'+str(year).zfill(4)+'/'+str(int(month_arr[ii])).zfill(2)+'/'+str(int(day_arr[ii])).zfill(2)+'.nc'
        #
        print(nc_file_GFED)
        #
        longitudet, latitudet, xCO_uncertaintyt, Y_GFEDt, Hx_GFEDt, Hx_GFASt, Hx_QFEDt, doy_arrt = read_obs(nc_file_TROPOMI,nc_file_GFED,nc_file_GFAS,nc_file_QFED,doy)

        if 'longitudeo' in locals():
            longitudeo = np.append(longitudeo,longitudet)
            latitudeo = np.append(latitudeo,latitudet)
            xCO_uncertaintyo = np.append(xCO_uncertaintyo,xCO_uncertaintyt)
            Y_GFEDo = np.append(Y_GFEDo,Y_GFEDt)
            Hx_GFEDo = np.append(Hx_GFEDo,Hx_GFEDt)
            Hx_GFASo = np.append(Hx_GFASo,Hx_GFASt)
            Hx_QFEDo = np.append(Hx_QFEDo,Hx_QFEDt)
            doy_arro = np.append(doy_arro,doy_arrt)
        else:
            longitudeo = longitudet * (1.)
            latitudeo = latitudet * (1.)
            xCO_uncertaintyo = xCO_uncertaintyt * (1.)
            Y_GFEDo = Y_GFEDt * (1.)
            Hx_GFEDo = Hx_GFEDt * (1.)
            Hx_GFASo = Hx_GFASt * (1.)
            Hx_QFEDo = Hx_QFEDt * (1.)
            doy_arro = doy_arrt * (1.)

    return longitudeo, latitudeo, xCO_uncertaintyo, Y_GFEDo, Hx_GFEDo, Hx_GFASo, Hx_QFEDo, doy_arro




def write_cosamples(year,longitude,latitude,xCO_uncertainty,Y_GFED,Hx_GFED,Hx_GFAS,Hx_QFED,doy_arr,OH_fields):
    if OH_fields == 'GC':
            file_out = 'TROPOMI_OH_prior_YHx_'+str(year).zfill(4)+'.nc'
    elif OH_fields == 'Kazu':
        file_out = 'TROPOMI_prior_YHx_'+str(year).zfill(4)+'.nc'
    print(file_out)
    dataset = Dataset(file_out,'w')
    nSamples = dataset.createDimension('nSamples',np.size(longitude))
    longitudes = dataset.createVariable('longitude', np.float64, ('nSamples',))
    longitudes[:]=longitude
    latitudes = dataset.createVariable('latitude', np.float64, ('nSamples',))
    latitudes[:]=latitude
    xCO_uncertaintys = dataset.createVariable('xCO_uncertainty', np.float64, ('nSamples',))
    xCO_uncertaintys[:]=xCO_uncertainty
    Y_GFEDs = dataset.createVariable('Y_GFED', np.float64, ('nSamples',))
    Y_GFEDs[:]=Y_GFED
    Hx_GFEDs = dataset.createVariable('Hx_GFED', np.float64, ('nSamples',))
    Hx_GFEDs[:]=Hx_GFED
    Hx_GFASs = dataset.createVariable('Hx_GFAS', np.float64, ('nSamples',))
    Hx_GFASs[:]=Hx_GFAS
    Hx_QFEDs = dataset.createVariable('Hx_QFED', np.float64, ('nSamples',))
    Hx_QFEDs[:]=Hx_QFED
    doy_arrs = dataset.createVariable('doy_arr', np.float64, ('nSamples',))
    doy_arrs[:]=doy_arr
    dataset.close()


def create_day_and_month_arrays(days_in_month,days_in_year):   
    #
    # Create array of month and day-of-month for each day of the year
    #
    n=0
    month_arr = np.zeros(days_in_year)
    day_arr = np.zeros(days_in_year)
    for i in range(12):
        for j in range(days_in_month[i]):
            month_arr[n] = int(i + 1)
            day_arr[n] = int(j + 1)
            n=n+1
    #
    return month_arr, day_arr


if __name__ == "__main__":

    # data for 02/29/2020 is missing, so we treat Feb as havin 28 days
    days_in_month = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
    days_in_year = 365
    # Create array of month and day-of-month for each day of the year
    month_arr, day_arr = create_day_and_month_arrays(days_in_month,days_in_year)
    
    doy_to_include = np.arange(273-90 -1 )+90

    for year in range(2023,2024):
        # Read TROPOMI for each day in range and combine into single vector
        longitude, latitude, xCO_uncertainty, Y_GFED, Hx_GFED, Hx_GFAS, Hx_QFED, doy_arr = combine_annual_obs(year,month_arr,day_arr,doy_to_include,'GC')
        
        II_finite = np.where(np.isfinite(longitude))
        longitude_out = longitude[II_finite]
        latitude_out = latitude[II_finite]
        xCO_uncertainty_out = xCO_uncertainty[II_finite]
        Y_GFED_out = Y_GFED[II_finite]
        Hx_GFED_out = Hx_GFED[II_finite]
        Hx_GFAS_out = Hx_GFAS[II_finite]
        Hx_QFED_out = Hx_QFED[II_finite]
        doy_arr_out = doy_arr[II_finite]
        print('DOY range:')
        print(np.nanmin(doy_arr_out))
        print(np.nanmax(doy_arr_out))
        # Write the co-samples
        write_cosamples(year,longitude_out,latitude_out,xCO_uncertainty_out,Y_GFED_out,Hx_GFED_out,Hx_GFAS_out,Hx_QFED_out,doy_arr_out,'GC')
