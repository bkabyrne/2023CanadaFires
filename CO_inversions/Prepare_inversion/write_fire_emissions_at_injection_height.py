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

# ######################################################
#
#  This program writes the prior fluxes used for the 3-day
#  inversions. The component emissions are combined and the
#  uncertainties are constructed.
#
# ######################################################


def calculate_daily_injh_and_CO_flux(year,month,day,CO_flux_BB_prior,CO_flux_BB_post):

    file_name = '/u/bbyrne1/inj_level/'+str(year).zfill(4)+'/'+str(month).zfill(2)+'/'+str(day).zfill(2)+'.nc'
    f=Dataset(file_name,mode='r')
    injh_level = np.repeat(f.variables['injh_level'][:][np.newaxis,:,:], 8, axis=0)
    f.close()

    # Diurnal information is from GFED                                                                           
    nc_GFED_diurnal_3hr = '/nobackupp17/bbyrne1/GFED41s_2x25_diurnal_scale/2022/'+str(month).zfill(2)+'/'+str(day).zfill(2)+'.nc'
    f=Dataset(nc_GFED_diurnal_3hr,mode='r')
    GFED_diurnal_3hr = f.variables['diurnal'][:]
    f.close()
    
    CO_flux_BB_prior_3hr = GFED_diurnal_3hr*np.repeat(CO_flux_BB_prior[np.newaxis,:,:], 8, axis=0)
    CO_flux_BB_post_3hr = GFED_diurnal_3hr*np.repeat(CO_flux_BB_post[np.newaxis,:,:], 8, axis=0)

    return CO_flux_BB_prior_3hr, CO_flux_BB_post_3hr, injh_level

    
def create_time_arrays(year):
    
    if np.logical_and(year % 4 == 0, year % 100 != 0):
        month_lengths = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        ndays = 366
    else:
        month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        ndays = 365

    day_of_year = np.zeros(ndays)
    month_of_year = np.zeros(ndays)
    day_of_month = np.zeros(ndays)
    
    i=0
    for month in range(12):
        for day in range(month_lengths[month]):
            day_of_year[i] = i+1
            month_of_year[i] = month+1
            day_of_month[i] = day+1
            i += 1

    return day_of_year.astype(int), month_of_year.astype(int), day_of_month.astype(int)
    # ===============================================


def write_fluxes_noUNC(inventory,year,month,day, CO_flux_BB_3hr, version, injh_level,optDays,rep=0):
    #
    # =========================================
    # Write out fluxes and SF uncertainty
    # =========================================
    #
    if rep==1:
        nc_out = '/nobackup/bbyrne1/Flux_2x25_CO/BiomassBurn/'+inventory+'_rep_'+version+'_'+optDays+'_injh/'+str(year).zfill(4)+'/'+str(month).zfill(2)+'/'+str(day).zfill(2)+'.nc'
    else:
        nc_out = '/nobackup/bbyrne1/Flux_2x25_CO/BiomassBurn/'+inventory+'_'+version+'_'+optDays+'_injh/'+str(year).zfill(4)+'/'+str(month).zfill(2)+'/'+str(day).zfill(2)+'.nc'
    print(nc_out)
    #
    dataset = Dataset(nc_out,'w')
    #
    times = dataset.createDimension('time',8)
    lats = dataset.createDimension('lat',91)
    lons = dataset.createDimension('lon',144)
    #
    postBBs = dataset.createVariable('CO_Flux', np.float64, ('time','lat','lon'))
    postBBs[:,:,:] = CO_flux_BB_3hr
    postBBs.units = 'kgC/km2/s'
    #
    heights = dataset.createVariable('HEIGHT',np.float64,('time','lat','lon'))
    heights[:,:,:]=injh_level
    heights.units='level'
    dataset.close()


year = 2023

day_of_year, month_of_year, day_of_month = create_time_arrays(year)
# Only Apr-Sep
I_AprSep = np.where(np.logical_and(month_of_year>=4,month_of_year<=9))


prior_model =''
for prior_model in ['GFED','QFED','GFAS']:
    for rep in [0,1]:
        for optDays in ['3day','7day']:
            #rep=0
            #optDays='7day'
    
            # Reads whole year
            if rep==1:
                file_name = '/u/bbyrne1/python_codes/Canada_Fires_2023/Byrne_etal_codes/plot_figures/data_for_figures/TROPOMI_rep_'+prior_model+'_COinv_2x25_'+str(year).zfill(4)+'_fire_'+optDays+'.nc'
            else:
                file_name = '/u/bbyrne1/python_codes/Canada_Fires_2023/Byrne_etal_codes/plot_figures/data_for_figures/TROPOMI_'+prior_model+'_COinv_2x25_'+str(year).zfill(4)+'_fire_'+optDays+'.nc'
            print(file_name)
            f=Dataset(file_name,mode='r')
            CO_flux_BB_prior = f.variables['CO_prior'][:] * 1000./(60.*60.*24.) # gC/m2/d --> kgC/km2/s
            CO_flux_BB_post = f.variables['CO_post'][:] * 1000./(60.*60.*24.) # gC/m2/d --> kgC/km2/s
            f.close()

            #fig = plt.figure(1)                                                                                                                                     
            #plt.plot(np.sum(np.sum(CO_flux_BB_prior,1),1),'k:')                                                                                                   
            #plt.plot(np.sum(np.sum(CO_flux_BB_post,1),1),'b')                                                                                                     
            #plt.savefig('Fire_test.png')                                                                                                                      
            #stophere 

            for i in day_of_year[I_AprSep]:
    
                CO_flux_BB_prior_3hr, CO_flux_BB_post_3hr, injh_level = calculate_daily_injh_and_CO_flux( year, month_of_year[i-1], day_of_month[i-1], CO_flux_BB_prior[i-1,:,:], CO_flux_BB_post[i-1,:,:])
                
                write_fluxes_noUNC( prior_model, year, month_of_year[i-1], day_of_month[i-1], CO_flux_BB_prior_3hr, 'prior', injh_level,optDays,rep=rep)
                write_fluxes_noUNC( prior_model, year, month_of_year[i-1], day_of_month[i-1], CO_flux_BB_post_3hr, 'post', injh_level,optDays,rep=rep)
    
