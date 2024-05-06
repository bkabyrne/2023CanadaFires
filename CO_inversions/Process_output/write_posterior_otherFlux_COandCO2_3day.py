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

'''
----- write_posterior_otherFlux_COandCO2_3day.py

Writes posterior fluxes for non-fire emissions (Fossil & Biogenic)
for inversions with 3-Day optimizatin

'''

def calc_fluxes(SF,nc_FF_fire,nc_Biogenic_fire,prior_model):
    #
    # ================================
    #
    # This function reads in posterior scale factors and daily prior fluxes
    # during Apr-Sep then calculates timeseries of prior and posterior fluxes
    #
    # inputs:
    #  - SF: array of posterior scale factors
    #  - nc_FF_fire: path to prior Fossil Fuel flux directory
    #  - nc_Biogenic_fire: path to prior Biogenic CO flux directory
    #
    # outputs:
    #  - FF_Prior_flux: timeseries of prior Fossil Fuel fluxes (time,lat,lon) 
    #  - FF_Posterior_flux: timeseries of posterior Fossil Fuel fluxes (time,lat,lon) 
    #  - Biogenic_Prior_flux: timeseries of prior Biogenic CO fluxes (time,lat,lon) 
    #  - Biogenic_Posterior_flux: timeseries of posterior Biogenic CO fluxes (time,lat,lon) 
    #
    # ================================
    #
    # Track day of year
    days_in_month = np.array([30, 31, 30, 31, 31, 30, 31, 30, 31])
    days_in_month_cum = np.zeros(13)
    for i in range(13-3):
        days_in_month_cum[i] = np.sum(days_in_month[0:i])
    #
    # Apply scale factors to fluxes
    FF_Prior_flux = np.zeros((365,np.size(lat),np.size(lon)))
    FF_Posterior_flux = np.zeros((365,np.size(lat),np.size(lon)))
    Biogenic_Prior_flux = np.zeros((365,np.size(lat),np.size(lon)))
    Biogenic_Posterior_flux = np.zeros((365,np.size(lat),np.size(lon)))
    for nnt in range(183):
        nn = nnt
        #
        SF_index = int(np.floor(nn/3.))
        month = np.argmax( nn < days_in_month_cum)+3
        day = int(nn-days_in_month_cum[month-1-3])
        #
        print(str(month).zfill(2)+'/'+str(day+1).zfill(2))
        file_in = nc_FF_fire+str(month).zfill(2)+'/'+str(day+1).zfill(2)+'.nc'
        f=Dataset(file_in,mode='r')
        FF_Prior_flux[nn+31+28+31,:,:] = f.variables['CO_Flux'][:]  * (60.*60.*24.)/1000. # kgC/km2/s -> gC/m2/d
        FF_Posterior_flux[nn+31+28+31,:,:] = FF_Prior_flux[nn+31+28+31,:,:] * SF[SF_index,:,:]
        #
        file_in = nc_Biogenic_fire+str(month).zfill(2)+'/'+str(day+1).zfill(2)+'.nc'
        f=Dataset(file_in,mode='r')
        Biogenic_Prior_flux[nn+31+28+31,:,:] = f.variables['CO_Flux'][:]  * (60.*60.*24.)/1000. # kgC/km2/s -> gC/m2/d 
        Biogenic_Posterior_flux[nn+31+28+31,:,:] = Biogenic_Prior_flux[nn+31+28+31,:,:] * SF[SF_index,:,:]
    #
    return FF_Prior_flux, FF_Posterior_flux, Biogenic_Prior_flux, Biogenic_Posterior_flux


def calculate_2x25_grid_area():
    #
    # =============================
    # Returns grid area (lat,lon) in m2
    # =============================
    #
    grid_area_2x25 = np.array([2.70084e+08,  2.16024e+09,  4.31787e+09,  6.47023e+09,  8.61471e+09,
                               1.07487e+10,  1.28696e+10,  1.49748e+10,  1.70617e+10,  1.91279e+10,
                               2.11708e+10,  2.31879e+10,  2.51767e+10,  2.71348e+10,  2.90599e+10,
                               3.09496e+10,  3.28016e+10,  3.46136e+10,  3.63835e+10,  3.81090e+10,
                               3.97881e+10,  4.14187e+10,  4.29988e+10,  4.45266e+10,  4.60001e+10,
                               4.74175e+10,  4.87772e+10,  5.00775e+10,  5.13168e+10,  5.24935e+10,
                               5.36063e+10,  5.46538e+10,  5.56346e+10,  5.65477e+10,  5.73920e+10,
                               5.81662e+10,  5.88696e+10,  5.95014e+10,  6.00606e+10,  6.05466e+10,
                               6.09588e+10,  6.12968e+10,  6.15601e+10,  6.17484e+10,  6.18615e+10,
                               6.18992e+10,  6.18615e+10,  6.17484e+10,  6.15601e+10,  6.12968e+10,
                               6.09588e+10,  6.05466e+10,  6.00606e+10,  5.95014e+10,  5.88696e+10,
                               5.81662e+10,  5.73920e+10,  5.65477e+10,  5.56346e+10,  5.46538e+10,
                               5.36063e+10,  5.24935e+10,  5.13168e+10,  5.00775e+10,  4.87772e+10,
                               4.74175e+10,  4.60001e+10,  4.45266e+10,  4.29988e+10,  4.14187e+10,
                               3.97881e+10,  3.81090e+10,  3.63835e+10,  3.46136e+10,  3.28016e+10,
                               3.09496e+10,  2.90599e+10,  2.71348e+10,  2.51767e+10,  2.31879e+10,
                               2.11708e+10,  1.91279e+10,  1.70617e+10,  1.49748e+10,  1.28696e+10,
                               1.07487e+10,  8.61471e+09,  6.47023e+09,  4.31787e+09,  2.16024e+09,
                               2.70084e+08])
    #
    grid_area_2x25_arr = np.zeros((91,144))
    for ii in range(144):
        grid_area_2x25_arr[:,ii] = grid_area_2x25
    #
    return grid_area_2x25_arr


def write_dataset(nc_out, FF_Flux_prior, FF_Flux_post, Biogenic_Flux_prior, Biogenic_Flux_post):
    #
    # =============================
    # Write prior/posterior fluxes to netcdf
    # =============================
    #
    # Read grid to write out area (m2)
    grid_area_2x25 = calculate_2x25_grid_area()
    #
    # Write out data
    dataset = Dataset(nc_out,'w')
    print(nc_out)
    times = dataset.createDimension('time',365)
    lats = dataset.createDimension('lat',91)
    lons = dataset.createDimension('lon',144)
    gridareas = dataset.createVariable('grid_area', np.float64, ('lat','lon'))
    gridareas[:,:] = grid_area_2x25
    gridareas.units = 'm2'
    latss = dataset.createVariable('latitude', np.float64, ('lat',))
    latss[:] = lat
    lonss = dataset.createVariable('longitude', np.float64, ('lon',))
    lonss[:] = lon
    FF_priors = dataset.createVariable('FF_prior', np.float64, ('time','lat','lon'))
    FF_priors[:,:,:] = FF_Flux_prior
    FF_priors.units = 'gC/m2/day'
    FF_posts = dataset.createVariable('FF_post', np.float64, ('time','lat','lon'))
    FF_posts[:,:,:] = FF_Flux_post
    FF_posts.units = 'gC/m2/day'
    Biogenic_priors = dataset.createVariable('Biogenic_prior', np.float64, ('time','lat','lon'))
    Biogenic_priors[:,:,:] = Biogenic_Flux_prior
    Biogenic_priors.units = 'gC/m2/day'
    Biogenic_posts = dataset.createVariable('Biogenic_post', np.float64, ('time','lat','lon'))
    Biogenic_posts[:,:,:] = Biogenic_Flux_post
    Biogenic_posts.units = 'gC/m2/day'
    dataset.close()

if __name__ == "__main__":
    
    # -- Parameters --
    iteration = '21'
    # ----------------


    for rep in [0,1]:
        for year in range(2019,2024):
            for prior_model in ['GFED','QFED','GFAS']:
                
                
                # Read in the scale factors
                if rep==1:
                    ncfile_SF = '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-2023/Run_COinv_rep_'+prior_model+'_'+str(year).zfill(4)+'/GDT-EMS/EMS-sf-'+iteration+'.nc'
                else:
                    ncfile_SF = '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-2023/Run_COinv_'+prior_model+'_'+str(year).zfill(4)+'/GDT-EMS/EMS-sf-'+iteration+'.nc'
                print(ncfile_SF)
                f=Dataset(ncfile_SF,mode='r')
                lon=f.variables['lon'][:]
                lat=f.variables['lat'][:]
                SF=f.variables['EMS-01'][:]
                f.close()
                
                # Directories of prior fluxes
                ncdir_FF_fire = '/nobackup/bbyrne1/Flux_2x25_CO/FossilFuel/CEDSdaily/'+str(year).zfill(4)+'/'
                ncdir_Biogenic_fire = '/nobackup/bbyrne1/Flux_2x25_CO/Biogenic/Biogenic_units/'+str(year).zfill(4)+'/'
            
                # Read & Caclulate prior and posterior fluxes
                FF_Flux_prior, FF_Flux_post, Biogenic_Flux_prior, Biogenic_Flux_post = calc_fluxes(SF,ncdir_FF_fire,ncdir_Biogenic_fire,prior_model)
            
                # Write out data
                dir_out = '/u/bbyrne1/python_codes/Canada_Fires_2023/Byrne_etal_codes/plot_figures/data_for_figures/'
                if rep==1:
                    ncfile_out = dir_out+'TROPOMI_rep_'+prior_model+'_COinv_2x25_'+str(year).zfill(4)+'_otherFlux_3day.nc'
                else:
                    ncfile_out = dir_out+'TROPOMI_'+prior_model+'_COinv_2x25_'+str(year).zfill(4)+'_otherFlux_3day.nc'
                
                write_dataset(ncfile_out, FF_Flux_prior, FF_Flux_post, Biogenic_Flux_prior, Biogenic_Flux_post)
