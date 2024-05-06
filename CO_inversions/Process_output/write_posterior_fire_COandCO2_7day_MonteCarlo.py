#from mpl_toolkits.basemap import Basemap, cm
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
------ write_posterior_fire_COandCO2_7day_MonteCarlo.py          
                                                                 
Writes posterior fire emissions for Monte Carlo ensemble (7-day)
'''

def find_file_with_largest_number(directory):
    #                                                                                         
    # --------------------------                                                              
    #                                                                                         
    # This function finds the file in the given directory                                     
    # that contains the largest number and 'SF' in its name                                   
    #                                                                                         
    # inputs:                                                                                 
    #   - directory: directory to conduct search                                              
    #                                                                                         
    # outputs:                                                                                
    #   - file with largest number in name                                                    
    #                                                                                         
    # --------------------------                                                              
    #                                                                                         
    files1 = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files = [item for item in files1 if "sf" in item]
    max_number = float('-inf')
    max_file = None
    #                                                                                         
    for file in files:
        file_name, file_extension = os.path.splitext(file)
        try:
            file_number = int(''.join(filter(str.isdigit, file_name)))
            if file_number > max_number:
                max_number = file_number
                max_file = file
        except ValueError:
            pass
    #                                                                                         
    return os.path.join(directory, max_file) if max_file else None


def calc_fluxes(SF,nc_CO_fire,nc_CO2_fire,prior_model):
    #
    # ================================
    #
    # This function reads in posterior scale factors and daily prior fluxes
    # during Apr-Sep then calculates timeseries of prior and posterior fluxes
    #
    # inputs:
    #  - SF: array of posterior scale factors
    #  - nc_CO_fire: path to prior CO fire flux directory
    #  - nc_CO2_fire: path to prior CO2 fire flux directory
    #
    # outputs:
    #  - CO_Prior_flux: timeseries of prior CO fire fluxes (time,lat,lon) 
    #  - CO_Posterior_flux: timeseries of posterior CO fire fluxes (time,lat,lon) 
    #  - CO2_Prior_flux: timeseries of prior CO2 fire fluxes (time,lat,lon) 
    #  - CO2_Posterior_flux: timeseries of posterior CO2 fire fluxes (time,lat,lon) 
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
    CO_Prior_flux = np.zeros((365,np.size(lat),np.size(lon)))
    CO_Posterior_flux = np.zeros((365,np.size(lat),np.size(lon)))
    CO2_Prior_flux = np.zeros((365,np.size(lat),np.size(lon)))
    CO2_Posterior_flux = np.zeros((365,np.size(lat),np.size(lon)))
    for nnt in range(183-30-1):
        nn = nnt+30
        #
        SF_index = int(np.floor(nn/7.))
        month = np.argmax( nn < days_in_month_cum)+3
        day = int(nn-days_in_month_cum[month-1-3])
        #
        #print(SF_index)
        #print(str(month).zfill(2)+'/'+str(day+1).zfill(2))
        file_in = nc_CO_fire+str(month).zfill(2)+'/'+str(day+1).zfill(2)+'.nc'
        f=Dataset(file_in,mode='r')
        if prior_model == 'GFED':
            CO_Prior_flux[nn+31+28+31,:,:] = np.mean(f.variables['CO_Flux'][:],0)  * (60.*60.*24.)/1000. # kgC/km2/s -> gC/m2/d
        else:
            CO_Prior_flux[nn+31+28+31,:,:] = f.variables['CO_Flux'][:]  * (60.*60.*24.)/1000. # kgC/km2/s -> gC/m2/d
        CO_Posterior_flux[nn+31+28+31,:,:] = CO_Prior_flux[nn+31+28+31,:,:] * SF[SF_index,:,:]
        #
        file_in = nc_CO2_fire+str(month).zfill(2)+'/'+str(day+1).zfill(2)+'.nc'
        f=Dataset(file_in,mode='r')
        if prior_model == 'GFED':
            CO2_Prior_flux[nn+31+28+31,:,:] = np.mean(f.variables['CO2_Flux'][:],0)  * (60.*60.*24.)/1000. # kgC/km2/s -> gC/m2/d
        else:
            CO2_Prior_flux[nn+31+28+31,:,:] = f.variables['CO2_Flux'][:]  * (60.*60.*24.)/1000. # kgC/km2/s -> gC/m2/d 
        CO2_Posterior_flux[nn+31+28+31,:,:] = CO2_Prior_flux[nn+31+28+31,:,:] * SF[SF_index,:,:]
    #
    return CO_Prior_flux, CO_Posterior_flux, CO2_Prior_flux, CO2_Posterior_flux


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


def write_dataset(nc_out, CO_Flux_prior, CO_Flux_post, CO2_Flux_prior, CO2_Flux_post):
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
    ens_members = dataset.createDimension('ens_member',40)
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
    CO_priors = dataset.createVariable('CO_prior', np.float64, ('ens_member','time','lat','lon'))
    CO_priors[:,:,:,:] = CO_Flux_prior
    CO_priors.units = 'gC/m2/day'
    CO_posts = dataset.createVariable('CO_post', np.float64, ('ens_member','time','lat','lon'))
    CO_posts[:,:,:,:] = CO_Flux_post
    CO_posts.units = 'gC/m2/day'
    CO2_priors = dataset.createVariable('CO2_prior', np.float64, ('ens_member','time','lat','lon'))
    CO2_priors[:,:,:,:] = CO2_Flux_prior
    CO2_priors.units = 'gC/m2/day'
    CO2_posts = dataset.createVariable('CO2_post', np.float64, ('ens_member','time','lat','lon'))
    CO2_posts[:,:,:,:] = CO2_Flux_post
    CO2_posts.units = 'gC/m2/day'
    dataset.close()

if __name__ == "__main__":
    
    CO_Flux_prior = np.zeros((40,365,91,144))
    CO_Flux_post = np.zeros((40,365,91,144))
    CO2_Flux_prior = np.zeros((40,365,91,144))
    CO2_Flux_post = np.zeros((40,365,91,144))

    for prior_model in ['GFED','QFED','GFAS']:

        
        for ens_member in range(1,41):
                        
            # Read in the scale factors
            if prior_model == 'GFED':
                ncfile_SF = find_file_with_largest_number('/nobackup/bbyrne1/GHGF-CMS-7day-COinv-MonteCarlo/Run_COinv_'+str(ens_member).zfill(2)+'/GDT-EMS/')
            else:
                ncfile_SF = find_file_with_largest_number('/nobackup/bbyrne1/GHGF-CMS-7day-COinv-MonteCarlo/Run_COinv_'+prior_model+'_'+str(ens_member).zfill(2)+'/GDT-EMS/')
            print(ncfile_SF)
            f=Dataset(ncfile_SF,mode='r')
            lon=f.variables['lon'][:]
            lat=f.variables['lat'][:]
            SF=f.variables['EMS-01'][:]
            f.close()

            # Directories of prior fluxes
            if prior_model == 'GFED':
                ncdir_CO_fire = '/nobackup/bbyrne1/Flux_2x25_CO/BiomassBurn/GFED41s/2023/'
                ncdir_CO2_fire ='/nobackup/bbyrne1/GFED41s_2x25/2023/'
            else:
                ncdir_CO_fire = '/nobackup/bbyrne1/Flux_2x25_CO/BiomassBurn/'+prior_model+'/2023/'
                ncdir_CO2_fire = '/nobackup/bbyrne1/Flux_2x25_CO/BiomassBurn/'+prior_model+'_CO2/2023/'
            
            # Read & Caclulate prior and posterior fluxes
            CO_Flux_prior[ens_member-1,:,:,:], CO_Flux_post[ens_member-1,:,:,:], CO2_Flux_prior[ens_member-1,:,:,:], CO2_Flux_post[ens_member-1,:,:,:] = calc_fluxes(SF,ncdir_CO_fire,ncdir_CO2_fire,prior_model)
            
        # Write out data
        dir_out = '/u/bbyrne1/python_codes/Canada_Fires_2023/Byrne_etal_codes/plot_figures/data_for_figures/'
        ncfile_out = dir_out+'TROPOMI_'+prior_model+'_COinv_2x25_2023_fire_7day_MonteCarlo.nc' 
        write_dataset(ncfile_out, CO_Flux_prior, CO_Flux_post, CO2_Flux_prior, CO2_Flux_post)
