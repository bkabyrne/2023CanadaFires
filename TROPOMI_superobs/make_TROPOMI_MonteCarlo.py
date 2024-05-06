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
----- make_TROPOMI_MonteCarlo.py

This program write TROPOMI L2# super-obs for the Monte
Carlo experiments.

'''

def write_TROPOMI_MC_peturb(inventory,year,month,day,iterN):
    
    '''
    Adds random perturbation to observations for Monte Carlo inversions

    inputs
     - inventory: prior inventory for the Monte Carlo experiment
     - year: year of observations
     - month: month of observations
     - day: day of observations
     - iterN: Monte Carlo ensemble member
    '''
    
    # Read TROPOMI obs ----
    nc_file_TROPOMI = '/nobackup/bbyrne1/TROPOMIrep_XCO_2x25/'+str(year).zfill(4)+'/'+str(int(month)).zfill(2)+'/'+str(int(day)).zfill(2)+'.nc'    # ---
    if os.path.isfile(nc_file_TROPOMI):
        
        # Read data
        f = Dataset(nc_file_TROPOMI,'r')
        longitude=f.variables['longitude'][:]
        latitude=f.variables['latitude'][:]
        mode=f.variables['mode'][:]
        time=f.variables['time'][:]
        pressure=f.variables['pressure'][:]
        xCO_act=f.variables['xCO'][:]
        xCOapriori=f.variables['xCO-apriori'][:]
        xCOpressureWeight=f.variables['xCO-pressureWeight'][:]
        xCOuncertainty=f.variables['xCO-uncertainty'][:]
        xCOaveragingKernel=f.variables['xCO-averagingKernel'][:]
        COapriori=f.variables['CO-apriori'][:]
        f.close()
        # ---------------------
        nc_fwd = '/nobackup/bbyrne1/GHGF-CMS-7day-COinv-2023/FWD_COinv_rep_'+inventory+'_2023/OBSF/TROPOMIrep_XCO_2x25/'+str(year).zfill(4)+'/'+str(int(month)).zfill(2)+'/'+str(int(day)).zfill(2)+'.nc'  
        f = Dataset(nc_fwd,'r')
        xCO=np.squeeze(f.variables['HX'][:])
        f.close()
        # ---------------------

        # Perform perturbation to XCO obs for Monte Carlo experiment
        unc_perturb = np.random.normal(loc=0, scale=xCOuncertainty)
        xCO_perturb = xCO + unc_perturb

        # Write out TROPOMI obs with total uncertainty
        nc_file_TROPOMIrep = '/nobackup/bbyrne1/TROPOMIrep_XCO_2x25_'+inventory+'_'+str(iterN).zfill(2)+'/'+str(year).zfill(4)+'/'+str(int(month)).zfill(2)+'/'+str(int(day)).zfill(2)+'.nc'
        print(nc_file_TROPOMIrep)
        dataset = Dataset(nc_file_TROPOMIrep,'w')
        nSamples = dataset.createDimension('nSamples',np.size(time))
        maxLevels = dataset.createDimension('maxLevels',np.size(pressure[0,:]))
        longitudes = dataset.createVariable('longitude', np.float64, ('nSamples',))
        longitudes[:]=longitude
        latitudes = dataset.createVariable('latitude', np.float64, ('nSamples',))
        latitudes[:]=latitude
        modes = dataset.createVariable('mode', np.float64, ('nSamples',))
        modes[:]=mode
        times = dataset.createVariable('time', np.float64, ('nSamples',))
        times[:]=time
        pressures = dataset.createVariable('pressure', np.float64, ('nSamples','maxLevels'))
        pressures[:,:]=pressure
        xCOs = dataset.createVariable('xCO', np.float64, ('nSamples',))
        xCOs[:]=xCO_perturb
        xCOaprioris = dataset.createVariable('xCO-apriori', np.float64, ('nSamples',))
        xCOaprioris[:]=xCOapriori
        xCOpressureWeights = dataset.createVariable('xCO-pressureWeight', np.float64, ('nSamples','maxLevels'))
        xCOpressureWeights[:,:]=xCOpressureWeight
        xCOuncertaintys = dataset.createVariable('xCO-uncertainty', np.float64, ('nSamples',))
        xCOuncertaintys[:]=xCOuncertainty
        xCOaveragingKernels = dataset.createVariable('xCO-averagingKernel', np.float64, ('nSamples','maxLevels'))
        xCOaveragingKernels[:,:]=xCOaveragingKernel
        COaprioris = dataset.createVariable('CO-apriori', np.float64, ('nSamples','maxLevels'))
        COaprioris[:,:]=COapriori
        dataset.close()



if __name__ == "__main__":

    # ---------------------------------
    # Create arrays of day, month for 365 days
    # ignore leap days
    days_in_month = np.array([30,31,30,31,31,30])
    days_in_year = np.sum(days_in_month)
    #
    # ----
    n=0
    month_arr = np.zeros(days_in_year)
    day_arr = np.zeros(days_in_year)
    for i in range(np.size(days_in_month)):
        for j in range(days_in_month[i]):
            month_arr[n] = int(i + 4)
            day_arr[n] = int(j + 1)
            n=n+1
    # ---------------------------------


    # Loop over years
    #for year in range(2022,2024):
    # Loop over days of the year
    year = 2023
    for inventory in ['GFED','GFAS','QFED']:
        for iteration in range(1,41):
            for ii in range(days_in_year-1):
                # Write the TROPOMI data with representativeness errors
                write_TROPOMI_MC_peturb(inventory,year,month_arr[ii],day_arr[ii],iteration)
                print('--------------------------------------------')
