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

# ===========================================
#
# This program writes TROPOMI obs with uncertainties that include
# representativeness errors. The observational error is calculated
# to be:
#
# unc^2 = retrieval_error^2 + Heald_error^2
#
# This program depends on the outputs of:
#   - write_daily_YHx_prior.py
#   - calculate_representativeness_error_Heald.py
#
# ===========================================

def write_TROPOMI_wRep_error(year,month,day):
    
    print('--------------------------------------------')
    
    # Read TROPOMI obs ----
    nc_file_TROPOMI = '/nobackup/bbyrne1/TROPOMI_XCO_2x25/'+str(year).zfill(4)+'/'+str(int(month)).zfill(2)+'/'+str(int(day)).zfill(2)+'.nc'
    # ---
    if os.path.isfile(nc_file_TROPOMI):
        # ----
        f = Dataset(nc_file_TROPOMI,'r')
        longitude=f.variables['longitude'][:]
        latitude=f.variables['latitude'][:]
        mode=f.variables['mode'][:]
        time=f.variables['time'][:]
        pressure=f.variables['pressure'][:]
        xCO=f.variables['xCO'][:]
        xCOapriori=f.variables['xCO-apriori'][:]
        xCOpressureWeight=f.variables['xCO-pressureWeight'][:]
        xCOuncertainty=f.variables['xCO-uncertainty'][:]
        xCOaveragingKernel=f.variables['xCO-averagingKernel'][:]
        COapriori=f.variables['CO-apriori'][:]
        f.close()
        # ---------------------
        
        # Calc total unc for each TROPOMI retrieval -----
        total_unc = np.zeros(np.size(xCO))    
        for obs_i in range(np.size(xCOuncertainty)):
            I = np.argmin(np.abs(longitude[obs_i] - lon_grid))
            J = np.argmin(np.abs(latitude[obs_i] - lat_grid))
            total_unc[obs_i] = unc_Heald[ii,J,I]
        # -----------------------------------------------
        
        
        print('old unc --> 5per: %.4f; 25per: %.4f; 50per: %.4f; 75per: %.4f; 95per: %.4f' %
              (np.percentile(xCOuncertainty*1e9,5), np.percentile(xCOuncertainty*1e9,25), np.percentile(xCOuncertainty*1e9,50), np.percentile(xCOuncertainty*1e9,75), np.percentile(xCOuncertainty*1e9,95) ) )

        print('new unc --> 5per: %.4f; 25per: %.4f; 50per: %.4f; 75per: %.4f; 95per: %.4f' %
              (np.percentile(total_unc*1e9,5), np.percentile(total_unc*1e9,25), np.percentile(total_unc*1e9,50), np.percentile(total_unc*1e9,75), np.percentile(total_unc*1e9,95) ) )


        # Write out TROPOMI obs with total uncertainty
        nc_file_TROPOMIrep = '/nobackup/bbyrne1/TROPOMIrep_XCO_2x25/'+str(year).zfill(4)+'/'+str(int(month)).zfill(2)+'/'+str(int(day)).zfill(2)+'.nc'
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
        xCOs[:]=xCO
        xCOaprioris = dataset.createVariable('xCO-apriori', np.float64, ('nSamples',))
        xCOaprioris[:]=xCOapriori
        xCOpressureWeights = dataset.createVariable('xCO-pressureWeight', np.float64, ('nSamples','maxLevels'))
        xCOpressureWeights[:,:]=xCOpressureWeight
        xCOuncertaintys = dataset.createVariable('xCO-uncertainty', np.float64, ('nSamples',))
        xCOuncertaintys[:]=total_unc
        xCOaveragingKernels = dataset.createVariable('xCO-averagingKernel', np.float64, ('nSamples','maxLevels'))
        xCOaveragingKernels[:,:]=xCOaveragingKernel
        COaprioris = dataset.createVariable('CO-apriori', np.float64, ('nSamples','maxLevels'))
        COaprioris[:,:]=COapriori
        dataset.close()



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

        
# ---------------------------------
# load Heald uncertainties, which have dimension [day of year, lat, lon]
file_in = 'Daily_unc_Heald.nc'
print(file_in)
f = Dataset(file_in,'r')
lon_grid=f.variables['longitude'][:]
lat_grid=f.variables['latitude'][:]
unc_Heald=f.variables['unc_Heald'][:] * 1e-9
f.close()
# ---------------------------------

# Loop over years
for year in range(2022,2024):
    # Loop over days of the year
    for ii in range(days_in_year-1):
        # Write the TROPOMI data with representativeness errors
        write_TROPOMI_wRep_error(year,month_arr[ii],day_arr[ii])
    
