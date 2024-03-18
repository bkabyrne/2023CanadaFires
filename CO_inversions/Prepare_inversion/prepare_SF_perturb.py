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

# #########################################################################
#
#  This program creates random perturbations to the prior Scale Factor
#  based on the prior uncertainties. These are used for the Monte Carlo
# experiments.
#
# #########################################################################


def read_fluxes(inventory,year,month,day):
    #
    # =========================================
    # Read SF uncertainty
    # =========================================
    #
    nc_in = '/nobackup/bbyrne1/Flux_2x25_CO/Combined/FF_'+inventory+'_Bio_UNCr_3Day/'+str(year).zfill(4)+'/'+str(month).zfill(2)+'/'+str(day).zfill(2)+'.nc'
    #
    f=Dataset(nc_in,mode='r')
    Uncertainty = f.variables['Uncertainty'][:]
    f.close()
    #
    return Uncertainty


def write_fluxes(inventory,yyyy,unc_perturb,iterN):
    #
    # =========================================
    # Write out SF uncertainty pertubation
    # =========================================
    #
    nc_out = 'SF_perturb/SF_perturb_'+inventory+'_'+str(yyyy).zfill(4)+'_'+str(iterN).zfill(2)+'.nc'
    #
    dataset = Dataset(nc_out,'w')
    #
    lats = dataset.createDimension('MMSCL',61)
    lats = dataset.createDimension('lat',91)
    lons = dataset.createDimension('lon',144)
    #
    CO2_UNC1 = dataset.createVariable('SF_perturb',np.float64,('MMSCL','lat','lon'))
    CO2_UNC1[:,:] = unc_perturb
    #
    dataset.close()
    #
    return Uncertainty


# ========================================================================================

inventory = 'QFED'
yyyy = 2023
days_in_months = np.array([31,28,31,30,31,30,31,31,30,31,30,31])

# Read and combine the Fire, FF and biogenic fluxes
Uncertainty = np.zeros((np.sum(days_in_months[3:9]), 91, 144))
n1=0
for mm in range(3,9):
    for dd in range(days_in_months[mm]):
        #
        Uncertainty[n1,:,:] = read_fluxes(inventory,yyyy,mm+1,dd+1)
        #
        n1=n1+1


# We will use 3-Day grouping for optimizations                    
unc_grouped = np.zeros((61,91,144))
for ind in range(61): # each ind is a temporal grouping of 3 day
    unc_grouped[ind,:,:] = np.mean(Uncertainty[(ind)*3:(ind+1)*3,:,:],0)


for iteration in range(1,41):
    #
    unc_perturb = np.random.normal(loc=0, scale=unc_grouped)
    #
    write_fluxes(inventory,yyyy,unc_perturb,iteration)

