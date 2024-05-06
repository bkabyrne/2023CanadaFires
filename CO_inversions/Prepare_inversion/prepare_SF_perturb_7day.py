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

'''
  This program creates random perturbations to the prior Scale Factor
  based on the prior uncertainties. These are used for the Monte Carlo
  experiments. This version is for 7-day optimization
'''



def write_fluxes(inventory,yyyy,unc_perturb,iterN):
    #
    # =========================================
    # Write out SF uncertainty pertubation
    # =========================================
    #
    nc_out = 'SF_perturb/SF_perturb_'+inventory+'_'+str(yyyy).zfill(4)+'_'+str(iterN).zfill(2)+'_7day.nc'
    #
    dataset = Dataset(nc_out,'w')
    #
    lats = dataset.createDimension('MMSCL',26)
    lats = dataset.createDimension('lat',91)
    lons = dataset.createDimension('lon',144)
    #
    CO2_UNC1 = dataset.createVariable('SF_perturb',np.float64,('MMSCL','lat','lon'))
    CO2_UNC1[:,:] = unc_perturb
    #
    dataset.close()
    #


# ========================================================================================

inventory = 'QFED'
yyyy = 2023

# We will use 3-Day grouping for optimizations                    
unc_grouped = np.ones((26,91,144)) * 2.

for iteration in range(1,41):
    #
    unc_perturb = np.random.normal(loc=0, scale=unc_grouped)
    #
    write_fluxes(inventory,yyyy,unc_perturb,iteration)

