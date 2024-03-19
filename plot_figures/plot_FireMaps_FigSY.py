import csv
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import datetime
from netCDF4 import Dataset
import glob, os
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Polygon
from math import pi, cos, radians
import numpy.matlib
from pylab import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from mpl_toolkits.basemap import Basemap, cm, maskoceans

# *******************************************************
# -------- plot_climate_anomalies.py
#
# This code processes data and plots Figures 2, S3 and S4
#
# contact: Brendan Byrne
# email: brendan.k.byrne@jpl.nasa.gov
#
# *******************************************************

days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
days_in_month_cum = np.zeros(13)
for i in range(13):
    days_in_month_cum[i] = np.sum(days_in_month[0:i])

# ===== Read data and calculate mean over Canadian forest =====                                                   
#                                                         
nc_out = './data_for_figures/Canada_forest_mask_2x25.nc'
f = Dataset(nc_out,'r')
lon_MERRA2_2x25 = f.variables['lon'][:]
lat_MERRA2_2x25 = f.variables['lat'][:]
Forest_mask_2x25 = f.variables['mask'][:]
f.close()
#                                                         
# ----------------------------------------------------------------------------------------



# =======================================================
def map_COandCO2_forest(file_name,Forest_mask):
    #nc_file = './data_for_figures/TROPOMI_GFED_COinv_2x25_2023_fire_3day.nc'
    #
    print(file_name)
    #                                                                                                    
    f = Dataset(file_name,'r')
    CO_prior_per_area = f.variables['CO_prior'][:] # gC/m2                                                 
    CO2_prior_per_area = f.variables['CO2_prior'][:] # gC/m2                                               
    CO_post_per_area = f.variables['CO_post'][:] # gC/m2                                                 
    CO2_post_per_area = f.variables['CO2_post'][:] # gC/m2                                               
    f.close()
    #
    COandCO2_prior = np.sum(CO_prior_per_area[121-1+1:274-1,:,:] + CO2_prior_per_area[121-1+1:274-1,:,:],0)
    COandCO2_prior[np.where(Forest_mask==0)] = np.nan
    #
    COandCO2_post = np.sum(CO_post_per_area[121-1+1:274-1,:,:] + CO2_post_per_area[121-1+1:274-1,:,:],0)
    COandCO2_post[np.where(Forest_mask==0)] = np.nan
    #
    return COandCO2_prior, COandCO2_post
    # =======================================================


# ===========================================================
# 3 day optimization w/out representativeness errors
COandCO2_prior_GFED_3day, COandCO2_post_GFED_3day = map_COandCO2_forest('./data_for_figures/TROPOMI_GFED_COinv_2x25_2023_fire_3day.nc',Forest_mask_2x25)
COandCO2_prior_GFAS_3day, COandCO2_post_GFAS_3day = map_COandCO2_forest('./data_for_figures/TROPOMI_GFAS_COinv_2x25_2023_fire_3day.nc',Forest_mask_2x25)
COandCO2_prior_QFED_3day, COandCO2_post_QFED_3day = map_COandCO2_forest('./data_for_figures/TROPOMI_QFED_COinv_2x25_2023_fire_3day.nc',Forest_mask_2x25)
# 3 day optimization w/out representativeness errors
COandCO2_prior_GFED_3day_rep, COandCO2_post_GFED_3day_rep = map_COandCO2_forest('./data_for_figures/TROPOMI_rep_GFED_COinv_2x25_2023_fire_3day.nc',Forest_mask_2x25)
COandCO2_prior_GFAS_3day_rep, COandCO2_post_GFAS_3day_rep = map_COandCO2_forest('./data_for_figures/TROPOMI_rep_GFAS_COinv_2x25_2023_fire_3day.nc',Forest_mask_2x25)
COandCO2_prior_QFED_3day_rep, COandCO2_post_QFED_3day_rep = map_COandCO2_forest('./data_for_figures/TROPOMI_rep_QFED_COinv_2x25_2023_fire_3day.nc',Forest_mask_2x25)
# 7 day optimization w/out representativeness errors
COandCO2_prior_GFED_7day, COandCO2_post_GFED_7day = map_COandCO2_forest('./data_for_figures/TROPOMI_GFED_COinv_2x25_2023_fire_7day.nc',Forest_mask_2x25)
COandCO2_prior_GFAS_7day, COandCO2_post_GFAS_7day = map_COandCO2_forest('./data_for_figures/TROPOMI_GFAS_COinv_2x25_2023_fire_7day.nc',Forest_mask_2x25)
COandCO2_prior_QFED_7day, COandCO2_post_QFED_7day = map_COandCO2_forest('./data_for_figures/TROPOMI_QFED_COinv_2x25_2023_fire_7day.nc',Forest_mask_2x25)
# 7 day optimization w/out representativeness errors
COandCO2_prior_GFED_7day_rep, COandCO2_post_GFED_7day_rep = map_COandCO2_forest('./data_for_figures/TROPOMI_rep_GFED_COinv_2x25_2023_fire_7day.nc',Forest_mask_2x25)
COandCO2_prior_GFAS_7day_rep, COandCO2_post_GFAS_7day_rep = map_COandCO2_forest('./data_for_figures/TROPOMI_rep_GFAS_COinv_2x25_2023_fire_7day.nc',Forest_mask_2x25)
COandCO2_prior_QFED_7day_rep, COandCO2_post_QFED_7day_rep = map_COandCO2_forest('./data_for_figures/TROPOMI_rep_QFED_COinv_2x25_2023_fire_7day.nc',Forest_mask_2x25)
# ====
COandCO2_prior_GFED = (COandCO2_prior_GFED_3day+
                       COandCO2_prior_GFED_3day_rep+
                       COandCO2_prior_GFED_7day+
                       COandCO2_prior_GFED_7day_rep)/4.
COandCO2_prior_GFAS = (COandCO2_prior_GFAS_3day+
                       COandCO2_prior_GFAS_3day_rep+
                       COandCO2_prior_GFAS_7day+
                       COandCO2_prior_GFAS_7day_rep)/4.
COandCO2_prior_QFED = (COandCO2_prior_QFED_3day+
                       COandCO2_prior_QFED_3day_rep+
                       COandCO2_prior_QFED_7day+
                       COandCO2_prior_QFED_7day_rep)/4.
# ====
COandCO2_post_GFED = (COandCO2_post_GFED_3day+
                       COandCO2_post_GFED_3day_rep+
                       COandCO2_post_GFED_7day+
                       COandCO2_post_GFED_7day_rep)/4.
COandCO2_post_GFAS = (COandCO2_post_GFAS_3day+
                       COandCO2_post_GFAS_3day_rep+
                       COandCO2_post_GFAS_7day+
                       COandCO2_post_GFAS_7day_rep)/4.
COandCO2_post_QFED = (COandCO2_post_QFED_3day+
                       COandCO2_post_QFED_3day_rep+
                       COandCO2_post_QFED_7day+
                       COandCO2_post_QFED_7day_rep)/4.
# ===========================================================


# ===========================================================================================
fig = plt.figure(98, figsize=(12*0.7,5*0.7), dpi=300)
#
m = Basemap(width=5040000,height=3600000,resolution='l',projection='laea',lat_ts=58,lat_0=58.,lon_0=(-150-40)/2.)
X,Y = np.meshgrid(lon_MERRA2_2x25[1:143],lat_MERRA2_2x25[50:91])
xx,yy=m(X,Y)
#                                                                                
ax1 = fig.add_axes([0.022+0./3.,0.01+1./2.,0.95/2.,0.94/2.])
m.drawlsmask(land_color='white',ocean_color='gainsboro',lakes=False)
m.drawcoastlines(linewidth=0.5,color='grey')
tt = m.pcolormesh(xx,yy,np.log10(COandCO2_prior_GFED[50:91,1:143]),cmap='inferno_r',vmin=0,vmax=4)
m.drawstates(linewidth=0.5,color='grey')
m.drawcountries(linewidth=0.5,color='grey')
plt.annotate('(a)', xy=(4./364, 0.98), xycoords='axes fraction',va='top',ha='left')
#                                                                                
ax1 = fig.add_axes([-0.014+1./3.,0.01+1./2.,0.95/2.,0.94/2.])
m.drawlsmask(land_color='white',ocean_color='gainsboro',lakes=False)
m.drawcoastlines(linewidth=0.5,color='grey')
tt = m.pcolormesh(xx,yy,np.log10(COandCO2_prior_GFAS[50:91,1:143]),cmap='inferno_r',vmin=0,vmax=4)
m.drawstates(linewidth=0.5,color='grey')
m.drawcountries(linewidth=0.5,color='grey')
plt.annotate('(b)', xy=(4./364, 0.98), xycoords='axes fraction',va='top',ha='left')
#                                                                                
ax1 = fig.add_axes([-0.05+2./3.,0.01+1./2.,0.95/2.,0.94/2.])
m.drawlsmask(land_color='white',ocean_color='gainsboro',lakes=False)
m.drawcoastlines(linewidth=0.5,color='grey')
tt = m.pcolormesh(xx,yy,np.log10(COandCO2_prior_QFED[50:91,1:143]),cmap='inferno_r',vmin=0,vmax=4)
m.drawstates(linewidth=0.5,color='grey')
m.drawcountries(linewidth=0.5,color='grey')
plt.annotate('(c)', xy=(4./364, 0.98), xycoords='axes fraction',va='top',ha='left')
#                                                                                               
ax1 = fig.add_axes([0.085, 0.01+1./2., 0.015, 0.94/2.])
cbar = plt.colorbar(tt,cax=ax1,extend='both',location='left',label='2023 Fires ($\mathrm{gC \, m^{-2}}$)',ticks=[0,1,2,3,4])
cbar.ax.set_yticklabels(['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])
#
####################
ax1 = fig.add_axes([0.022+0./3.,0.01+0./2.,0.95/2.,0.94/2.])
m.drawlsmask(land_color='white',ocean_color='gainsboro',lakes=False)
m.drawcoastlines(linewidth=0.5,color='grey')
tt = m.pcolormesh(xx,yy,np.log10(COandCO2_post_GFED[50:91,1:143]),cmap='inferno_r',vmin=0,vmax=4)
m.drawstates(linewidth=0.5,color='grey')
m.drawcountries(linewidth=0.5,color='grey')
plt.annotate('(d)', xy=(4./364, 0.98), xycoords='axes fraction',va='top',ha='left')
#                                                                                
ax1 = fig.add_axes([-0.014+1./3.,0.01+0./2.,0.95/2.,0.94/2.])
m.drawlsmask(land_color='white',ocean_color='gainsboro',lakes=False)
m.drawcoastlines(linewidth=0.5,color='grey')
tt = m.pcolormesh(xx,yy,np.log10(COandCO2_post_GFAS[50:91,1:143]),cmap='inferno_r',vmin=0,vmax=4)
m.drawstates(linewidth=0.5,color='grey')
m.drawcountries(linewidth=0.5,color='grey')
plt.annotate('(e)', xy=(4./364, 0.98), xycoords='axes fraction',va='top',ha='left')
#                                                                                
ax1 = fig.add_axes([-0.05+2./3.,0.01+0./2.,0.95/2.,0.94/2.])
m.drawlsmask(land_color='white',ocean_color='gainsboro',lakes=False)
m.drawcoastlines(linewidth=0.5,color='grey')
tt = m.pcolormesh(xx,yy,np.log10(COandCO2_post_QFED[50:91,1:143]),cmap='inferno_r',vmin=0,vmax=4)
m.drawstates(linewidth=0.5,color='grey')
m.drawcountries(linewidth=0.5,color='grey')
plt.annotate('(f)', xy=(4./364, 0.98), xycoords='axes fraction',va='top',ha='left')
#                                                                                               
ax1 = fig.add_axes([0.085, 0.01, 0.015, 0.94/2.])
cbar = plt.colorbar(tt,cax=ax1,extend='both',location='left',label='2023 Fires ($\mathrm{gC \, m^{-2}}$)',ticks=[0,1,2,3,4])
cbar.ax.set_yticklabels(['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])
#
plt.savefig('Figures/Byrne_etal_FigSY_revision.png', dpi=300)


