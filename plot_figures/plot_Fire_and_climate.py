from mpl_toolkits.basemap import Basemap, cm, maskoceans
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
-------- plot_Fire_and_climate.py                            
                                                           
This code processes data and plots Figures 3 and S6        
                                                          
contact: Brendan Byrne                                    
email: brendan.k.byrne@jpl.nasa.gov                       

'''


nc_out = './data_for_figures/GFED_T2MZscore_precipZscore.nc'
f = Dataset(nc_out,'r')
T2M_zvalR = f.variables['T2M_vec'][:]
precip_zvalR = f.variables['precip_vec'][:]
GFED_tot_perFireR = f.variables['GFED_arr'][:]
T2M_zscore_mean = f.variables['T2M_mean_zscore'][:]
precip_zscore_mean = f.variables['precip_mean_zscore'][:]
f.close()
#
nc_out = './data_for_figures/North_America_pr.nc'
f = Dataset(nc_out,'r')
lon = f.variables['longitude'][:]
lat = f.variables['latitude'][:]
pr_hist = f.variables['pr_hist'][:] # kg/m2/s = mm/s
pr_ssp245 = f.variables['pr_ssp245'][:]
pr_ssp585 = f.variables['pr_ssp585'][:]
f.close()
# Append historical simulation with projection
pr_hist_ssp245 = np.append(pr_hist[1800:1980,:,:],pr_ssp245,axis=0)
pr_hist_ssp585 = np.append(pr_hist[1800:1980,:,:],pr_ssp585,axis=0)

nc_out = './data_for_figures/North_America_tas.nc'
f = Dataset(nc_out,'r')
tas_hist = f.variables['tas_hist'][:]
tas_ssp245 = f.variables['tas_ssp245'][:]
tas_ssp585 = f.variables['tas_ssp585'][:]
f.close()
# Append historical simulation with projection
tas_hist_ssp245 = np.append(tas_hist[1800:1980,:,:],tas_ssp245,axis=0)
tas_hist_ssp585 = np.append(tas_hist[1800:1980,:,:],tas_ssp585,axis=0)

# Calculate cumulative precip and May-Sep Temperature
pr_hist_ssp245_yr = np.zeros((101,60,110))
pr_hist_ssp585_yr = np.zeros((101,60,110))
tas_hist_ssp245_yr = np.zeros((101,60,110))
tas_hist_ssp585_yr = np.zeros((101,60,110))
for i in range(101):
    pr_hist_ssp245_yr[i,:,:] = np.mean(pr_hist_ssp245[int(i*12.+0):int(i*12.+10),:,:],0) # Jan-Sep precip
    pr_hist_ssp585_yr[i,:,:] = np.mean(pr_hist_ssp585[int(i*12.+0):int(i*12.+10),:,:],0)
    tas_hist_ssp245_yr[i,:,:] = np.mean(tas_hist_ssp245[int(i*12.+4):int(i*12.+10),:,:],0) # May-Sep TAS
    tas_hist_ssp585_yr[i,:,:] = np.mean(tas_hist_ssp585[int(i*12.+4):int(i*12.+10),:,:],0)


# Standard deviation of reanalysis datasets
nc_in = './data_for_figures/Reanalysis_std_regrid.nc'
f = Dataset(nc_in,'r')
precip_std_mm_JanSep = f.variables['precip_std'][:]  # convert mm per Jan-Sep
T2M_std = f.variables['T2M_std'][:]
f.close()
#
# convert mm per Jan-Sep to kg/m2/s (mm/s)
# 272 days
# 60.*60.*24.s/d
precip_std = precip_std_mm_JanSep / (60.*60.*24.*272.)
#

# =========== FOREST MASK ===========
nc_out = './data_for_figures/Canada_forest_mask.nc'
f = Dataset(nc_out,'r')
lon_MERRA2 = f.variables['lon'][:]
lat_MERRA2 = f.variables['lat'][:]
Forest_mask = f.variables['mask'][:]
f.close()
# ===================================
# Regrid forest mask
Forest_mask_CMIP = np.zeros((np.size(lat),np.size(lon)))
for i in range(np.size(lat)):
    iii = np.argmin(np.abs(lat_MERRA2-lat[i]))
    for j in range(np.size(lon)):
        jjj = np.argmin(np.abs(lon_MERRA2-lon[j]))
        #
        Forest_mask_CMIP[i,j] = Forest_mask[iii,jjj]
        
# Apply forest mask
Forest_mask_CMIP_arr = np.repeat(Forest_mask_CMIP[np.newaxis, :, :], 101, axis=0)
pr_hist_ssp245_yr[np.where(Forest_mask_CMIP_arr != 1)] = np.nan
pr_hist_ssp585_yr[np.where(Forest_mask_CMIP_arr != 1)] = np.nan
tas_hist_ssp245_yr[np.where(Forest_mask_CMIP_arr != 1)] = np.nan
tas_hist_ssp585_yr[np.where(Forest_mask_CMIP_arr != 1)] = np.nan


# Calculate z-scores
pr_hist_ssp245_zscore = pr_hist*0.
tas_hist_ssp245_zscore = pr_hist*0.
pr_hist_ssp585_zscore = pr_hist*0.
tas_hist_ssp585_zscore = pr_hist*0.
pr_year_ssp245_mean_zscore = np.zeros(101)
tas_year_ssp245_mean_zscore = np.zeros(101)
for i in range(101):
    pr_hist_ssp245_zscore[i,:,:] = ( pr_hist_ssp245_yr[i,:,:]-np.mean(pr_hist_ssp245_yr[0:20,:,:],0) ) / precip_std
    tas_hist_ssp245_zscore[i,:,:] = ( tas_hist_ssp245_yr[i,:,:]-np.mean(tas_hist_ssp245_yr[0:20,:,:],0) ) / T2M_std # np.std(tas_hist_ssp245_yr[0:20,:,:],0)
    pr_hist_ssp585_zscore[i,:,:] = ( pr_hist_ssp585_yr[i,:,:]-np.mean(pr_hist_ssp585_yr[0:20,:,:],0) ) / precip_std #np.std(pr_hist_ssp585_yr[0:20,:,:],0)
    tas_hist_ssp585_zscore[i,:,:] = ( tas_hist_ssp585_yr[i,:,:]-np.mean(tas_hist_ssp585_yr[0:20,:,:],0) ) / T2M_std # np.std(tas_hist_ssp585_yr[0:20,:,:],0)
    #
    pr_year_ssp245_mean_zscore[i] = np.nanmean(pr_hist_ssp245_zscore[i,:,:].flatten())
    tas_year_ssp245_mean_zscore[i] = np.nanmean(tas_hist_ssp245_zscore[i,:,:].flatten())

pr_decade_ssp245_mean_zscore = np.zeros(10)
tas_decade_ssp245_mean_zscore = np.zeros(10)    
for d in range(10):    
    pr_decade_ssp245_mean_zscore[d] = np.nanmean(pr_hist_ssp245_zscore[d*10:(d+1)*10,:,:].flatten())
    tas_decade_ssp245_mean_zscore[d] = np.nanmean(tas_hist_ssp245_zscore[d*10:(d+1)*10,:,:].flatten())

    
pr_decade_ssp585_mean_zscore = np.zeros(10)
tas_decade_ssp585_mean_zscore = np.zeros(10)    
for d in range(10):    
    pr_decade_ssp585_mean_zscore[d] = np.nanmean(pr_hist_ssp585_zscore[d*10:(d+1)*10,:,:].flatten())
    tas_decade_ssp585_mean_zscore[d] = np.nanmean(tas_hist_ssp585_zscore[d*10:(d+1)*10,:,:].flatten())

    
# Create grids for plotting
T2M_zval = np.arange(33)/2.-8
precip_zval = np.arange(33)/2.-8
#                                                                                                                             
T2M_zval2 = np.arange(65)/4.-8
precip_zval2 = np.arange(65)/4.-8


X,Y = np.meshgrid(precip_zval,T2M_zval)
#
fig = plt.figure(817792, figsize=(4.5*0.925,3*0.925), dpi=300)
ax1 = fig.add_axes([0.06/1.+0.044,0.158,0.55/1.,0.75])
tt=plt.pcolormesh(X,Y,ma.masked_invalid(np.log10(GFED_tot_perFireR)),vmin=0.0,vmax=2.3,cmap='inferno_r') 
plt.xlim([-3.25,3.25])
plt.ylim([-3.25,3.25])
plt.plot([0,0],[-3.25,3.25],'k:')
plt.plot([-3.25,3.25],[0,0],'k:')
plt.xlabel('Jan-Sep Precipitation Z-score')
plt.ylabel('May-Sep Termperature Z-score')
plt.yticks([-3,-2,-1,0,1,2,3])
plt.xticks([-3,-2,-1,0,1,2,3])
ax1.set_xticklabels(['-3','-2','-1','0','1','2','3'])
ax1.set_yticklabels(['-3','-2','-1','0','1','2','3'])
plt.title('Fire and climate anomalies')
#
for i in range(20):
    l1=plt.plot(precip_zscore_mean[i],T2M_zscore_mean[i],'kx', markersize=10,mew=2.0,zorder=1)
l2=plt.plot(precip_zscore_mean[20],T2M_zscore_mean[20],'rx', markersize=10,mew=2.5)
legend=plt.legend([l1[0],l2[0]],('2003-2022','2023'),loc='lower left',frameon=True,handletextpad=0.5,labelspacing=0.4,handlelength=1.,ncol=2,bbox_to_anchor=(-0.0075, -0.0175),columnspacing=0.5,fontsize=9,framealpha=1.0)

cmap1 = plt.cm.YlGnBu
bounds1 = [2000,2010,2020,2030,2040,2050,2060,2070,2080,2090,2100]
norm1 = mpl.colors.BoundaryNorm(bounds1, cmap1.N)

tt2=plt.scatter(pr_decade_ssp245_mean_zscore,tas_decade_ssp245_mean_zscore,c=np.arange(10)*10.+2005,s=55.,edgecolors='black',zorder=2,cmap=cmap1,norm=norm1)
#
ax1 = fig.add_axes([0.05/1.+0.021+0.601/1.,0.158,0.025,0.75])
cbar=plt.colorbar(tt,cax=ax1,extend='both',ticks=[])
plt.text(1.07,2,'$10^2$',ha='left',va='center')
plt.text(1.07,1,'$10^1$',ha='left',va='center')
plt.text(1.07,0,'$10^0$',ha='left',va='center')
plt.text(4.25,1.15,'Fire CO$_2$+CO ($\mathrm{gC \, m^{-2}}$)',ha='center',va='center',rotation=270)
#
ax1 = fig.add_axes([0.00/1.+0.017+0.67/1.+0.145,0.158,0.025,0.75])
plt.colorbar(tt2,cax=ax1, orientation='vertical',ticks=[])
plt.text(1.15,2005,'2000s',ha='left',va='center',fontsize=8.5)
plt.text(1.15,2015,'2010s',ha='left',va='center',fontsize=8.5)
plt.text(1.15,2025,'2020s',ha='left',va='center',fontsize=8.5)
plt.text(1.15,2035,'2030s',ha='left',va='center',fontsize=8.5)
plt.text(1.15,2045,'2040s',ha='left',va='center',fontsize=8.5)
plt.text(1.15,2055,'2050s',ha='left',va='center',fontsize=8.5)
plt.text(1.15,2065,'2060s',ha='left',va='center',fontsize=8.5)
plt.text(1.15,2075,'2070s',ha='left',va='center',fontsize=8.5)
plt.text(1.15,2085,'2080s',ha='left',va='center',fontsize=8.5)
plt.text(1.15,2095,'2090s',ha='left',va='center',fontsize=8.5)
plt.text(5.5,2050,'Decade of SSP2-4.5 projection',ha='center',va='center',rotation=270)
#
#plt.savefig('Figures/Byrne_etal_Fig3.png')
plt.savefig('Figures/Byrne_etal_Fig3.eps', format='eps')


 
X,Y = np.meshgrid(precip_zval,T2M_zval)
#
fig = plt.figure(117792, figsize=(4.5*0.925,3*0.925), dpi=300)
ax1 = fig.add_axes([0.06/1.+0.044,0.158,0.55/1.,0.75])
tt=plt.pcolormesh(X,Y,ma.masked_invalid(np.log10(GFED_tot_perFireR)),vmin=0.0,vmax=2.3,cmap='inferno_r') 
plt.xlim([-3.25,3.25])
plt.ylim([-3.25,3.25])
plt.plot([0,0],[-3.25,3.25],'k:')
plt.plot([-3.25,3.25],[0,0],'k:')
plt.xlabel('Jan-Sep Precipitation Z-score')
plt.ylabel('May-Sep Termperature Z-score')
plt.yticks([-3,-2,-1,0,1,2,3])
plt.xticks([-3,-2,-1,0,1,2,3])
ax1.set_xticklabels(['-3','-2','-1','0','1','2','3'])
ax1.set_yticklabels(['-3','-2','-1','0','1','2','3'])
plt.title('Fire and climate anomalies')
#
for i in range(20):
    l1=plt.plot(precip_zscore_mean[i],T2M_zscore_mean[i],'kx', markersize=10,mew=2.0,zorder=1)
l2=plt.plot(precip_zscore_mean[20],T2M_zscore_mean[20],'rx', markersize=10,mew=2.5)
legend=plt.legend([l1[0],l2[0]],('2003-2022','2023'),loc='lower left',frameon=True,handletextpad=0.5,labelspacing=0.4,handlelength=1.,ncol=2,bbox_to_anchor=(-0.0075, -0.0175),columnspacing=0.5,fontsize=9,framealpha=1.0)
#

cmap1 = plt.cm.YlGnBu
bounds1 = [2000,2010,2020,2030,2040,2050,2060,2070,2080,2090,2100]
norm1 = mpl.colors.BoundaryNorm(bounds1, cmap1.N)

tt2=plt.scatter(pr_decade_ssp585_mean_zscore,tas_decade_ssp585_mean_zscore,c=np.arange(10)*10.+2005,s=55.,edgecolors='black',zorder=2,cmap=cmap1,norm=norm1)
#
ax1 = fig.add_axes([0.05/1.+0.021+0.601/1.,0.158,0.025,0.75])
cbar=plt.colorbar(tt,cax=ax1,extend='both',ticks=[])
plt.text(1.07,2,'$10^2$',ha='left',va='center')
plt.text(1.07,1,'$10^1$',ha='left',va='center')
plt.text(1.07,0,'$10^0$',ha='left',va='center')
plt.text(4.25,1.15,'Fire CO$_2$+CO ($\mathrm{gC \, m^{-2}}$)',ha='center',va='center',rotation=270)
#
ax1 = fig.add_axes([0.00/1.+0.017+0.67/1.+0.145,0.158,0.025,0.75])
plt.colorbar(tt2,cax=ax1, orientation='vertical',ticks=[])
plt.text(1.15,2005,'2000s',ha='left',va='center',fontsize=8.5)
plt.text(1.15,2015,'2010s',ha='left',va='center',fontsize=8.5)
plt.text(1.15,2025,'2020s',ha='left',va='center',fontsize=8.5)
plt.text(1.15,2035,'2030s',ha='left',va='center',fontsize=8.5)
plt.text(1.15,2045,'2040s',ha='left',va='center',fontsize=8.5)
plt.text(1.15,2055,'2050s',ha='left',va='center',fontsize=8.5)
plt.text(1.15,2065,'2060s',ha='left',va='center',fontsize=8.5)
plt.text(1.15,2075,'2070s',ha='left',va='center',fontsize=8.5)
plt.text(1.15,2085,'2080s',ha='left',va='center',fontsize=8.5)
plt.text(1.15,2095,'2090s',ha='left',va='center',fontsize=8.5)
plt.text(5.5,2050,'Decade of SSP5-8.5 projection',ha='center',va='center',rotation=270)
#
plt.savefig('Figures/Byrne_etal_FigS6.png')


