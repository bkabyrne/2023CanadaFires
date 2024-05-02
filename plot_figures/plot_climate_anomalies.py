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

'''
-------- plot_climate_anomalies.py

This code processes data and plots Figures 2, S4 and S5

contact: Brendan Byrne
email: brendan.k.byrne@jpl.nasa.gov

'''

days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
days_in_month_cum = np.zeros(13)
for i in range(13):
    days_in_month_cum[i] = np.sum(days_in_month[0:i])

nc_out = './data_for_figures/Canada_forest_mask.nc'
f = Dataset(nc_out,'r')
lon_MERRA2 = f.variables['lon'][:]
lat_MERRA2 = f.variables['lat'][:]
Forest_mask = f.variables['mask'][:]
f.close()

nc_out='./data_for_figures/GFED_Forest_timeseries.nc'
f = Dataset(nc_out,'r')
GFED_timeseries = f.variables['GFED_CO2andCO'][:] # TgC
GFED_timeseriesf1 = f.variables['GFED_CO2andCOf1'][:] # TgC
GFED_timeseriesf2 = f.variables['GFED_CO2andCOf2'][:] # TgC
f.close()


nc_out = './data_for_figures/GFED_MERRA2_regrid_MaytoSep.nc'
f = Dataset(nc_out,'r')
GFED_CO_Flux_per_area = f.variables['CO_Flux'][:] # gC/m2
GFED_CO2_Flux_per_area = f.variables['CO2_Flux'][:] # gC/m2
f.close()
GFED_total_Flux_per_area = GFED_CO2_Flux_per_area + GFED_CO_Flux_per_area
GFED_forest_map = GFED_total_Flux_per_area * 0.                                                                                               
for ii in range(21):
    #
    prior_temp = GFED_total_Flux_per_area[ii,:,:]
    prior_temp[np.where(Forest_mask==0)] = np.nan
    GFED_forest_map[ii,:,:] = prior_temp
    #
GFED_anom_map = GFED_forest_map[20,:,:] - np.mean(GFED_forest_map[0:20,:,:],0)  


nc_out='./data_for_figures/T2M_timeseries.nc'
f = Dataset(nc_out,'r')
T2M_timeseries = f.variables['T2M'][:] # TgC
T2M_timeseriesf1 = f.variables['T2Mf1'][:] # TgC
T2M_timeseriesf2 = f.variables['T2Mf2'][:] # TgC
T2M_timeseries[np.where(T2M_timeseries>200)] = np.nan
T2M_timeseriesf1[np.where(T2M_timeseriesf1>200)] = np.nan
T2M_timeseriesf2[np.where(T2M_timeseriesf2>200)] = np.nan
f.close()
nc_out='./data_for_figures/T2M_anom_map.nc'
f = Dataset(nc_out,'r')
T2M_anom_map = f.variables['T2M_anom'][:] # TgC
T2M_mean_map = f.variables['T2M_mean'][:] # TgC
f.close()

nc_out='./data_for_figures/VPD_timeseries.nc'
f = Dataset(nc_out,'r')
VPD_timeseries = f.variables['VPD'][:] # TgC
VPD_timeseriesf1 = f.variables['VPDf1'][:] # TgC
VPD_timeseriesf2 = f.variables['VPDf2'][:] # TgC
f.close()
nc_out='./data_for_figures/VPD_anom_map.nc'
f = Dataset(nc_out,'r')
VPD_anom_map = f.variables['VPD_anom'][:] # TgC
VPD_mean_map = f.variables['VPD_mean'][:] # TgC
#VPD_anom_map = f.variables['VPD'][:] # TgC
f.close()

nc_out='./data_for_figures/precip_timeseries.nc'
f = Dataset(nc_out,'r')
Precip_timeseries = f.variables['Precip'][:]/10. #cm
Precip_timeseriesf1 = f.variables['Precipf1'][:]/10. #cm
Precip_timeseriesf2 = f.variables['Precipf2'][:]/10. #cm
f.close()
nc_out='./data_for_figures/precip_anom_map_regrid.nc'
f = Dataset(nc_out,'r')
Precip_anom_map = f.variables['precip_anom'][:] /10. #cm
Precip_mean_map = f.variables['precip_mean'][:] /10. #cm
#Precip_anom_map = f.variables['precip'][:]/10. #cm
f.close()

fig = plt.figure(98, figsize=(8*0.7,10*0.7), dpi=300)
#                                                                               
m = Basemap(width=5040000,height=3600000,resolution='l',projection='laea',lat_ts=58,lat_0=58.,lon_0=(-150-40)/2.)
X,Y = np.meshgrid(lon_MERRA2[1:360],lat_MERRA2[180:360])
xx,yy=m(X,Y)
#                                                                               
# ===========================================================================================
Precip_cum_d272 = Precip_timeseries[:,272]
Precip_cum_anom = Precip_cum_d272[20] - np.mean(Precip_cum_d272[0:20])
Precip_cum_anom_zscore = Precip_cum_anom / np.std(Precip_cum_d272[0:20])
#
doy=np.arange(365)+1                                           
ax1 = fig.add_axes([0.05+0.05,0.0390+3./4.,1.5/3.-0.05,0.84/4.])
m.drawlsmask(land_color='white',ocean_color='gainsboro',lakes=True)
m.drawcoastlines(linewidth=0.5,color='grey')
Precip2 = Precip_anom_map * 1.
Precip2[np.where(np.isfinite(T2M_anom_map)==0)] = np.nan
tt = m.pcolormesh(xx,yy,Precip2[180:360,1:360],cmap='RdBu',vmin=-45,vmax=45)
m.drawstates(linewidth=0.5,color='grey')
m.drawcountries(linewidth=0.5,color='grey')
plt.annotate('(ai)', xy=(4./364, 0.98), xycoords='axes fraction',va='top',ha='left')
#                                                                               
ax1 = fig.add_axes([0.09+0.025, 0.0390+3./4., 0.015, 0.84/4.])
plt.colorbar(tt,cax=ax1,extend='both',location='left',label='Jan-Sep $\Delta \Sigma$P (cm)')
#                                                                               
ax1 = fig.add_axes([0.11+0.05+1.3/3., 0.0390+3./4., 1.35/3.-0.05, 0.84/4.])
for i in range(20):
    plt.plot(doy,Precip_timeseries[i,:],'k',alpha=0.3,linewidth=0.5)
l1=plt.plot(doy,np.mean(Precip_timeseries[0:20,:],0),'k')
l2=plt.plot(doy[0:273],Precip_timeseries[20,0:273],'r')
plt.text(5,62.*0.85,'$\mathrm{Jan-Sep}$',ha='left',va='top',fontsize=8)
plt.text(5,62.*0.76,'$\Delta \Sigma$P = '+str(np.round(Precip_cum_anom*10.)/10.)+' cm',ha='left',va='top',fontsize=8)
plt.text(5,62.*0.67,'Z-score = '+str(np.round(Precip_cum_anom_zscore*10.)/10.),ha='left',va='top',fontsize=8)
plt.ylim([0,62])
plt.xlim([1,365])
plt.xticks(days_in_month_cum+1)
plt.text(5,62*0.98,'(aii) $\Sigma$P (cm)',va='top',ha='left')
ax1.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D','J'])
plt.legend([l1[0],l2[0]],('2003-2022','2023'),loc='lower right',frameon=False,handletextpad=0.3,labelspacing=0.5,handlelength=1.5,bbox_to_anchor=(1.04, -0.06))
# ===========================================================================================
T2M_timeseries_anom = np.mean(T2M_timeseries[20,121:273]) - np.mean(np.mean(T2M_timeseries[0:20,121:273],1))
T2M_timeseries_anom_zscore = T2M_timeseries_anom / np.std(np.mean(T2M_timeseries[0:20,121:273],1))
#
doy=np.arange(366)+1
ax1 = fig.add_axes([0.05+0.05,0.0390+2./4.,1.5/3.-0.05,0.84/4.])
m.drawlsmask(land_color='white',ocean_color='gainsboro',lakes=True)
m.drawcoastlines(linewidth=0.5,color='grey')
tt = m.pcolormesh(xx,yy,T2M_anom_map[180:360,1:360],cmap='RdBu_r',vmin=-3,vmax=3)
m.drawstates(linewidth=0.5,color='grey')
m.drawcountries(linewidth=0.5,color='grey')
plt.annotate('(bi)', xy=(4./364, 0.98), xycoords='axes fraction',va='top',ha='left')
#                                                                                
ax1 = fig.add_axes([0.09+0.025, 0.0390+2./4., 0.015, 0.84/4.])
plt.colorbar(tt,cax=ax1,extend='both',location='left',label='May-Sep $\Delta$T2M ($^\circ$C)')
#                                                                                
ax1 = fig.add_axes([0.11+0.05+1.3/3., 0.0390+2./4., 1.35/3.-0.05, 0.84/4.])
T2M_timeseries_2w = T2M_timeseries*np.nan
doy_2w = doy*np.nan
for i in range(365-14):
    T2M_timeseries_2w[:,i+7] = np.mean(T2M_timeseries[:,i:i+14],1)
    doy_2w[i] = np.mean(doy[i:i+14])
#for i in range(20):
#    plt.plot(doy_2w,T2M_timeseries_2w[i,:],'k',alpha=0.22,linewidth=0.5)
plt.fill_between(doy_2w,np.mean(T2M_timeseries_2w[0:20,:],0)-np.std(T2M_timeseries_2w[0:20,:],0),np.mean(T2M_timeseries_2w[0:20,:],0)+np.std(T2M_timeseries_2w[0:20,:],0),color='k',alpha=0.3)
plt.plot(doy_2w,np.mean(T2M_timeseries_2w[0:20,:],0),'k')
plt.plot(doy_2w,T2M_timeseries_2w[20,:],'r')
plt.text(5,-20+(20+20)*0.85,'$\mathrm{May-Sep}$',ha='left',va='top',fontsize=8)
plt.text(5,-20+(20+20)*0.76,'$\Delta$T2M = '+str(np.round(T2M_timeseries_anom*10.)/10.)+'$^\circ$C',ha='left',va='top',fontsize=8)
plt.text(5,-20+(20+20)*0.67,'Z-score = '+str(np.round(T2M_timeseries_anom_zscore*10.)/10.),ha='left',va='top',fontsize=8)
plt.ylim([-20,20])
plt.xlim([1,365])
plt.xticks(days_in_month_cum+1)
plt.text(5,-20+(20+20)*0.98,'(bii) T2M ($^\circ$C)',va='top',ha='left')
ax1.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D','J'])
# ===========================================================================================
VPD_timeseries_anom = np.mean(VPD_timeseries[20,121:273]) - np.mean(np.mean(VPD_timeseries[0:20,121:273],1))
VPD_timeseries_anom_zscore = VPD_timeseries_anom / np.std(np.mean(VPD_timeseries[0:20,121:273],1))
#
doy=np.arange(366)+1
ax1 = fig.add_axes([0.05+0.05,0.0390+1./4.,1.5/3.-0.05,0.84/4.])
m.drawlsmask(land_color='white',ocean_color='gainsboro',lakes=False)
m.drawcoastlines(linewidth=0.5,color='grey')
tt = m.pcolormesh(xx,yy,VPD_anom_map[180:360,1:360],cmap='RdBu_r',vmin=-2,vmax=2)
m.drawstates(linewidth=0.5,color='grey')
m.drawcountries(linewidth=0.5,color='grey')
plt.annotate('(ci)', xy=(4./364, 0.98), xycoords='axes fraction',va='top',ha='left')
#                                                                               
ax1 = fig.add_axes([0.09+0.025, 0.0390+1./4., 0.015, 0.84/4.])
plt.colorbar(tt,cax=ax1,extend='both',location='left',label='May-Sep $\Delta$VPD (hPa)')
#                                                                               
ax1 = fig.add_axes([0.11+1.3/3.+0.05, 0.0390+1./4., 1.35/3.-0.05, 0.84/4.])
VPD_timeseries_2w = VPD_timeseries*np.nan
doy_2w = doy*np.nan
for i in range(365-14):
    VPD_timeseries_2w[:,i+7] = np.mean(VPD_timeseries[:,i:i+14],1)
    doy_2w[i] = np.mean(doy[i:i+14])
#for i in range(20):
#    plt.plot(doy_2w,VPD_timeseries_2w[i,:],'k',alpha=0.22,linewidth=0.5)
plt.fill_between(doy_2w,np.mean(VPD_timeseries_2w[0:20,:],0)-np.std(VPD_timeseries_2w[0:20,:],0),np.mean(VPD_timeseries_2w[0:20,:],0)+np.std(VPD_timeseries_2w[0:20,:],0),color='k',alpha=0.3)
plt.plot(doy_2w,np.mean(VPD_timeseries_2w[0:20,:],0),'k')
plt.plot(doy_2w,VPD_timeseries_2w[20,:],'r')
plt.text(5,6.5*0.85,'$\mathrm{May-Sep}$',ha='left',va='top',fontsize=8)
plt.text(5,6.5*0.76,'$\Delta$VPD = '+str(np.round(VPD_timeseries_anom*10.)/10.)+' hPa',ha='left',va='top',fontsize=8)
plt.text(5,6.5*0.67,'Z-score = '+str(np.round(VPD_timeseries_anom_zscore*10.)/10.),ha='left',va='top',fontsize=8)
plt.ylim([0,6.5])
plt.xlim([1,365])
plt.xticks(days_in_month_cum+1)
plt.text(5,6.5*0.98,'(cii) VPD (hPa)',va='top',ha='left')
ax1.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D','J'])
# ===========================================================================================
GFED_timeseries_anom = GFED_timeseries[20,272] - np.mean(GFED_timeseries[0:20,272])
GFED_timeseries_anom_zscore = GFED_timeseries_anom / np.std(np.mean(GFED_timeseries[0:20,121:273],1))
#
doy=np.arange(365)+1
ax1 = fig.add_axes([0.05+0.05,0.0390+0./1.,1.5/3.-0.05,0.84/4.])
m.drawlsmask(land_color='white',ocean_color='gainsboro',lakes=False)
m.drawcoastlines(linewidth=0.5,color='grey')
tt = m.pcolormesh(xx,yy,np.log10(GFED_anom_map[180:360,1:360]),cmap='inferno_r',vmin=0,vmax=4)
m.drawstates(linewidth=0.5,color='grey')
m.drawcountries(linewidth=0.5,color='grey')
plt.annotate('(di)', xy=(4./364, 0.98), xycoords='axes fraction',va='top',ha='left')
#                                                                                
ax1 = fig.add_axes([0.09+0.025, 0.0390, 0.015, 0.84/4.])
cbar = plt.colorbar(tt,cax=ax1,extend='both',location='left',label='2023 Fires ($\mathrm{gC \, m^{-2}}$)',ticks=[0,1,2,3,4])
cbar.ax.set_yticklabels(['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])
#                                                                                
ax1 = fig.add_axes([0.11+1.3/3.+0.05, 0.0390, 1.35/3.-0.05, 0.84/4.])
for i in range(20):
    plt.plot(doy,GFED_timeseries[i,:],'k',alpha=0.3,linewidth=0.5)
l1=plt.plot(doy,np.mean(GFED_timeseries[0:20,:],0),'k')
l2=plt.plot(doy[0:273],GFED_timeseries[20,0:273],'r')
#plt.text(5,750*0.70,'2023 $\Sigma$Fire = '+str(np.round(GFED_timeseries[20,272]))+' TgC',ha='left',va='top',fontsize=8)
#plt.text(5,750*0.61,'2003-22 $\Sigma$Fire = '+str(np.round(np.mean(GFED_timeseries[0:20,272])))+' TgC',ha='left',va='top',fontsize=8)
#plt.text(5,750*0.7,'$\mathrm{May-Sep}$',ha='left',va='top',fontsize=8)
#plt.text(5,750*0.61,'$\Delta \Sigma$Fire = '+str(np.round(GFED_timeseries_anom))+' TgC',ha='left',va='top',fontsize=8)
#plt.text(5,750*0.52,'Z-score = '+str(np.round(GFED_timeseries_anom_zscore*10.)/10.),ha='left',va='top',fontsize=8)
plt.ylim([0,750])
plt.xlim([1,365])
plt.xticks(days_in_month_cum+1)
plt.text(5,750*0.98,'(dii)',va='top',ha='left')
plt.text(55,750*0.98,'Cumulative fire',va='top',ha='left')
plt.text(55,665*0.98,'CO$_2 +$CO (TgC)',va='top',ha='left')
ax1.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D','J'])
#
plt.savefig('Figures/Byrne_etal_Fig2.png', dpi=300)





fig = plt.figure(198, figsize=(4*0.7,10*0.7), dpi=300)
#                                                                               
m = Basemap(width=5040000,height=3600000,resolution='l',projection='laea',lat_ts=58,lat_0=58.,lon_0=(-150-40)/2.)
X,Y = np.meshgrid(lon_MERRA2[1:360],lat_MERRA2[180:360])
xx,yy=m(X,Y)
#                                                                               
# ===========================================================================================
#
doy=np.arange(365)+1                                           
ax1 = fig.add_axes([0.05+0.05+0.04+0.005,0.0290+3./4.,3./3.-0.05,0.84/4.])
m.drawlsmask(land_color='white',ocean_color='gainsboro',lakes=True)
m.drawcoastlines(linewidth=0.5,color='grey')
Precip2 = Precip_mean_map * 1.
Precip2[np.where(np.isfinite(T2M_anom_map)==0)] = np.nan
tt = m.pcolormesh(xx,yy,Precip2[180:360,1:360],cmap='gist_earth_r',vmin=0,vmax=90)
m.drawstates(linewidth=0.5,color='grey')
m.drawcountries(linewidth=0.5,color='grey')
plt.annotate('(a)', xy=(4./364, 0.98), xycoords='axes fraction',va='top',ha='left')
#                                                                               
ax1 = fig.add_axes([0.09+0.025+0.04+0.04+0.007, 0.0290+3./4., 0.03, 0.84/4.])
plt.colorbar(tt,cax=ax1,extend='max',location='left',label='Jan-Sep $\Sigma$P (cm)')
#                                                                               
# ===========================================================================================
#
doy=np.arange(366)+1
ax1 = fig.add_axes([0.05+0.05+0.04+0.005,0.0290+2./4.,3./3.-0.05,0.84/4.])
m.drawlsmask(land_color='white',ocean_color='gainsboro',lakes=True)
m.drawcoastlines(linewidth=0.5,color='grey')
tt = m.pcolormesh(xx,yy,T2M_mean_map[180:360,1:360],cmap='plasma',vmin=5,vmax=18)
m.drawstates(linewidth=0.5,color='grey')
m.drawcountries(linewidth=0.5,color='grey')
plt.annotate('(b)', xy=(4./364, 0.98), xycoords='axes fraction',va='top',ha='left')
#                                                                                
ax1 = fig.add_axes([0.09+0.025+0.04+0.04+0.007, 0.0290+2./4., 0.03, 0.84/4.])
plt.colorbar(tt,cax=ax1,extend='both',location='left',label='May-Sep T2M ($^\circ$C)')
#                                                                                
# ===========================================================================================
#
doy=np.arange(366)+1
ax1 = fig.add_axes([0.05+0.05+0.04+0.005,0.0290+1./4.,3./3.-0.05,0.84/4.])
m.drawlsmask(land_color='white',ocean_color='gainsboro',lakes=False)
m.drawcoastlines(linewidth=0.5,color='grey')
tt = m.pcolormesh(xx,yy,VPD_mean_map[180:360,1:360],cmap='plasma',vmin=0,vmax=7.5)
m.drawstates(linewidth=0.5,color='grey')
m.drawcountries(linewidth=0.5,color='grey')
plt.annotate('(c)', xy=(4./364, 0.98), xycoords='axes fraction',va='top',ha='left')
#                                                                               
ax1 = fig.add_axes([0.09+0.025+0.04+0.04+0.007, 0.0290+1./4., 0.03, 0.84/4.])
plt.colorbar(tt,cax=ax1,extend='max',location='left',label='May-Sep VPD (hPa)')
#                                                                               
# ===========================================================================================
#
doy=np.arange(365)+1
ax1 = fig.add_axes([0.05+0.05+0.04+0.005,0.0290+0./1.,3./3.-0.05,0.84/4.])
m.drawlsmask(land_color='white',ocean_color='gainsboro',lakes=False)
m.drawcoastlines(linewidth=0.5,color='grey')
tt = m.pcolormesh(xx,yy,np.log10(np.mean(GFED_forest_map[0:20,:,:],0)[180:360,1:360]),cmap='inferno_r',vmin=0,vmax=4)
m.drawstates(linewidth=0.5,color='grey')
m.drawcountries(linewidth=0.5,color='grey')
plt.annotate('(d)', xy=(4./364, 0.98), xycoords='axes fraction',va='top',ha='left')
#                                                                                
ax1 = fig.add_axes([0.09+0.025+0.04+0.04+0.007, 0.0290, 0.03, 0.84/4.])
cbar = plt.colorbar(tt,cax=ax1,extend='both',location='left',label='Fires ($\mathrm{gC \, m^{-2}}$)',ticks=[0,1,2,3,4])
cbar.ax.set_yticklabels(['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])
#                                                                                
plt.savefig('Figures/Byrne_etal_FigS4.png', dpi=300)




fig = plt.figure(928, figsize=(8*0.7,10*0.7), dpi=300)
#                                                                               
m = Basemap(width=5040000,height=3600000,resolution='l',projection='laea',lat_ts=58,lat_0=58.,lon_0=(-150-40)/2.)
X,Y = np.meshgrid(lon_MERRA2[1:360],lat_MERRA2[180:360])
xx,yy=m(X,Y)
#                                                                               
# ===========================================================================================
Precip_cum_d272f2 = Precip_timeseriesf2[:,272]
Precip_cum_anomf2 = Precip_cum_d272f2[20] - np.mean(Precip_cum_d272f2[0:20])
Precip_cum_anom_zscoref2 = Precip_cum_anomf2 / np.std(Precip_cum_d272f2[0:20])
doy=np.arange(365)+1                                           
ax1 = fig.add_axes([0.135+0./3., 0.0390+3./4.-0.15/4., 1.35/3.-0.05, 0.84/4.])
for i in range(20):
    plt.plot(doy,Precip_timeseriesf2[i,:],'k',alpha=0.3,linewidth=0.5)
l1=plt.plot(doy,np.mean(Precip_timeseriesf2[0:20,:],0),'k')
l2=plt.plot(doy[0:273],Precip_timeseriesf2[20,0:273],'r')
plt.text(5,62.*0.98,'$\mathrm{Jan-Sep}$',ha='left',va='top',fontsize=8)
plt.text(5,62.*0.89,'$\Delta \Sigma$P = '+str(np.round(Precip_cum_anomf2*10.)/10.)+' cm',ha='left',va='top',fontsize=8)
plt.text(5,62.*0.80,'Z-score = '+str(np.round(Precip_cum_anom_zscoref2*10.)/10.),ha='left',va='top',fontsize=8)
plt.ylim([0,62])
plt.xlim([1,365])
plt.xticks(days_in_month_cum+1)
plt.ylabel('$\Sigma$P (cm)')
ax1.set_xticklabels([])
plt.legend([l1[0],l2[0]],('2003-2022','2023'),loc='upper right',frameon=False,handletextpad=0.3,labelspacing=0.5,handlelength=1.5,bbox_to_anchor=(1.04, 1.05))
plt.title('Northwest')
#
Precip_cum_d272f1 = Precip_timeseriesf1[:,272]
Precip_cum_anomf1 = Precip_cum_d272f1[20] - np.mean(Precip_cum_d272f1[0:20])
Precip_cum_anom_zscoref1 = Precip_cum_anomf1 / np.std(Precip_cum_d272f1[0:20])
doy=np.arange(365)+1                                           
ax1 = fig.add_axes([0.11+0.05+1.3/3., 0.0390+3./4.-0.15/4., 1.35/3.-0.05, 0.84/4.])
for i in range(20):
    plt.plot(doy,Precip_timeseriesf1[i,:],'k',alpha=0.3,linewidth=0.5)
l1=plt.plot(doy,np.mean(Precip_timeseriesf1[0:20,:],0),'k')
l2=plt.plot(doy[0:273],Precip_timeseriesf1[20,0:273],'r')
plt.text(5,62.*0.98,'$\mathrm{Jan-Sep}$',ha='left',va='top',fontsize=8)
plt.text(5,62.*0.89,'$\Delta \Sigma$P = '+str(np.round(Precip_cum_anomf1*10.)/10.)+' cm',ha='left',va='top',fontsize=8)
plt.text(5,62.*0.80,'Z-score = '+str(np.round(Precip_cum_anom_zscoref1*10.)/10.),ha='left',va='top',fontsize=8)
plt.ylim([0,62])
plt.xlim([1,365])
plt.xticks(days_in_month_cum+1)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
plt.title('Western Quebec')
# ===========================================================================================
T2M_timeseries_anomf2 = np.mean(T2M_timeseriesf2[20,121:273]) - np.mean(np.mean(T2M_timeseriesf2[0:20,121:273],1))
T2M_timeseries_anom_zscoref2 = T2M_timeseries_anomf2 / np.std(np.mean(T2M_timeseriesf2[0:20,121:273],1))
doy=np.arange(366)+1
ax1 = fig.add_axes([0.135+0./3., 0.0390+2./4.-0.1/4., 1.35/3.-0.05, 0.84/4.])
T2M_timeseries_2wf2 = T2M_timeseriesf2*np.nan
doy_2w = doy*np.nan
for i in range(365-14):
    T2M_timeseries_2wf2[:,i+7] = np.mean(T2M_timeseriesf2[:,i:i+14],1)
    doy_2w[i] = np.mean(doy[i:i+14])
plt.fill_between(doy_2w,np.mean(T2M_timeseries_2wf2[0:20,:],0)-np.std(T2M_timeseries_2wf2[0:20,:],0),np.mean(T2M_timeseries_2wf2[0:20,:],0)+np.std(T2M_timeseries_2wf2[0:20,:],0),color='k',alpha=0.3)
plt.plot(doy_2w,np.mean(T2M_timeseries_2wf2[0:20,:],0),'k')
plt.plot(doy_2w,T2M_timeseries_2wf2[20,:],'r')
plt.text(5,-20+(20+20)*0.98,'$\mathrm{May-Sep}$',ha='left',va='top',fontsize=8)
plt.text(5,-20+(20+20)*0.89,'$\Delta$T2M = '+str(np.round(T2M_timeseries_anomf2*10.)/10.)+'$^\circ$C',ha='left',va='top',fontsize=8)
plt.text(5,-20+(20+20)*0.80,'Z-score = '+str(np.round(T2M_timeseries_anom_zscoref2*10.)/10.),ha='left',va='top',fontsize=8)
plt.ylim([-20,20])
plt.xlim([1,365])
plt.xticks(days_in_month_cum+1)
plt.ylabel('T2M ($^\circ$C)')
ax1.set_xticklabels([])
#
T2M_timeseries_anomf1 = np.mean(T2M_timeseriesf1[20,121:273]) - np.mean(np.mean(T2M_timeseriesf1[0:20,121:273],1))
T2M_timeseries_anom_zscoref1 = T2M_timeseries_anomf1 / np.std(np.mean(T2M_timeseriesf1[0:20,121:273],1))
doy=np.arange(366)+1
ax1 = fig.add_axes([0.11+0.05+1.3/3., 0.0390+2./4.-0.1/4., 1.35/3.-0.05, 0.84/4.])
T2M_timeseries_2wf1 = T2M_timeseriesf1*np.nan
doy_2w = doy*np.nan
for i in range(365-14):
    T2M_timeseries_2wf1[:,i+7] = np.mean(T2M_timeseriesf1[:,i:i+14],1)
    doy_2w[i] = np.mean(doy[i:i+14])
plt.fill_between(doy_2w,np.mean(T2M_timeseries_2wf1[0:20,:],0)-np.std(T2M_timeseries_2wf1[0:20,:],0),np.mean(T2M_timeseries_2wf1[0:20,:],0)+np.std(T2M_timeseries_2wf1[0:20,:],0),color='k',alpha=0.3)
plt.plot(doy_2w,np.mean(T2M_timeseries_2wf1[0:20,:],0),'k')
plt.plot(doy_2w,T2M_timeseries_2wf1[20,:],'r')
plt.text(5,-20+(20+20)*0.98,'$\mathrm{May-Sep}$',ha='left',va='top',fontsize=8)
plt.text(5,-20+(20+20)*0.89,'$\Delta$T2M = '+str(np.round(T2M_timeseries_anomf1*10.)/10.)+'$^\circ$C',ha='left',va='top',fontsize=8)
plt.text(5,-20+(20+20)*0.80,'Z-score = '+str(np.round(T2M_timeseries_anom_zscoref1*10.)/10.),ha='left',va='top',fontsize=8)
plt.ylim([-20,20])
plt.xlim([1,365])
plt.xticks(days_in_month_cum+1)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
# ===========================================================================================
VPD_timeseries_anomf2 = np.mean(VPD_timeseriesf2[20,121:273]) - np.mean(np.mean(VPD_timeseriesf2[0:20,121:273],1))
VPD_timeseries_anom_zscoref2 = VPD_timeseries_anomf2 / np.std(np.mean(VPD_timeseriesf2[0:20,121:273],1))
doy=np.arange(366)+1
ax1 = fig.add_axes([0.135+0./3., 0.0390+1./4.-0.05/4., 1.35/3.-0.05, 0.84/4.])
VPD_timeseries_2wf2 = VPD_timeseriesf2*np.nan
doy_2w = doy*np.nan
for i in range(365-14):
    VPD_timeseries_2wf2[:,i+7] = np.mean(VPD_timeseriesf2[:,i:i+14],1)
    doy_2w[i] = np.mean(doy[i:i+14])
plt.fill_between(doy_2w,np.mean(VPD_timeseries_2wf2[0:20,:],0)-np.std(VPD_timeseries_2wf2[0:20,:],0),np.mean(VPD_timeseries_2wf2[0:20,:],0)+np.std(VPD_timeseries_2wf2[0:20,:],0),color='k',alpha=0.3)
plt.plot(doy_2w,np.mean(VPD_timeseries_2wf2[0:20,:],0),'k')
plt.plot(doy_2w,VPD_timeseries_2wf2[20,:],'r')
plt.text(5,6.5*0.98,'$\mathrm{May-Sep}$',ha='left',va='top',fontsize=8)
plt.text(5,6.5*0.89,'$\Delta$VPD = '+str(np.round(VPD_timeseries_anomf2*10.)/10.)+' hPa',ha='left',va='top',fontsize=8)
plt.text(5,6.5*0.80,'Z-score = '+str(np.round(VPD_timeseries_anom_zscoref2*10.)/10.),ha='left',va='top',fontsize=8)
plt.ylim([0,6.5])
plt.xlim([1,365])
plt.xticks(days_in_month_cum+1)
plt.ylabel('VPD (hPa)')
ax1.set_xticklabels([])
#
VPD_timeseries_anomf1 = np.mean(VPD_timeseriesf1[20,121:273]) - np.mean(np.mean(VPD_timeseriesf1[0:20,121:273],1))
VPD_timeseries_anom_zscoref1 = VPD_timeseries_anomf1 / np.std(np.mean(VPD_timeseriesf1[0:20,121:273],1))
doy=np.arange(366)+1
ax1 = fig.add_axes([0.11+1.3/3.+0.05, 0.0390+1./4.-0.05/4., 1.35/3.-0.05, 0.84/4.])
VPD_timeseries_2wf1 = VPD_timeseriesf1*np.nan
doy_2w = doy*np.nan
for i in range(365-14):
    VPD_timeseries_2wf1[:,i+7] = np.mean(VPD_timeseriesf1[:,i:i+14],1)
    doy_2w[i] = np.mean(doy[i:i+14])
plt.fill_between(doy_2w,np.mean(VPD_timeseries_2wf1[0:20,:],0)-np.std(VPD_timeseries_2wf1[0:20,:],0),np.mean(VPD_timeseries_2wf1[0:20,:],0)+np.std(VPD_timeseries_2wf1[0:20,:],0),color='k',alpha=0.3)
plt.plot(doy_2w,np.mean(VPD_timeseries_2wf1[0:20,:],0),'k')
plt.plot(doy_2w,VPD_timeseries_2wf1[20,:],'r')
plt.text(5,6.5*0.98,'$\mathrm{May-Sep}$',ha='left',va='top',fontsize=8)
plt.text(5,6.5*0.89,'$\Delta$VPD = '+str(np.round(VPD_timeseries_anomf1*10.)/10.)+' hPa',ha='left',va='top',fontsize=8)
plt.text(5,6.5*0.80,'Z-score = '+str(np.round(VPD_timeseries_anom_zscoref1*10.)/10.),ha='left',va='top',fontsize=8)
plt.ylim([0,6.5])
plt.xlim([1,365])
plt.xticks(days_in_month_cum+1)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
# ===========================================================================================
doy=np.arange(365)+1
ax1 = fig.add_axes([0.135+0./3., 0.0390, 1.35/3.-0.05, 0.84/4.])
for i in range(20):
    plt.plot(doy,GFED_timeseriesf2[i,:],'k',alpha=0.3,linewidth=0.5)
l1=plt.plot(doy,np.mean(GFED_timeseriesf2[0:20,:],0),'k')
l2=plt.plot(doy[0:273],GFED_timeseriesf2[20,0:273],'r')
plt.text(5,750*0.98,'2023 $\Sigma$Fire = '+str(np.round(GFED_timeseriesf2[20,272]))+' TgC',ha='left',va='top',fontsize=8)
plt.text(5,750*0.89,'2003-22 $\Sigma$Fire = '+str(np.round(np.mean(GFED_timeseriesf2[0:20,272])))+' TgC',ha='left',va='top',fontsize=8)
plt.ylim([0,750])
plt.xlim([1,365])
plt.xticks(days_in_month_cum+1)
plt.ylabel('$\Sigma$Fire CO$_2 +$CO (TgC)')
ax1.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D','J'])
#
doy=np.arange(365)+1
ax1 = fig.add_axes([0.11+1.3/3.+0.05, 0.0390, 1.35/3.-0.05, 0.84/4.])
for i in range(20):
    plt.plot(doy,GFED_timeseriesf1[i,:],'k',alpha=0.3,linewidth=0.5)
l1=plt.plot(doy,np.mean(GFED_timeseriesf1[0:20,:],0),'k')
l2=plt.plot(doy[0:273],GFED_timeseriesf1[20,0:273],'r')
plt.text(5,750*0.98,'2023 $\Sigma$Fire = '+str(np.round(GFED_timeseriesf1[20,272]))+' TgC',ha='left',va='top',fontsize=8)
plt.text(5,750*0.89,'2003-22 $\Sigma$Fire = '+str(np.round(np.mean(GFED_timeseriesf1[0:20,272])))+' TgC',ha='left',va='top',fontsize=8)
plt.ylim([0,750])
plt.xlim([1,365])
plt.xticks(days_in_month_cum+1)
ax1.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D','J'])
ax1.set_yticklabels([])
#
plt.savefig('Figures/Byrne_etal_FigS5.png', dpi=300)
