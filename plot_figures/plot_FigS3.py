from mpl_toolkits.basemap import Basemap, cm, maskoceans
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
-------- plot_FigS4.py
                                                         
This code processes data and plots Figure S3
                                                         
contact: Brendan Byrne                                  
email: brendan.k.byrne@jpl.nasa.gov                     
                                                        
'''

#
nc_out = './data_for_figures/Canada_forest_mask.nc'
f = Dataset(nc_out,'r')
lon_MERRA2 = f.variables['lon'][:]
lat_MERRA2 = f.variables['lat'][:]
Forest_mask = f.variables['mask'][:]
f.close()

#
nc_out = './data_for_figures/precip_JanSep_MERRA2_regrid_1980to2023.nc'
f = Dataset(nc_out,'r')
precip = f.variables['precip'][:] /10. # cm
area = f.variables['area'][:] # mm
f.close()
#


T2Mt = np.zeros((44,366, 361, 576))
VPDt = np.zeros((44,366, 361, 576))
for yy in range(44):
    year = yy + 1980
    nc_in = './data_for_figures/MERRA2_daily_mean_T2M'+str(year).zfill(4)+'.nc'
    print(nc_in)
    f=Dataset(nc_in,mode='r')
    T2Mt[yy,:,:,:] = f.variables['T2M'][:] + 273.15
    VPDt[yy,:,:,:] = f.variables['VPD'][:]*(-1.)
    f.close()
    


T2M = np.zeros((44,np.size(lat_MERRA2),np.size(lon_MERRA2)))
for yy in range(44):
    year = yy + 1980
    print(year)
    if np.logical_and(year % 4 == 0, year % 100 != 0):
        print('leap year!')
        T2M[yy,:,:] = np.nanmean(T2Mt[yy,121-1+1:274-1+1,:,:],0)
    else:
        T2M[yy,:,:] = np.nanmean(T2Mt[yy,121-1:274-1,:,:],0)

VPD = np.zeros((44,np.size(lat_MERRA2),np.size(lon_MERRA2)))
for yy in range(44):
    year = yy + 1980
    print(year)
    if np.logical_and(year % 4 == 0, year % 100 != 0):
        print('leap year!')
        VPD[yy,:,:] = np.nanmean(VPDt[yy,121-1+1:274-1+1,:,:],0)
    else:
        VPD[yy,:,:] = np.nanmean(VPDt[yy,121-1:274-1,:,:],0)


#
area_mask = area * 1.
area_mask[np.where(Forest_mask==0)] = np.nan

precip_mask = precip*0.
precip_vec = np.zeros(44)
for i in range(44):
    #
    TEMP = precip[i,:,:] * 1.
    TEMP[np.where(Forest_mask==0)] = np.nan
    precip_mask[i,:,:] = TEMP
    #
    precip_vec[i] = np.nansum(TEMP * area_mask) / np.nansum(area_mask)
    
T2M_mask = precip*0.
T2M_vec = np.zeros(44)
for i in range(44):
    #
    TEMP = T2M[i,:,:] * 1.
    TEMP[np.where(Forest_mask==0)] = np.nan
    T2M_mask[i,:,:] = TEMP
    #
    T2M_vec[i] = np.nansum(TEMP * area_mask) / np.nansum(area_mask)

VPD_mask = precip*0.
VPD_vec = np.zeros(44)
for i in range(44):
    #
    TEMP = VPD[i,:,:] * 1.
    TEMP[np.where(Forest_mask==0)] = np.nan
    VPD_mask[i,:,:] = TEMP
    #
    VPD_vec[i] = np.nansum(TEMP * area_mask) / np.nansum(area_mask)



precip_zscore = precip*0.
for i in range(44):
    precip_zscore[i,:,:] = ( precip_mask[i,:,:]-np.mean(precip_mask[23:43,:,:],0) ) / np.std(precip_mask[23:43,:,:],0)

T2M_zscore = precip*0.
for i in range(44):
    T2M_zscore[i,:,:] = ( T2M_mask[i,:,:]-np.mean(T2M_mask[23:43,:,:],0) ) / np.std(T2M_mask[23:43,:,:],0)

VPD_zscore = precip*0.
for i in range(44):
    VPD_zscore[i,:,:] = ( VPD_mask[i,:,:]-np.mean(VPD_mask[23:43,:,:],0) ) / np.std(VPD_mask[23:43,:,:],0)



#
#
#

N_T2M_total = np.zeros(44)
N_T2M_minus1 = np.zeros(44)
N_T2M_0 = np.zeros(44)
N_T2M_1 = np.zeros(44)
#
for iii in range(44):
    #
    T2M_zscore_vec_temp = T2M_zscore[iii,:,:].flatten()
    #
    T2M_zscore_vec_temp2 = T2M_zscore_vec_temp[np.where(np.isfinite(T2M_zscore_vec_temp))]
    #
    N_T2M_total[iii] = np.size(T2M_zscore_vec_temp2)
    N_T2M_minus1[iii] = np.size(T2M_zscore_vec_temp2[np.where(T2M_zscore_vec_temp2>-1)]) / N_T2M_total[iii]
    N_T2M_0[iii] = np.size(T2M_zscore_vec_temp2[np.where(T2M_zscore_vec_temp2>0)]) / N_T2M_total[iii]
    N_T2M_1[iii] = np.size(T2M_zscore_vec_temp2[np.where(T2M_zscore_vec_temp2>1)]) / N_T2M_total[iii]


N_VPD_total = np.zeros(44)
N_VPD_minus1 = np.zeros(44)
N_VPD_0 = np.zeros(44)
N_VPD_1 = np.zeros(44)
#
for iii in range(44):
    #
    VPD_zscore_vec_temp = VPD_zscore[iii,:,:].flatten()
    #
    VPD_zscore_vec_temp2 = VPD_zscore_vec_temp[np.where(np.isfinite(VPD_zscore_vec_temp))]
    #
    N_VPD_total[iii] = np.size(VPD_zscore_vec_temp2)
    N_VPD_minus1[iii] = np.size(VPD_zscore_vec_temp2[np.where(VPD_zscore_vec_temp2>-1)]) / N_VPD_total[iii]
    N_VPD_0[iii] = np.size(VPD_zscore_vec_temp2[np.where(VPD_zscore_vec_temp2>0)]) / N_VPD_total[iii]
    N_VPD_1[iii] = np.size(VPD_zscore_vec_temp2[np.where(VPD_zscore_vec_temp2>1)]) / N_VPD_total[iii]



N_precip_total = np.zeros(44)
N_precip_minus1 = np.zeros(44)
N_precip_0 = np.zeros(44)
N_precip_1 = np.zeros(44)
#
for iii in range(44):
    #
    precip_zscore_vec_temp = precip_zscore[iii,:,:].flatten()
    #
    precip_zscore_vec_temp2 = precip_zscore_vec_temp[np.where(np.isfinite(precip_zscore_vec_temp))]
    #
    N_precip_total[iii] = np.size(precip_zscore_vec_temp2)
    N_precip_minus1[iii] = np.size(precip_zscore_vec_temp2[np.where(precip_zscore_vec_temp2<-1)]) / N_precip_total[iii]
    N_precip_0[iii] = np.size(precip_zscore_vec_temp2[np.where(precip_zscore_vec_temp2<0)]) / N_precip_total[iii]
    N_precip_1[iii] = np.size(precip_zscore_vec_temp2[np.where(precip_zscore_vec_temp2<1)]) / N_precip_total[iii]



cmap_test1 = plt.get_cmap('RdYlBu_r')
colors = cmap_test1(np.linspace(0, 1.0, 4))



years = np.arange(45)+1980

fig = plt.figure(5201, figsize=(7.75*0.7,9*0.7), dpi=300)

ax1 = fig.add_axes([0.135, 0.6/3. + 2./3., .855, 0.30/3.])
plt.plot(years[0:44]+0.5,precip_vec,'k')
plt.plot([1980,2024],[precip_vec[43],precip_vec[43]],'k:')
plt.plot(2023.5,precip_vec[43],'k.')
plt.xticks(years)
plt.xlim([1980,2024])
plt.ylim([np.min(precip_vec)-np.mean(precip_vec)*0.025,np.max(precip_vec)+np.mean(precip_vec)*0.025])
ax1.set_xticklabels([])
plt.ylabel('Jan-Sep $\Sigma$P\n(cm)')
plt.title('(a) Precipitation', pad=-20)
plt.text(1980,np.max(precip_vec)+np.mean(precip_vec)*0.025,'(i)',va='top',ha='left')
ax1 = fig.add_axes([0.135, 0.090/3. + 2./3., .855, 0.4/3.])
for i in range(44):
    plt.fill_between([years[i],years[i+1]],[N_precip_1[i],N_precip_1[i]],[1,1],color=colors[0],alpha=1)
    plt.fill_between([years[i],years[i+1]],[N_precip_0[i],N_precip_0[i]],[N_precip_1[i],N_precip_1[i]],color=colors[1],alpha=1)
    plt.fill_between([years[i],years[i+1]],[N_precip_minus1[i],N_precip_minus1[i]],[N_precip_0[i],N_precip_0[i]],color=colors[2],alpha=1)
    plt.fill_between([years[i],years[i+1]],[0,0],[N_precip_minus1[i],N_precip_minus1[i]],color=colors[3],alpha=1)
plt.plot([1980,2024],[0.25,0.25],'k',linewidth=0.5)
plt.plot([1980,2024],[0.5,0.5],'k',linewidth=0.5)
plt.plot([1980,2024],[0.75,0.75],'k',linewidth=0.5)
plt.xlim([1980,2024])
plt.ylim([0,1])
plt.yticks([0,0.25,0.50,0.75,1.00])
l0 = plt.fill(np.NaN, np.NaN, color=colors[0], alpha=1.)
l1 = plt.fill(np.NaN, np.NaN, color=colors[1], alpha=1.)
l2 = plt.fill(np.NaN, np.NaN, color=colors[2], alpha=1.)
l3 = plt.fill(np.NaN, np.NaN, color=colors[3], alpha=1.)
plt.xticks(years)
plt.legend([l3[0],l2[0],l1[0],l0[0]],('Z-score<-1','-1<Z-score<0','0<Z-score<1','1<Z-score'),loc='upper center',frameon=False,bbox_to_anchor=(0.5, 1.30),ncol=4,handletextpad=0.5,labelspacing=0.4,handlelength=1.,columnspacing=0.5,fontsize=9)
ax1.set_xticklabels([])
ax1.set_yticklabels(['0','25','50','75',''])
plt.ylabel('Percent\nof area')
for i in range(44):
    plt.text(1980.5+i,0,str(int((1980+i)-100*np.floor((1980+i)/100.))).zfill(2),ha='center',va='top',rotation=90,fontsize=7.5)
#plt.savefig('precip_timeseries_20231113.png')
plt.text(1980,1,'(ii)',va='top',ha='left')



#fig = plt.figure(4202, figsize=(7,3), dpi=300)
ax1 = fig.add_axes([0.135, 0.6/3. + 1./3., .855, 0.30/3.])
plt.plot(years[0:44]+0.5,T2M_vec,'k')
plt.plot([1980,2024],[T2M_vec[43],T2M_vec[43]],'k:')
plt.plot(2023.5,T2M_vec[43],'k.')
plt.xticks(years)
plt.xlim([1980,2024])
plt.yticks([10,11,12,13])
ax1.set_xticklabels([])
#plt.ylim([np.min(T2M_vec)*0.9,np.max(T2M_vec)*1.1])
plt.ylim([np.min(T2M_vec)-np.mean(T2M_vec)*0.025,np.max(T2M_vec)+np.mean(T2M_vec)*0.025])
plt.ylabel('May-Sep T2M\n($^\circ$C)')
plt.title('(b) 2 m air Temperature', pad=-20)
plt.text(1980,np.max(T2M_vec)+np.mean(T2M_vec)*0.025,'(i)',va='top',ha='left')
ax1 = fig.add_axes([0.135, 0.090/3. + 1./3., .855, 0.4/3.])
for i in range(44):
    plt.fill_between([years[i],years[i+1]],[N_T2M_minus1[i],N_T2M_minus1[i]],[1,1],color=colors[0],alpha=1)
    plt.fill_between([years[i],years[i+1]],[N_T2M_0[i],N_T2M_0[i]],[N_T2M_minus1[i],N_T2M_minus1[i]],color=colors[1],alpha=1)
    plt.fill_between([years[i],years[i+1]],[N_T2M_1[i],N_T2M_1[i]],[N_T2M_0[i],N_T2M_0[i]],color=colors[2],alpha=1)
    plt.fill_between([years[i],years[i+1]],[0,0],[N_T2M_1[i],N_T2M_1[i]],color=colors[3],alpha=1)
plt.plot([1980,2024],[0.25,0.25],'k',linewidth=0.5)
plt.plot([1980,2024],[0.5,0.5],'k',linewidth=0.5)
plt.plot([1980,2024],[0.75,0.75],'k',linewidth=0.5)
plt.xlim([1980,2024])
plt.ylim([0,1])
plt.yticks([0,0.25,0.50,0.75,1.00])
l0 = plt.fill(np.NaN, np.NaN, color=colors[0], alpha=1.)
l1 = plt.fill(np.NaN, np.NaN, color=colors[1], alpha=1.)
l2 = plt.fill(np.NaN, np.NaN, color=colors[2], alpha=1.)
l3 = plt.fill(np.NaN, np.NaN, color=colors[3], alpha=1.)
plt.xticks(years)
plt.legend([l0[0],l1[0],l2[0],l3[0]],('Z-score<-1','-1<Z-score<0','0<Z-score<1','1<Z-score'),loc='upper center',frameon=False,bbox_to_anchor=(0.5, 1.30),ncol=4,handletextpad=0.5,labelspacing=0.4,handlelength=1.,columnspacing=0.5,fontsize=9)
ax1.set_xticklabels([])
ax1.set_yticklabels(['0','25','50','75',''])
plt.ylabel('Percent\nof area')
for i in range(44):
    plt.text(1980.5+i,0,str(int((1980+i)-100*np.floor((1980+i)/100.))).zfill(2),ha='center',va='top',rotation=90,fontsize=7.5)#    plt.text(1980.5+i,0,str((1980+i)-100*np.floor((1980+i)/100.)),ha='center',va='top',rotation=90,fontsize=8)
    #plt.text(1980.5+i,0,str(1980+i),ha='center',va='top',rotation=90,fontsize=8)
#plt.savefig('T2M_timeseries_20231113.png')
plt.text(1980,1,'(ii)',va='top',ha='left')



#fig = plt.figure(4203, figsize=(7,3), dpi=300)
ax1 = fig.add_axes([0.135, 0.6/3., .855, 0.30/3.])
plt.plot(years[0:44]+0.5,VPD_vec,'k')
plt.plot([1980,2024],[VPD_vec[43],VPD_vec[43]],'k:')
plt.plot(2023.5,VPD_vec[43],'k.')
plt.xticks(years)
plt.xlim([1980,2024])
ax1.set_xticklabels([])
#plt.ylim([np.min(VPD_vec)*0.9,np.max(VPD_vec)*1.1])
plt.ylim([np.min(VPD_vec)-np.mean(VPD_vec)*0.025,np.max(VPD_vec)+np.mean(VPD_vec)*0.025])
plt.ylabel('May-Sep VPD\n(hPa)')
plt.title('(c) Vapor Pressure Deficit', pad=-20)
plt.text(1980,np.max(VPD_vec)+np.mean(VPD_vec)*0.025,'(i)',va='top',ha='left')
ax1 = fig.add_axes([0.135, 0.090/3., .855, 0.4/3.])
for i in range(44):
    plt.fill_between([years[i],years[i+1]],[N_VPD_minus1[i],N_VPD_minus1[i]],[1,1],color=colors[0],alpha=1)
    plt.fill_between([years[i],years[i+1]],[N_VPD_0[i],N_VPD_0[i]],[N_VPD_minus1[i],N_VPD_minus1[i]],color=colors[1],alpha=1)
    plt.fill_between([years[i],years[i+1]],[N_VPD_1[i],N_VPD_1[i]],[N_VPD_0[i],N_VPD_0[i]],color=colors[2],alpha=1)
    plt.fill_between([years[i],years[i+1]],[0,0],[N_VPD_1[i],N_VPD_1[i]],color=colors[3],alpha=1)
plt.plot([1980,2024],[0.25,0.25],'k',linewidth=0.5)
plt.plot([1980,2024],[0.5,0.5],'k',linewidth=0.5)
plt.plot([1980,2024],[0.75,0.75],'k',linewidth=0.5)
plt.xlim([1980,2024])
plt.ylim([0,1])
plt.yticks([0,0.25,0.50,0.75,1.00])
plt.ylabel('Percent\nof area')
l0 = plt.fill(np.NaN, np.NaN, color=colors[0], alpha=1.)
l1 = plt.fill(np.NaN, np.NaN, color=colors[1], alpha=1.)
l2 = plt.fill(np.NaN, np.NaN, color=colors[2], alpha=1.)
l3 = plt.fill(np.NaN, np.NaN, color=colors[3], alpha=1.)
plt.xticks(years)
plt.legend([l0[0],l1[0],l2[0],l3[0]],('Z-score<-1','-1<Z-score<0','0<Z-score<1','1<Z-score'),loc='upper center',frameon=False,bbox_to_anchor=(0.5, 1.30),ncol=4,handletextpad=0.5,labelspacing=0.4,handlelength=1.,columnspacing=0.5,fontsize=9)
ax1.set_xticklabels([])
ax1.set_yticklabels(['0','25','50','75',''])
for i in range(44):
    plt.text(1980.5+i,0,str(int((1980+i)-100*np.floor((1980+i)/100.))).zfill(2),ha='center',va='top',rotation=90,fontsize=7.5)#    plt.text(1980.5+i,0,str((1980+i)-100*np.floor((1980+i)/100.)),ha='center',va='top',rotation=90,fontsize=8)
    #plt.text(1980.5+i,0,str(1980+i),ha='center',va='top',rotation=90,fontsize=8)
plt.text(1980,1,'(ii)',va='top',ha='left')
plt.savefig('Figures/Byrne_etal_FigS3.png')

