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
-------- plt_CMPI5_data.py
                                                          
Plot CMIP6 timeseries (Figure S11)
                                                           
contact: Brendan Byrne                                    
email: brendan.k.byrne@jpl.nasa.gov                       
                                                           
'''


def calc_JanSep_precip(file_name):
    #
    # =========================
    # Reads CMIP6 preciption data and calculates cumulative precip for Jan-Sep
    # =========================
    #
    # -- unit conversion --
    # kg m-2 s-1 / (997 kg m-3) = m s-1
    # m s-1 * 100 cm/m = cm s-1
    # cm s-1 (60*60*24)s/d = cm d-1
    #
    f = Dataset(file_name,'r')
    pr = f.variables['pr'][:] * (100.*60.*60.*24.) / 997.
    f.close()
    #
    n_years = int(np.shape(pr)[0]/12.)
    #
    pr_JanSep_orig = np.zeros((n_years,180,360))
    for i in range(n_years):
        pr_JanSep_orig[i,:,:] = np.mean(pr[i*12+0:i*12+9,:,:],0) * 273 # Cumulative precip [cm]
    #
    pr_JanSep = pr_JanSep_orig.copy()
    pr_JanSep[:,:,0:180] = pr_JanSep_orig[:,:,180:360]
    pr_JanSep[:,:,180:360] = pr_JanSep_orig[:,:,0:180]
    #
    return pr_JanSep

def calc_MaySep_tas(file_name):
    #
    # =========================
    # Reads CMIP6 tempererature data and calculates the May-Sep mean
    # =========================
    #
    f = Dataset(file_name,'r')
    tas = f.variables['tas'][:] - 273.15 # deg C
    f.close()
    n_years = int(np.shape(tas)[0]/12.)
    tas_MaySep_orig = np.zeros((n_years,180,360))
    for i in range(n_years):
        tas_MaySep_orig[i,:,:] = np.mean(tas[i*12+4:i*12+9,:,:],0) # mean Temp
    tas_MaySep = tas_MaySep_orig.copy()
    tas_MaySep[:,:,0:180] = tas_MaySep_orig[:,:,180:360]
    tas_MaySep[:,:,180:360] = tas_MaySep_orig[:,:,0:180]
    #
    return tas_MaySep

def make_mask(lon,lat):
    #
    # =========================
    # Creates a Canadian forest for a given grid
    # =========================
    #
    nc_out = './data_for_figures/Canada_forest_mask.nc'
    f = Dataset(nc_out,'r')
    lon_MERRA2 = f.variables['lon'][:]
    lat_MERRA2 = f.variables['lat'][:]
    Forest_mask = f.variables['mask'][:]
    f.close()
    #
    Forest_mask_CMIP = np.zeros((np.size(lat),np.size(lon)))
    for i in range(np.size(lat)):
        iii = np.argmin(np.abs(lat_MERRA2-lat[i]))
        for j in range(np.size(lon)):
            jjj = np.argmin(np.abs(lon_MERRA2-lon[j]))
            Forest_mask_CMIP[i,j] = Forest_mask[iii,jjj]
    #
    return Forest_mask_CMIP

def calc_area(lat,lon,lat_grid_size,lon_grid_size):
    #
    # =========================
    # Calculates area [m2] for given grid
    # =========================
    #
    earth_radius = 6371009 # in meters                                       
    lat_dist0 = pi * earth_radius / 180.0
    y = lon*0. + lat_grid_size * lat_dist0
    x= lat*0.
    for i in range(np.size(lat)):
        x[i]= lon_grid_size * lat_dist0 * cos(radians(lat[i]))
    area = np.zeros((np.size(x),np.size(y)))
    for i in range(np.size(y)):
        for j in range(np.size(x)):
            area[j,i] = np.abs(x[j]*y[i])
    return area

def calculate_area_weigted_forest(data,mask,area):
    #
    # =========================
    # Calculates area-weighted mean of data over given mask
    # =========================
    #
    n_years = int(np.shape(data)[0])
    II = np.where(mask==1)
    forest_data = np.zeros(n_years)
    for i in range(n_years):
        data_year = data[i,:,:]
        forest_data[i] = np.sum(data_year[II]*area[II]) / np.sum(area[II])
    return forest_data

def plot_bar(ax1,decade1,data_05,data_25,data_50,data_75,data_95):
    #
    # =========================
    # Box-and-whiskers plot given percentiles
    # =========================
    #
    ax1.fill_between([decade1-4,decade1+4],[data_25,data_25],[data_75,data_75],color='k',alpha=0.4)
    ax1.plot([decade1-4,decade1+4],[data_50,data_50],'k',linewidth=2,solid_capstyle='butt')
    ax1.plot([decade1-4,decade1+4],[data_95,data_95],'k',linewidth=1,solid_capstyle='butt')
    ax1.plot([decade1-4,decade1+4],[data_05,data_05],'k',linewidth=1,solid_capstyle='butt')
    ax1.plot([decade1,decade1],[data_05,data_25],'k',linewidth=1,solid_capstyle='butt')
    ax1.plot([decade1,decade1],[data_75,data_95],'k',linewidth=1,solid_capstyle='butt')


    
if __name__ == '__main__':

    # ================
    # projections are 2015-2100
    # historical is 1850-2015
    #
    # Read lat-lon grid
    nc_out = '/u/bbyrne1/CMIP6_data/pr_Global_ensemble_historical_r1i1p1f1_p25.nc'
    f = Dataset(nc_out,'r')
    time = f.variables['time'][:] # monthly 2015-2100 # days since 1850-01-01
    lat = f.variables['lat'][:]
    lon_orig = f.variables['lon'][:]
    f.close()
    lon = lon_orig.copy()
    lon[0:180] = lon_orig[180:360]-360.
    lon[180:360] = lon_orig[0:180]

    #Create Forest mask and area grid
    Forest_mask_CMIP = make_mask(lon,lat)
    area = calc_area(lat,lon,1.,1.)

    # =================== Read CMIP6 data =====================
    pr_JanSep_histssp245_forest_decade = np.zeros((5,25))
    pr_JanSep_histssp585_forest_decade = np.zeros((5,25))
    tas_MaySep_histssp245_forest_decade = np.zeros((5,25))
    tas_MaySep_histssp585_forest_decade = np.zeros((5,25))
    # ---
    percentiles = ['5','25','50','75','95']
    for i, p in enumerate(percentiles):
        # Read data
        pr_JanSep_historical =  calc_JanSep_precip('/u/bbyrne1/CMIP6_data/pr_Global_ensemble_historical_r1i1p1f1_p'+p+'.nc')
        pr_JanSep_ssp245 =  calc_JanSep_precip('/u/bbyrne1/CMIP6_data/pr_Global_ensemble_ssp245_r1i1p1f1_p'+p+'.nc')
        pr_JanSep_ssp585 =  calc_JanSep_precip('/u/bbyrne1/CMIP6_data/pr_Global_ensemble_ssp585_r1i1p1f1_p'+p+'.nc')
        tas_MaySep_historical =  calc_MaySep_tas('/u/bbyrne1/CMIP6_data/tas_Global_ensemble_historical_r1i1p1f1_p'+p+'.nc')
        tas_MaySep_ssp245 =  calc_MaySep_tas('/u/bbyrne1/CMIP6_data/tas_Global_ensemble_ssp245_r1i1p1f1_p'+p+'.nc')
        tas_MaySep_ssp585 =  calc_MaySep_tas('/u/bbyrne1/CMIP6_data/tas_Global_ensemble_ssp585_r1i1p1f1_p'+p+'.nc')
        # Calc forest value
        pr_JanSep_historical_forest = calculate_area_weigted_forest(pr_JanSep_historical , Forest_mask_CMIP , area)
        pr_JanSep_ssp245_forest = calculate_area_weigted_forest(pr_JanSep_ssp245 , Forest_mask_CMIP , area)
        pr_JanSep_ssp585_forest = calculate_area_weigted_forest(pr_JanSep_ssp585 , Forest_mask_CMIP , area)
        tas_MaySep_historical_forest = calculate_area_weigted_forest(tas_MaySep_historical , Forest_mask_CMIP , area)
        tas_MaySep_ssp245_forest = calculate_area_weigted_forest(tas_MaySep_ssp245 , Forest_mask_CMIP , area)
        tas_MaySep_ssp585_forest = calculate_area_weigted_forest(tas_MaySep_ssp585 , Forest_mask_CMIP , area)
        # Append to full timeseries
        pr_JanSep_histssp245_forest = np.append( pr_JanSep_historical_forest , pr_JanSep_ssp245_forest )
        pr_JanSep_histssp585_forest = np.append( pr_JanSep_historical_forest , pr_JanSep_ssp585_forest )
        tas_MaySep_histssp245_forest = np.append( tas_MaySep_historical_forest , tas_MaySep_ssp245_forest )
        tas_MaySep_histssp585_forest = np.append( tas_MaySep_historical_forest , tas_MaySep_ssp585_forest )
        # 10-year bins                                                                                                                                                             
        for j in range(25):
            pr_JanSep_histssp245_forest_decade[i,j] = np.mean(pr_JanSep_histssp245_forest[int(j*10):int((j+1)*10)])
            pr_JanSep_histssp585_forest_decade[i,j] = np.mean(pr_JanSep_histssp585_forest[int(j*10):int((j+1)*10)])
            tas_MaySep_histssp245_forest_decade[i,j] = np.mean(tas_MaySep_histssp245_forest[int(j*10):int((j+1)*10)])
            tas_MaySep_histssp585_forest_decade[i,j] = np.mean(tas_MaySep_histssp585_forest[int(j*10):int((j+1)*10)])
    # =========================================================


    # ================ Data-driven estimates  ================
    nc_file = '/u/bbyrne1/python_codes/Canada_Fires_2023/Reanalysis_timeseries_Precip_T2M_VPD.nc'
    f = Dataset(nc_file,'r')
    obs_pr_orig = f.variables['JanSep_cumulative_precip'][:]
    MERRA2_T2M = f.variables['MaySep_T2M'][:]
    f.close()
    # ---
    # Area correction from different masks
    obs_pr = obs_pr_orig * 5.464657990113554 / 5.364546686674433
    # ---
    obs_pr_decade = np.zeros(4)
    MERRA2_T2M_decade = np.zeros(4)
    for i in range(4):
        obs_pr_decade[i] = np.mean(obs_pr[int(i*10):int((i+1)*10)])
        MERRA2_T2M_decade[i] = np.mean(MERRA2_T2M[int(i*10):int((i+1)*10)])
    # ========================================================
    

    # ============ Make Figure ============
    # projections are 2015-2100
    # historical is 1850-2015
    #
    decade = np.arange(1850,2100,10)+5
    #
    fig = plt.figure(100,dpi=300)
    #
    ax1 = fig.add_axes([0.095, 0.55, 0.44, 0.44])
    for i in range(25):
        plot_bar(ax1,decade[i],
                 pr_JanSep_histssp245_forest_decade[0,i],
                 pr_JanSep_histssp245_forest_decade[1,i],
                 pr_JanSep_histssp245_forest_decade[2,i],
                 pr_JanSep_histssp245_forest_decade[3,i],
                 pr_JanSep_histssp245_forest_decade[4,i])
    plt.plot([1985-4,1985+4],[obs_pr_decade[0],obs_pr_decade[0]],'r',linewidth=2,solid_capstyle='butt')
    plt.plot([1995-4,1995+4],[obs_pr_decade[1],obs_pr_decade[1]],'r',linewidth=2,solid_capstyle='butt')
    plt.plot([2005-4,2005+4],[obs_pr_decade[2],obs_pr_decade[2]],'r',linewidth=2,solid_capstyle='butt')
    plt.plot([2015-4,2015+4],[obs_pr_decade[3],obs_pr_decade[3]],'r',linewidth=2,solid_capstyle='butt')
    plt.plot(1980+np.arange(44),obs_pr,'r',linewidth=0.75,alpha=0.6)
    plt.ylim([20,130])
    plt.xlim([1980,2100])
    plt.xticks([1980,2000,2020,2040,2060,2080])
    plt.ylabel('$\Sigma$P (cm)')
    plt.text(1980+(2100-1980)*0.015,20+(130-20)*0.975,'(a) SSP2-4.5',ha='left',va='top')
    plt.plot(2023,obs_pr[43],'ro', markersize=3)
    plt.plot([1980,2100],[obs_pr[43],obs_pr[43]],'r:')
    # ----
    ax1 = fig.add_axes([0.555, 0.55, 0.44, 0.44])
    for i in range(25):
        plot_bar(ax1,decade[i],
                 pr_JanSep_histssp585_forest_decade[0,i],
                 pr_JanSep_histssp585_forest_decade[1,i],
                 pr_JanSep_histssp585_forest_decade[2,i],
                 pr_JanSep_histssp585_forest_decade[3,i],
                 pr_JanSep_histssp585_forest_decade[4,i])
    plt.plot([1985-4,1985+4],[obs_pr_decade[0],obs_pr_decade[0]],'r',linewidth=2,solid_capstyle='butt')
    plt.plot([1995-4,1995+4],[obs_pr_decade[1],obs_pr_decade[1]],'r',linewidth=2,solid_capstyle='butt')
    plt.plot([2005-4,2005+4],[obs_pr_decade[2],obs_pr_decade[2]],'r',linewidth=2,solid_capstyle='butt')
    plt.plot([2015-4,2015+4],[obs_pr_decade[3],obs_pr_decade[3]],'r',linewidth=2,solid_capstyle='butt')
    plt.plot(1980+np.arange(44),obs_pr,'r',linewidth=0.75,alpha=0.6)
    plt.ylim([20,130])
    plt.xlim([1980,2100])
    ax1.set_yticklabels([])
    plt.xticks([1980,2000,2020,2040,2060,2080])
    plt.text(1980+(2100-1980)*0.015,20+(130-20)*0.975,'(b) SSP5-8.5',ha='left',va='top')
    plt.plot(2023,obs_pr[43],'ro', markersize=3)
    plt.plot([1980,2100],[obs_pr[43],obs_pr[43]],'r:')
    # ----
    ax1 = fig.add_axes([0.095, 0.05, 0.44, 0.44])
    for i in range(25):
        plot_bar(ax1,decade[i],
                 tas_MaySep_histssp245_forest_decade[0,i],
                 tas_MaySep_histssp245_forest_decade[1,i],
                 tas_MaySep_histssp245_forest_decade[2,i],
                 tas_MaySep_histssp245_forest_decade[3,i],
                 tas_MaySep_histssp245_forest_decade[4,i])
    plt.plot([1985-4,1985+4],[MERRA2_T2M_decade[0],MERRA2_T2M_decade[0]],'r',linewidth=2,solid_capstyle='butt')
    plt.plot([1995-4,1995+4],[MERRA2_T2M_decade[1],MERRA2_T2M_decade[1]],'r',linewidth=2,solid_capstyle='butt')
    plt.plot([2005-4,2005+4],[MERRA2_T2M_decade[2],MERRA2_T2M_decade[2]],'r',linewidth=2,solid_capstyle='butt')
    plt.plot([2015-4,2015+4],[MERRA2_T2M_decade[3],MERRA2_T2M_decade[3]],'r',linewidth=2,solid_capstyle='butt')
    plt.plot(1980+np.arange(44),MERRA2_T2M,'r',linewidth=0.75,alpha=0.6)
    plt.plot(2023,MERRA2_T2M[43],'ro', markersize=3)
    plt.plot([1980,2100],[MERRA2_T2M[43],MERRA2_T2M[43]],'r:')
    plt.ylim([6,24])
    plt.xlim([1980,2100])
    plt.ylabel('T2M ($^\circ$C)')
    plt.xticks([1980,2000,2020,2040,2060,2080])
    plt.text(1980+(2100-1980)*0.015,6+(24-6)*0.975,'(c) SSP2-4.5',ha='left',va='top')
    # ----
    ax1 = fig.add_axes([0.555, 0.05, 0.44, 0.44])
    for i in range(25):
        plot_bar(ax1,decade[i],
                 tas_MaySep_histssp585_forest_decade[0,i],
                 tas_MaySep_histssp585_forest_decade[1,i],
                 tas_MaySep_histssp585_forest_decade[2,i],
                 tas_MaySep_histssp585_forest_decade[3,i],
                 tas_MaySep_histssp585_forest_decade[4,i])
    plt.plot([1985-4,1985+4],[MERRA2_T2M_decade[0],MERRA2_T2M_decade[0]],'r',linewidth=2,solid_capstyle='butt')
    plt.plot([1995-4,1995+4],[MERRA2_T2M_decade[1],MERRA2_T2M_decade[1]],'r',linewidth=2,solid_capstyle='butt')
    plt.plot([2005-4,2005+4],[MERRA2_T2M_decade[2],MERRA2_T2M_decade[2]],'r',linewidth=2,solid_capstyle='butt')
    plt.plot([2015-4,2015+4],[MERRA2_T2M_decade[3],MERRA2_T2M_decade[3]],'r',linewidth=2,solid_capstyle='butt')
    plt.plot(1980+np.arange(44),MERRA2_T2M,'r',linewidth=0.75,alpha=0.6)
    plt.plot(2023,MERRA2_T2M[43],'ro', markersize=3)
    plt.plot([1980,2100],[MERRA2_T2M[43],MERRA2_T2M[43]],'r:')
    plt.ylim([6,24])
    plt.xlim([1980,2100])
    ax1.set_yticklabels([])
    plt.xticks([1980,2000,2020,2040,2060,2080])
    plt.text(1980+(2100-1980)*0.015,6+(24-6)*0.975,'(d) SSP5-8.5',ha='left',va='top')
    # ----
    plt.savefig('T2M_precip_timeseries.png', dpi=300)
    plt.clf()
