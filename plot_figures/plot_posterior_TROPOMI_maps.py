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


def plot_bar(ax1,xv,bv,barr,tlab,ylab='None',xlab='None'):
    bar_width = 1.0
    rects2 = plt.bar(xv, bv, bar_width)
    plt.text(-65+2, 0.06*0.985,tlab,va='top',ha='left')
    plt.text(-65+2, 0.06*0.885,'mean = '+str(round(np.mean(barr),2)),va='top',ha='left')
    plt.text(-65+2, 0.06*0.785,'std = '+str(round(np.std(barr),2)),va='top',ha='left')
    plt.ylim([0,0.06])
    plt.xlim([-65,65])
    plt.xticks([-50,-25,0,25,50])
    # ---                                                                                                                                               
    if ylab != 'None':
        plt.ylabel(ylab)
    else:
        ax1.set_yticklabels([])
    # ---                                                                                                                                               
    if xlab != 'None':
        plt.xlabel(xlab)
    else:
        ax1.set_xticklabels([])
    # ================================================

def plot_timeseries(lon,lat,Y,Hx_GFAS_prior,Hx_GFED_prior,Hx_QFED_prior,Hx_GFAS_post,Hx_GFED_post,Hx_QFED_post,doy,figure_name):

    # ===================================================
    #
    # Makes figure of XCO timeseries over Canada
    # 
    # ===================================================

    # Data within Canada region
    II = np.where(np.logical_and(np.logical_and(lon>-170,lon<75),np.logical_and(lat>20,lat<86)))
    regY = Y[II]
    regHx_GFAS_prior = Hx_GFAS_prior[II]
    regHx_GFED_prior = Hx_GFED_prior[II]
    regHx_QFED_prior = Hx_QFED_prior[II]
    regHx_GFAS_post = Hx_GFAS_post[II]
    regHx_GFED_post = Hx_GFED_post[II]
    regHx_QFED_post = Hx_QFED_post[II]
    regdoy = doy[II]
    
    # Calculate daily average
    MEANregY = np.zeros(365)
    MEANregHx_GFAS_prior = np.zeros(365)
    MEANregHx_GFED_prior = np.zeros(365)
    MEANregHx_QFED_prior = np.zeros(365)
    MEANregHx_GFAS_post = np.zeros(365)
    MEANregHx_GFED_post = np.zeros(365)
    MEANregHx_QFED_post = np.zeros(365)
    for i in range(365):
        II = np.where(regdoy==i+1)
        if np.size(II)>0:
            MEANregY[i] = np.mean(regY[II])
            MEANregHx_GFAS_prior[i] = np.mean(regHx_GFAS_prior[II])
            MEANregHx_GFED_prior[i] = np.mean(regHx_GFED_prior[II])
            MEANregHx_QFED_prior[i] = np.mean(regHx_QFED_prior[II])
            MEANregHx_GFAS_post[i] = np.mean(regHx_GFAS_post[II])
            MEANregHx_GFED_post[i] = np.mean(regHx_GFED_post[II])
            MEANregHx_QFED_post[i] = np.mean(regHx_QFED_post[II])
            
    # make plot
    fig = plt.figure(2,figsize=(5,4),dpi=300)
    plt.plot(MEANregY,'k')
    plt.plot(MEANregHx_GFAS_prior,'r:')
    plt.plot(MEANregHx_GFED_prior,'g:')
    plt.plot(MEANregHx_QFED_prior,'b:')
    plt.plot(MEANregHx_GFAS_post,'r')
    plt.plot(MEANregHx_GFED_post,'g')
    plt.plot(MEANregHx_QFED_post,'b')
    plt.savefig('Figures/'+figure_name)
    plt.clf()
    plt.close()
    # ==========================================================================


def plot_histogram(lon,lat,Y,Hx_GFAS_prior,Hx_GFED_prior,Hx_QFED_prior,Hx_GFAS_post,Hx_GFED_post,Hx_QFED_post,doy,figure_name):

    # ===================================================
    #
    # Makes figure of Y-Hx histograms over Canada
    # 
    # ===================================================

    # Data within Canada region
    II = np.where(np.logical_and(np.logical_and(lon>-170,lon<75),np.logical_and(lat>20,lat<86)))
    regY = Y[II]
    regHx_GFAS_prior = Hx_GFAS_prior[II]
    regHx_GFED_prior = Hx_GFED_prior[II]
    regHx_QFED_prior = Hx_QFED_prior[II]
    regHx_GFAS_post = Hx_GFAS_post[II]
    regHx_GFED_post = Hx_GFED_post[II]
    regHx_QFED_post = Hx_QFED_post[II]
    regdoy = doy[II]

    # Calculate Y-Hx
    YHx_GFED_prior = regY-regHx_GFED_prior
    YHx_GFED_post = regY-regHx_GFED_post
    YHx_GFAS_prior = regY-regHx_GFAS_prior
    YHx_GFAS_post = regY-regHx_GFAS_post
    YHx_QFED_prior = regY-regHx_QFED_prior
    YHx_QFED_post = regY-regHx_QFED_post

    # Find number of occurances for different Y-Hx bins
    num_occur = np.zeros(200)
    num_occur_GFED_prior = np.zeros(200)
    num_occur_GFED_post = np.zeros(200)
    num_occur_GFAS_prior = np.zeros(200)
    num_occur_GFAS_post = np.zeros(200)
    num_occur_QFED_prior = np.zeros(200)
    num_occur_QFED_post = np.zeros(200)
    interv = np.arange(200)-100
    for i in range(200):
        #
        INDs = np.where(np.logical_and(YHx_GFED_prior>=interv[i],YHx_GFED_prior<interv[i]+1))
        num_occur_GFED_prior[i] = np.size(INDs)
        INDs = np.where(np.logical_and(YHx_GFED_post>=interv[i],YHx_GFED_post<interv[i]+1))
        num_occur_GFED_post[i] = np.size(INDs)
        #
        INDs = np.where(np.logical_and(YHx_GFAS_prior>=interv[i],YHx_GFAS_prior<interv[i]+1))
        num_occur_GFAS_prior[i] = np.size(INDs)
        INDs = np.where(np.logical_and(YHx_GFAS_post>=interv[i],YHx_GFAS_post<interv[i]+1))
        num_occur_GFAS_post[i] = np.size(INDs)
        #
        INDs = np.where(np.logical_and(YHx_QFED_prior>=interv[i],YHx_QFED_prior<interv[i]+1))
        num_occur_QFED_prior[i] = np.size(INDs)
        INDs = np.where(np.logical_and(YHx_QFED_post>=interv[i],YHx_QFED_post<interv[i]+1))
        num_occur_QFED_post[i] = np.size(INDs)
        # --------------------

    # Calculate fractional occurances
    frac_occur_GFED_prior = num_occur_GFED_prior / np.sum(num_occur_GFED_prior)
    frac_occur_GFED_post = num_occur_GFED_post / np.sum(num_occur_GFED_post)
    frac_occur_GFAS_prior = num_occur_GFAS_prior / np.sum(num_occur_GFAS_prior)
    frac_occur_GFAS_post = num_occur_GFAS_post / np.sum(num_occur_GFAS_post)
    frac_occur_QFED_prior = num_occur_QFED_prior / np.sum(num_occur_QFED_prior)
    frac_occur_QFED_post = num_occur_QFED_post / np.sum(num_occur_QFED_post)

    # Make the histogram
    fig = plt.figure(110,figsize=(5,6),dpi=300)    
    ax1 = fig.add_axes([0.13+0./2., 0.00+2./3., 0.8/2., 0.8/3.])
    plot_bar(ax1,interv,frac_occur_GFED_prior,YHx_GFED_prior,'(ai) GFED prior',ylab='Fraction obs')
    plt.title('TROPOMI')
    ax1 = fig.add_axes([0.13+0./2., 0.04+1./3., 0.8/2., 0.8/3.])
    plot_bar(ax1,interv,frac_occur_GFAS_prior,YHx_GFAS_prior,'(bi) GFAS prior',ylab='Fraction obs')
    ax1 = fig.add_axes([0.13+0./2., 0.08+0./3., 0.8/2., 0.8/3.])
    plot_bar(ax1,interv,frac_occur_QFED_prior,YHx_QFED_prior,'(ci) QFED prior',ylab='Fraction obs',xlab='obs minus model (ppb)')
    ax1 = fig.add_axes([0.08+1./2., 0.00+2./3., 0.8/2., 0.8/3.])
    plot_bar(ax1,interv,frac_occur_GFED_post,YHx_GFED_post,'(aii) GFED posterior')
    plt.title('TROPOMI')
    ax1 = fig.add_axes([0.08+1./2., 0.04+1./3., 0.8/2., 0.8/3.])
    plot_bar(ax1,interv,frac_occur_GFAS_post,YHx_GFAS_post,'(bii) GFAS posterior')
    ax1 = fig.add_axes([0.08+1./2., 0.08+0./3., 0.8/2., 0.8/3.])
    plot_bar(ax1,interv,frac_occur_QFED_post,YHx_QFED_post,'(cii) QFED posterior',xlab='obs minus model (ppb)')
    plt.savefig('Figures/'+figure_name)
    plt.clf()
    plt.close()
    # ==========================================================================


def plot_maps(lon,lat,Y,Hx_GFAS_prior,Hx_GFED_prior,Hx_QFED_prior,Hx_GFAS_post,Hx_GFED_post,Hx_QFED_post,doy,figure_name):


    # Create empty arrays
    Hx_GFAS_prior_grid = np.zeros((np.size(lat_grid),np.size(lon_grid)))*np.nan
    Hx_GFED_prior_grid = np.zeros((np.size(lat_grid),np.size(lon_grid)))*np.nan
    Hx_QFED_prior_grid = np.zeros((np.size(lat_grid),np.size(lon_grid)))*np.nan
    Hx_GFAS_post_grid = np.zeros((np.size(lat_grid),np.size(lon_grid)))*np.nan
    Hx_GFED_post_grid = np.zeros((np.size(lat_grid),np.size(lon_grid)))*np.nan
    Hx_QFED_post_grid = np.zeros((np.size(lat_grid),np.size(lon_grid)))*np.nan
    Y_grid = np.zeros((np.size(lat_grid),np.size(lon_grid)))*np.nan

    # Grid the data
    for i in range(np.size(lon_grid)):
        #
        Ilon = np.where(np.logical_and(lon >= lon_grid[i]-2.5/2., lon < lon_grid[i]+2.5/2.))
        Hx_GFAS_prior_grid_II = Hx_GFAS_prior[Ilon]
        Hx_GFED_prior_grid_II = Hx_GFED_prior[Ilon]
        Hx_QFED_prior_grid_II = Hx_QFED_prior[Ilon]
        Hx_GFAS_post_grid_II = Hx_GFAS_post[Ilon]
        Hx_GFED_post_grid_II = Hx_GFED_post[Ilon]
        Hx_QFED_post_grid_II = Hx_QFED_post[Ilon]
        Y_grid_II = Y[Ilon]
        lat_II = lat[Ilon]
        #
        if np.size(Ilon)>0:
            for j in range(np.size(lat_grid)):
                #
                Ilat = np.where(np.logical_and(lat_II >= lat_grid[j]-2./2., lat_II < lat_grid[j]+2./2.))
                Hx_GFAS_prior_grid_III = Hx_GFAS_prior_grid_II[Ilat]
                Hx_GFED_prior_grid_III = Hx_GFED_prior_grid_II[Ilat]
                Hx_QFED_prior_grid_III = Hx_QFED_prior_grid_II[Ilat]
                Hx_GFAS_post_grid_III = Hx_GFAS_post_grid_II[Ilat]
                Hx_GFED_post_grid_III = Hx_GFED_post_grid_II[Ilat]
                Hx_QFED_post_grid_III = Hx_QFED_post_grid_II[Ilat]
                Y_grid_III = Y_grid_II[Ilat]
                #
                if np.size(Ilat)>0:
                    #
                    Hx_GFAS_prior_grid[j,i] = np.mean(Hx_GFAS_prior_grid_III)
                    Hx_GFED_prior_grid[j,i] = np.mean(Hx_GFED_prior_grid_III)
                    Hx_QFED_prior_grid[j,i] = np.mean(Hx_QFED_prior_grid_III)
                    Hx_GFAS_post_grid[j,i] = np.mean(Hx_GFAS_post_grid_III)
                    Hx_GFED_post_grid[j,i] = np.mean(Hx_GFED_post_grid_III)
                    Hx_QFED_post_grid[j,i] = np.mean(Hx_QFED_post_grid_III)
                    Y_grid[j,i] = np.mean(Y_grid_III)

    # Calculate Y-Hx
    YHx_GFAS_prior_grid=Y_grid-Hx_GFAS_prior_grid
    YHx_GFED_prior_grid=Y_grid-Hx_GFED_prior_grid
    YHx_QFED_prior_grid=Y_grid-Hx_QFED_prior_grid
    YHx_GFAS_post_grid=Y_grid-Hx_GFAS_post_grid
    YHx_GFED_post_grid=Y_grid-Hx_GFED_post_grid
    YHx_QFED_post_grid=Y_grid-Hx_QFED_post_grid

    # Create the plots
    m = Basemap(projection='mill',llcrnrlat=20,urcrnrlat=86,
                llcrnrlon=-170,urcrnrlon=75,resolution='c')
    #                                                                                            
    X,Y = np.meshgrid(lon_grid[2:109]-2./2.,lat_grid[54:90]-2.5/2.)
    xx,yy=m(X,Y)
    #                                                                                            
    fig = plt.figure(1,figsize=(6*1.25,3.25*1.25),dpi=300)
    #
    ax1 = fig.add_axes([0.025/2., 2.05/3., 0.95/2., 0.85/3.])
    m.pcolormesh(xx,yy,ma.masked_invalid(YHx_GFED_prior_grid[54:90,2:109]),cmap='RdBu_r',vmin=-50,vmax=50)
    m.drawcoastlines(color='grey',linewidth=0.5)
    plt.annotate('(ai) GFED prior', xy=(0.005, 0.985), xycoords='axes fraction',va='top',ha='left')
    plt.colorbar()
    #
    ax1 = fig.add_axes([0.025/2., 1.05/3., 0.95/2., 0.85/3.])
    m.pcolormesh(xx,yy,ma.masked_invalid(YHx_GFAS_prior_grid[54:90,2:109]),cmap='RdBu_r',vmin=-50,vmax=50)
    m.drawcoastlines(color='grey',linewidth=0.5)
    plt.annotate('(bi) GFAS prior', xy=(0.005, 0.985), xycoords='axes fraction',va='top',ha='left')
    plt.colorbar()
    #
    ax1 = fig.add_axes([0.025/2., 0.05/3., 0.95/2., 0.85/3.])
    m.pcolormesh(xx,yy,ma.masked_invalid(YHx_QFED_prior_grid[54:90,2:109]),cmap='RdBu_r',vmin=-50,vmax=50)
    m.drawcoastlines(color='grey',linewidth=0.5)
    plt.annotate('(ci) QFED prior', xy=(0.005, 0.985), xycoords='axes fraction',va='top',ha='left')
    plt.colorbar()
    #
    ax1 = fig.add_axes([1.025/2., 2.05/3., 0.95/2., 0.85/3.])
    m.pcolormesh(xx,yy,ma.masked_invalid(YHx_GFED_post_grid[54:90,2:109]),cmap='RdBu_r',vmin=-50,vmax=50)
    m.drawcoastlines(color='grey',linewidth=0.5)
    plt.annotate('(aii) GFED posterior', xy=(0.005, 0.985), xycoords='axes fraction',va='top',ha='left')
    plt.colorbar()
    #
    ax1 = fig.add_axes([1.025/2., 1.05/3., 0.95/2., 0.85/3.])
    m.pcolormesh(xx,yy,ma.masked_invalid(YHx_GFAS_post_grid[54:90,2:109]),cmap='RdBu_r',vmin=-50,vmax=50)
    m.drawcoastlines(color='grey',linewidth=0.5)
    plt.annotate('(bii) GFAS posterior', xy=(0.005, 0.985), xycoords='axes fraction',va='top',ha='left')
    plt.colorbar()
    #
    ax1 = fig.add_axes([1.025/2., 0.05/3., 0.95/2., 0.85/3.])
    m.pcolormesh(xx,yy,ma.masked_invalid(YHx_QFED_post_grid[54:90,2:109]),cmap='RdBu_r',vmin=-50,vmax=50)
    m.drawcoastlines(color='grey',linewidth=0.5)
    plt.annotate('(cii) QFED posterior', xy=(0.005, 0.985), xycoords='axes fraction',va='top',ha='left')
    plt.colorbar()
    #                                                                                            
    plt.savefig('Figures/'+figure_name)
    plt.clf()
    plt.close()
    # ================================================
    


# =========
nc_file ='/nobackup/bbyrne1/MERRA2/2x2.5/2022/01/MERRA2.20220114.I3.2x25.nc4'
f=Dataset(nc_file,mode='r')
lon_grid=f.variables['lon'][:]
lat_grid=f.variables['lat'][:]
f.close()
# =========


# ============================================================================
nc_file = './data_for_figures/TROPOMI_CanadaFire_prior_YHx.nc'
f=Dataset(nc_file,mode='r')
lon=f.variables['longitude'][:]
lat=f.variables['latitude'][:]
Y=f.variables['Y_GFED'][:]
Hx_GFAS_prior=f.variables['Hx_GFAS'][:]
Hx_GFED_prior=f.variables['Hx_GFED'][:]
Hx_QFED_prior=f.variables['Hx_QFED'][:]
doy = f.variables['doy_arr'][:]
f.close()
# -------------------------------------------------------------------------
nc_file = './data_for_figures/TROPOMI_CanadaFire_posterior_YHx_3day.nc'
f=Dataset(nc_file,mode='r')
Hx_GFAS_post_3day=f.variables['Hx_GFAS'][:]
Hx_GFED_post_3day=f.variables['Hx_GFED'][:]
Hx_QFED_post_3day=f.variables['Hx_QFED'][:]
f.close()
nc_file = './data_for_figures/TROPOMI_CanadaFire_posterior_YHx_3day_rep.nc'
f=Dataset(nc_file,mode='r')
Hx_GFAS_post_3day_rep=f.variables['Hx_GFAS'][:]
Hx_GFED_post_3day_rep=f.variables['Hx_GFED'][:]
Hx_QFED_post_3day_rep=f.variables['Hx_QFED'][:]
f.close()
nc_file = './data_for_figures/TROPOMI_CanadaFire_posterior_YHx_7day.nc'
f=Dataset(nc_file,mode='r')
Hx_GFAS_post_7day=f.variables['Hx_GFAS'][:]
Hx_GFED_post_7day=f.variables['Hx_GFED'][:]
Hx_QFED_post_7day=f.variables['Hx_QFED'][:]
f.close()
nc_file = './data_for_figures/TROPOMI_CanadaFire_posterior_YHx_7day_rep.nc'
f=Dataset(nc_file,mode='r')
Hx_GFAS_post_7day_rep=f.variables['Hx_GFAS'][:]
Hx_GFED_post_7day_rep=f.variables['Hx_GFED'][:]
Hx_QFED_post_7day_rep=f.variables['Hx_QFED'][:]
f.close()
# -------------------------------------------------------------------------
plot_histogram(lon,lat,Y,Hx_GFAS_prior,Hx_GFED_prior,Hx_QFED_prior,Hx_GFAS_post_3day_rep,Hx_GFED_post_3day_rep,Hx_QFED_post_3day_rep,doy,'TROPOMI_histogram_4case_3day_rep.png')
plot_histogram(lon,lat,Y,Hx_GFAS_prior,Hx_GFED_prior,Hx_QFED_prior,Hx_GFAS_post_3day,Hx_GFED_post_3day,Hx_QFED_post_3day,doy,'TROPOMI_histogram_4case_3day.png')
plot_histogram(lon,lat,Y,Hx_GFAS_prior,Hx_GFED_prior,Hx_QFED_prior,Hx_GFAS_post_7day_rep,Hx_GFED_post_7day_rep,Hx_QFED_post_7day_rep,doy,'TROPOMI_histogram_4case_7day_rep.png')
plot_histogram(lon,lat,Y,Hx_GFAS_prior,Hx_GFED_prior,Hx_QFED_prior,Hx_GFAS_post_7day,Hx_GFED_post_7day,Hx_QFED_post_7day,doy,'TROPOMI_histogram_4case_7day.png')
# ---
Hx_GFAS_post = (Hx_GFAS_post_3day+Hx_GFAS_post_3day_rep+Hx_GFAS_post_7day+Hx_GFAS_post_7day_rep)/4.
Hx_GFED_post = (Hx_GFED_post_3day+Hx_GFED_post_3day_rep+Hx_GFED_post_7day+Hx_GFED_post_7day_rep)/4.
Hx_QFED_post = (Hx_QFED_post_3day+Hx_QFED_post_3day_rep+Hx_QFED_post_7day+Hx_QFED_post_7day_rep)/4.
# ---
plot_timeseries(lon,lat,Y,Hx_GFAS_prior,Hx_GFED_prior,Hx_QFED_prior,Hx_GFAS_post,Hx_GFED_post,Hx_QFED_post,doy,'timeseries_XCO_4case.png')
plot_histogram(lon,lat,Y,Hx_GFAS_prior,Hx_GFED_prior,Hx_QFED_prior,Hx_GFAS_post,Hx_GFED_post,Hx_QFED_post,doy,'TROPOMI_histogram_4case.png')
plot_maps(lon,lat,Y,Hx_GFAS_prior,Hx_GFED_prior,Hx_QFED_prior,Hx_GFAS_post,Hx_GFED_post,Hx_QFED_post,doy,'TROPOMI_maps_fit_4case.png')
# ============================================================================



# ============================================================================
nc_file = './data_for_figures/TROPOMI_CanadaFire_prior_YHx_3day_injh.nc'
f=Dataset(nc_file,mode='r')
Hx_GFAS_prior_3day=f.variables['Hx_GFAS'][:]
Hx_GFED_prior_3day=f.variables['Hx_GFED'][:]
Hx_QFED_prior_3day=f.variables['Hx_QFED'][:]
f.close()
nc_file = './data_for_figures/TROPOMI_CanadaFire_prior_YHx_3day_rep_injh.nc'
f=Dataset(nc_file,mode='r')
Hx_GFAS_prior_3day_rep=f.variables['Hx_GFAS'][:]
Hx_GFED_prior_3day_rep=f.variables['Hx_GFED'][:]
Hx_QFED_prior_3day_rep=f.variables['Hx_QFED'][:]
f.close()
nc_file = './data_for_figures/TROPOMI_CanadaFire_prior_YHx_7day_injh.nc'
f=Dataset(nc_file,mode='r')
Hx_GFAS_prior_7day=f.variables['Hx_GFAS'][:]
Hx_GFED_prior_7day=f.variables['Hx_GFED'][:]
Hx_QFED_prior_7day=f.variables['Hx_QFED'][:]
f.close()
nc_file = './data_for_figures/TROPOMI_CanadaFire_prior_YHx_7day_rep_injh.nc'
f=Dataset(nc_file,mode='r')
Hx_GFAS_prior_7day_rep=f.variables['Hx_GFAS'][:]
Hx_GFED_prior_7day_rep=f.variables['Hx_GFED'][:]
Hx_QFED_prior_7day_rep=f.variables['Hx_QFED'][:]
f.close()
# -------------------------------------------------------------------------
nc_file = './data_for_figures/TROPOMI_CanadaFire_posterior_YHx_3day_injh.nc'
f=Dataset(nc_file,mode='r')
Hx_GFAS_post_3day=f.variables['Hx_GFAS'][:]
Hx_GFED_post_3day=f.variables['Hx_GFED'][:]
Hx_QFED_post_3day=f.variables['Hx_QFED'][:]
f.close()
nc_file = './data_for_figures/TROPOMI_CanadaFire_posterior_YHx_3day_rep_injh.nc'
f=Dataset(nc_file,mode='r')
Hx_GFAS_post_3day_rep=f.variables['Hx_GFAS'][:]
Hx_GFED_post_3day_rep=f.variables['Hx_GFED'][:]
Hx_QFED_post_3day_rep=f.variables['Hx_QFED'][:]
f.close()
nc_file = './data_for_figures/TROPOMI_CanadaFire_posterior_YHx_7day_injh.nc'
f=Dataset(nc_file,mode='r')
Hx_GFAS_post_7day=f.variables['Hx_GFAS'][:]
Hx_GFED_post_7day=f.variables['Hx_GFED'][:]
Hx_QFED_post_7day=f.variables['Hx_QFED'][:]
f.close()
nc_file = './data_for_figures/TROPOMI_CanadaFire_posterior_YHx_7day_rep_injh.nc'
f=Dataset(nc_file,mode='r')
Hx_GFAS_post_7day_rep=f.variables['Hx_GFAS'][:]
Hx_GFED_post_7day_rep=f.variables['Hx_GFED'][:]
Hx_QFED_post_7day_rep=f.variables['Hx_QFED'][:]
f.close()
# -------------------------------------------------------------------------
plot_histogram(lon,lat,Y,Hx_GFAS_prior_3day_rep,Hx_GFED_prior_3day_rep,Hx_QFED_prior_3day_rep,Hx_GFAS_post_3day_rep,Hx_GFED_post_3day_rep,Hx_QFED_post_3day_rep,doy,'TROPOMI_histogram_4case_3day_rep_injh.png')
plot_histogram(lon,lat,Y,Hx_GFAS_prior_3day,Hx_GFED_prior_3day,Hx_QFED_prior_3day,Hx_GFAS_post_3day,Hx_GFED_post_3day,Hx_QFED_post_3day,doy,'TROPOMI_histogram_4case_3day_injh.png')
plot_histogram(lon,lat,Y,Hx_GFAS_prior_7day_rep,Hx_GFED_prior_7day_rep,Hx_QFED_prior_7day_rep,Hx_GFAS_post_7day_rep,Hx_GFED_post_7day_rep,Hx_QFED_post_7day_rep,doy,'TROPOMI_histogram_4case_7day_rep_injh.png')
plot_histogram(lon,lat,Y,Hx_GFAS_prior_7day,Hx_GFED_prior_7day,Hx_QFED_prior_7day,Hx_GFAS_post_7day,Hx_GFED_post_7day,Hx_QFED_post_7day,doy,'TROPOMI_histogram_4case_7day_injh.png')
# ---
Hx_GFAS_prior = (Hx_GFAS_prior_3day+Hx_GFAS_prior_3day_rep+Hx_GFAS_prior_7day+Hx_GFAS_prior_7day_rep)/4.
Hx_GFED_prior = (Hx_GFED_prior_3day+Hx_GFED_prior_3day_rep+Hx_GFED_prior_7day+Hx_GFED_prior_7day_rep)/4.
Hx_QFED_prior = (Hx_QFED_prior_3day+Hx_QFED_prior_3day_rep+Hx_QFED_prior_7day+Hx_QFED_prior_7day_rep)/4.
# ---
Hx_GFAS_post = (Hx_GFAS_post_3day+Hx_GFAS_post_3day_rep+Hx_GFAS_post_7day+Hx_GFAS_post_7day_rep)/4.
Hx_GFED_post = (Hx_GFED_post_3day+Hx_GFED_post_3day_rep+Hx_GFED_post_7day+Hx_GFED_post_7day_rep)/4.
Hx_QFED_post = (Hx_QFED_post_3day+Hx_QFED_post_3day_rep+Hx_QFED_post_7day+Hx_QFED_post_7day_rep)/4.
# ---
plot_timeseries(lon,lat,Y,Hx_GFAS_prior,Hx_GFED_prior,Hx_QFED_prior,Hx_GFAS_post,Hx_GFED_post,Hx_QFED_post,doy,'timeseries_XCO_4case_injh.png')
plot_histogram(lon,lat,Y,Hx_GFAS_prior,Hx_GFED_prior,Hx_QFED_prior,Hx_GFAS_post,Hx_GFED_post,Hx_QFED_post,doy,'TROPOMI_histogram_4case_injh.png')
plot_maps(lon,lat,Y,Hx_GFAS_prior,Hx_GFED_prior,Hx_QFED_prior,Hx_GFAS_post,Hx_GFED_post,Hx_QFED_post,doy,'TROPOMI_maps_fit_4case_injh.png')
# ============================================================================

