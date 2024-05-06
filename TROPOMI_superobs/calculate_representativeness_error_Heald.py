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
------ calculate_representativeness_error_Heald.py

 This files estimates observation error based on the method of
 Heald et al. (https://doi.org/10.1029/2004JD005185). This is
 called the relative residual error (RRE) method and estimates
 the observational error to be the standard deviation of Y-Hx
 using prior fluxes.

 This file uses the prior simulated TROPOMI obs created in write_daily_YHx_prior.py

'''


# ---------------------------
class parameter_obj:
    #
    # ---------
    # Class for parameters used to generate representativeness errors
    # ---------
    #
    def __init__(self):
        self.inventory = None
        self.lon_range = None
        self.lat_range = None
        self.doy_range = None
        self.lat_grid = None
        self.lon_grid = None
        self.doy = None
# ---------------------------


# ---------------------------
class TROPOMI_cosamples:
    #
    # ---------
    # Class for TROPOMI co-samples
    # ---------
    #
    def __init__(self, year):
        self.year = year
        self.lon = None
        self.lat = None
        self.xCOunc = None
        self.Y = None
        self.Hx_GFED = None
        self.Hx_GFAS = None
        self.Hx_QFED = None
        self.doy = None
        self.read_arrays()
    #
    def read_arrays(self):
        try:
            with Dataset('TROPOMI_prior_YHx_'+str(self.year).zfill(4)+'.nc', 'r') as f:
                self.lon = f.variables['longitude'][:]
                self.lat = f.variables['latitude'][:]
                self.xCOunc = f.variables['xCO_uncertainty'][:]
                self.Y = f.variables['Y_GFED'][:]
                self.Hx_GFED = f.variables['Hx_GFED'][:]
                self.Hx_GFAS = f.variables['Hx_GFAS'][:]
                self.Hx_QFED = f.variables['Hx_QFED'][:]
                self.doy = f.variables['doy_arr'][:]
        except Exception as e:
            print(f"An error occurred while reading NetCDF file: {e}")
# ---------------------------


# ---------------------------
def calculate_error(TROP,p,lon_center,lat_center,doy_center):
    #
    # -----------------------------------------------------------------------------------------------------------------
    #
    # This function cancluates (Y-Hx)-mean(Y-Hx) within a spatiotemporal domain for a given set of TROPOMI co-samples
    #
    # --Inputs--
    #             TROP - object containing the TROPOMI co-samples
    #                p - object containing parameters for analysys (spatial-temporal domain)
    #       lon_center - longitude center of spatial domain
    #       lat_center - latitude center of spatial domain
    #       doy_center - day of year center of spatial domain
    #
    # --Outputs--
    #  YHx_GFED_box_nm - vector of (Y-Hx)-mean(Y-Hx) co-samples withing the spatial domain for GFED simulation
    #  YHx_GFAS_box_nm - vector of (Y-Hx)-mean(Y-Hx) co-samples withing the spatial domain for GFAS simulation
    #  YHx_QFED_box_nm - vector of (Y-Hx)-mean(Y-Hx) co-samples withing the spatial domain for QFED simulation
    #       xCOunc_box - vector of TROPOMI uncertainties for co-samples withing the spatial domain for QFED simulation
    #
    # ------------------------------------------------------------------------------------------------------------------
    #
    # -----
    # Find TROPOMI obs within a longitude range
    II = np.where( np.logical_and( TROP.lon.data>=lon_center-p.lon_range/2. , TROP.lon.data<=lon_center+p.lon_range/2. ) )
    lat_lonB = TROP.lat[II].data
    xCOunc_lonB = TROP.xCOunc[II].data
    Y_lonB = TROP.Y[II].data
    Hx_GFED_lonB = TROP.Hx_GFED[II].data
    Hx_GFAS_lonB = TROP.Hx_GFAS[II].data
    Hx_QFED_lonB = TROP.Hx_QFED[II].data
    doy_lonB = TROP.doy[II].data
    # -----    
    # Find TROPOMI obs within a latitude range
    JJ = np.where( np.logical_and( lat_lonB>=lat_center-p.lat_range/2. , lat_lonB<=lat_center+p.lat_range/2. ) )
    xCOunc_lonB_latB = xCOunc_lonB[JJ]
    Y_lonB_latB = Y_lonB[JJ]
    Hx_GFED_lonB_latB = Hx_GFED_lonB[JJ]
    Hx_GFAS_lonB_latB = Hx_GFAS_lonB[JJ]
    Hx_QFED_lonB_latB = Hx_QFED_lonB[JJ]
    doy_lonB_latB = doy_lonB[JJ]
    # -----
    # Find TROPOMI obs within a day of year range range
    if doy_center<p.doy_range/2.:
        LLall = where( np.logical_or( doy_lonB_latB>=doy_center+365-p.doy_range/2. , doy_lonB_latB<=doy_center+p.doy_range/2. ) )
    elif doy_center>365-p.doy_range/2.: # 
        LLall = where( np.logical_or( doy_lonB_latB>=doy_center+365-p.doy_range/2. , doy_lonB_latB<=doy_center+p.doy_range/2. ) )
    else:
        LLall = where( np.logical_and( doy_lonB_latB>=doy_center-p.doy_range/2. , doy_lonB_latB<=doy_center+p.doy_range/2. ) )
    # 
    xCOunc_box = xCOunc_lonB_latB[LLall]
    Y_box = Y_lonB_latB[LLall]
    Hx_GFED_box = Hx_GFED_lonB_latB[LLall]
    Hx_GFAS_box = Hx_GFAS_lonB_latB[LLall]
    Hx_QFED_box = Hx_QFED_lonB_latB[LLall]
    # -----
    if np.size(Y_box)>0:
        # -----
        # calculate Y - Hx for obs within spatiotemporal domain
        YHx_GFED_box = Y_box - Hx_GFED_box
        YHx_GFAS_box = Y_box - Hx_GFAS_box
        YHx_QFED_box = Y_box - Hx_QFED_box
        # -----
        # Remove mean within domain
        YHx_GFED_box_nm = YHx_GFED_box - np.median(YHx_GFED_box)
        YHx_GFAS_box_nm = YHx_GFAS_box - np.median(YHx_GFAS_box)
        YHx_QFED_box_nm = YHx_QFED_box - np.median(YHx_QFED_box)
        ## -----
        #
        # Return estimates
        return YHx_GFED_box_nm, YHx_GFAS_box_nm, YHx_QFED_box_nm, xCOunc_box
    else:
        return np.nan, np.nan, np.nan, np.nan
# ---------------------------


# ---------------------------
def map_to_run_grid(params,doy_grid_out,lat_grid_out,lon_grid_out):

    # ##################################################
    #
    # Representativeness errors are calculated over a coarsened
    # time/space grid to speed up the calculation. This function
    # maps this coarse grid to the output grid
    #
    # ##################################################
    
    # Coarse grid bounds 
    lon_grid_bounds = np.concatenate(([-180], (params.lon_grid[:-1] + params.lon_grid[1:]) / 2, [180]))
    lat_grid_bounds = np.concatenate(([-90], (params.lat_grid[:-1] + params.lat_grid[1:]) / 2, [90]))
    doy_grid_bounds = np.concatenate(([90], (params.doy[:-1] + params.doy[1:]) / 2, [273]))
    #
    # Map from coarse grid back to output grid
    full_grid_total_error = np.zeros((np.size(doy_grid_out),np.size(lat_grid_out),np.size(lon_grid_out)))
    full_grid_retrieval_error = np.zeros((np.size(doy_grid_out),np.size(lat_grid_out),np.size(lon_grid_out)))
    for lonI in range(np.size(lon_grid_bounds)-1):
        II = np.where(np.logical_and(lon_grid_out>=lon_grid_bounds[lonI],lon_grid_out<=lon_grid_bounds[lonI+1]))
        for latI in range(np.size(lat_grid_bounds)-1):
            JJ = np.where(np.logical_and(lat_grid_out>=lat_grid_bounds[latI],lat_grid_out<=lat_grid_bounds[latI+1]))
            for doyI in range(np.size(doy_grid_bounds)-1):
                KK = np.where(np.logical_and(doy_grid_out>=doy_grid_bounds[doyI],doy_grid_out<=doy_grid_bounds[doyI+1]))
                full_grid_total_error[KK[0][0]:KK[0][-1]+1, JJ[0][0]:JJ[0][-1]+1, II[0][0]:II[0][-1]+1] = total_error[doyI,latI,lonI]
                full_grid_retrieval_error[KK[0][0]:KK[0][-1]+1, JJ[0][0]:JJ[0][-1]+1, II[0][0]:II[0][-1]+1] = retrieval_error[doyI,latI,lonI]*1e9
    #
    # Smooth errors over time/space
    smooth_total_error = full_grid_total_error.copy()
    smooth_retrieval_error = full_grid_retrieval_error.copy()
    for i in range(np.size(lon_grid_out)-7):
        for j in range(np.size(lat_grid_out)-7):
            for k in range(np.size(doy_grid_out)-15):
                smooth_total_error[k+7,j+3,i+3] = np.mean(full_grid_total_error[k:k+15,j:j+7,i:i+7])
                smooth_retrieval_error[k+7,j+3,i+3] = np.mean(full_grid_retrieval_error[k:k+15,j:j+7,i:i+7])
    #
    return smooth_total_error, smooth_retrieval_error
# ---------------------------

# ============================================================================

if __name__ == "__main__":

    # -- Parameters for representativeness errors --
    params = parameter_obj()
    params.lon_range = 30 # number of lon degrees to include in range [lon_center - lon_range/2 , lon_center + lon_range/2]
    params.lat_range = 30 # [lat_center - lat_range/2 , lat_center + lat_range/2]
    params.doy_range = 30 # [doy_center - doy_range/2 , doy_center + doy_range/2]
    params.lon_grid = np.arange(35)*10-170
    params.lat_grid = np.arange(17)*10-80
    params.doy = np.arange((27-9)*2)*5+90
    
    # -- Read in TROPOMI data --
    TROPOMI_data_2019 = TROPOMI_cosamples(2019)
    TROPOMI_data_2020 = TROPOMI_cosamples(2020)
    TROPOMI_data_2021 = TROPOMI_cosamples(2021)
    TROPOMI_data_2022 = TROPOMI_cosamples(2022)
    TROPOMI_data_2023 = TROPOMI_cosamples(2023)

    # -- Loop over grids and calculate the representativeness errors
    total_error = np.zeros((np.size(params.doy),np.size(params.lat_grid),np.size(params.lon_grid)))
    retrieval_error = np.zeros((np.size(params.doy),np.size(params.lat_grid),np.size(params.lon_grid)))
    for lon_index, lon_center in enumerate(params.lon_grid):
        print('lon_center')
        print(lon_center)
        for lat_index, lat_center in enumerate(params.lat_grid):
            for doy_index, doy_center in enumerate(params.doy):
                # 2019 co-samples
                YHx_GFED_box_nm_2019, YHx_GFAS_box_nm_2019, YHx_QFED_box_nm_2019, xCOunc_box_2019 = calculate_error(TROPOMI_data_2019,params,lon_center,lat_center,doy_center)
                YHx_box_nm_2019 = np.append(np.append(YHx_GFED_box_nm_2019,YHx_GFAS_box_nm_2019),YHx_QFED_box_nm_2019)
                # 2020 co-samples
                YHx_GFED_box_nm_2020, YHx_GFAS_box_nm_2020, YHx_QFED_box_nm_2020, xCOunc_box_2020 = calculate_error(TROPOMI_data_2020,params,lon_center,lat_center,doy_center)
                YHx_box_nm_2020 = np.append(np.append(YHx_GFED_box_nm_2020,YHx_GFAS_box_nm_2020),YHx_QFED_box_nm_2020)
                # 2021 co-samples
                YHx_GFED_box_nm_2021, YHx_GFAS_box_nm_2021, YHx_QFED_box_nm_2021, xCOunc_box_2021 = calculate_error(TROPOMI_data_2021,params,lon_center,lat_center,doy_center)
                YHx_box_nm_2021 = np.append(np.append(YHx_GFED_box_nm_2021,YHx_GFAS_box_nm_2021),YHx_QFED_box_nm_2021)
                # 2022 co-samples
                YHx_GFED_box_nm_2022, YHx_GFAS_box_nm_2022, YHx_QFED_box_nm_2022, xCOunc_box_2022 = calculate_error(TROPOMI_data_2022,params,lon_center,lat_center,doy_center)
                YHx_box_nm_2022 = np.append(np.append(YHx_GFED_box_nm_2022,YHx_GFAS_box_nm_2022),YHx_QFED_box_nm_2022)
                # 2023 co-samples
                YHx_GFED_box_nm_2023, YHx_GFAS_box_nm_2023, YHx_QFED_box_nm_2023, xCOunc_box_2023 = calculate_error(TROPOMI_data_2023,params,lon_center,lat_center,doy_center)
                YHx_box_nm_2023 = np.append(np.append(YHx_GFED_box_nm_2023,YHx_GFAS_box_nm_2023),YHx_QFED_box_nm_2023)

                # Append all of the data
                YHx_box_nm_all_temp = np.append(np.append(np.append(np.append(YHx_box_nm_2019,YHx_box_nm_2020),YHx_box_nm_2021),YHx_box_nm_2022),YHx_box_nm_2023)
                YHx_box_nm_all = YHx_box_nm_all_temp[np.where(np.isfinite(YHx_box_nm_all_temp))]
                II = np.where(np.isfinite(YHx_box_nm_all))
                if np.size(YHx_box_nm_all[II])>0:
                    total_error[doy_index,lat_index,lon_index] = (np.percentile(YHx_box_nm_all[II],75)-np.percentile(YHx_box_nm_all[II],25))/1.35

                xCOunc_box_all_temp = np.append(np.append(np.append(np.append(xCOunc_box_2019,xCOunc_box_2020),xCOunc_box_2021),xCOunc_box_2022),xCOunc_box_2023)
                xCOunc_box_all = xCOunc_box_all_temp[np.where(np.isfinite(xCOunc_box_all_temp))]
                II = np.where(np.isfinite(xCOunc_box_all))
                if np.size(xCOunc_box_all[II])>0:
                    retrieval_error[doy_index,lat_index,lon_index] = (np.percentile(xCOunc_box_all[II],75)-np.percentile(xCOunc_box_all[II],25))/1.35




    # -- Read 2x25 lat/lon grid --
    nc_out = '/nobackup/bbyrne1/MERRA2/2x2.5/2023/05/MERRA2.20230503.I3.2x25.nc4'
    f = Dataset(nc_out,'r')
    lat_grid_2x25 = f.variables['lat'][:]
    lon_grid_2x25 = f.variables['lon'][:]
    f.close()
    doy_grid_2x25 = np.arange(273-90-1)+90 # grid of DOY centers (Only consider Apr 1 to Sep 30)
    
    # Map error array to output grid
    smooth_total_error, smooth_retrieval_error = map_to_run_grid(params,doy_grid_2x25,lat_grid_2x25,lon_grid_2x25)
    
                
    # Write data
    file_out = 'Daily_unc_Heald.nc'
    print(file_out)
    dataset = Dataset(file_out,'w')
    days = dataset.createDimension('day',np.size(doy_grid_2x25))
    lats = dataset.createDimension('lat',np.size(lat_grid_2x25))
    lons = dataset.createDimension('lon',np.size(lon_grid_2x25))
    longitudes = dataset.createVariable('longitude', np.float64, ('lon',))
    longitudes[:] = lon_grid_2x25
    latitudes = dataset.createVariable('latitude', np.float64, ('lat',))
    latitudes[:] = lat_grid_2x25
    doys = dataset.createVariable('DOY', np.float64, ('day',))
    doys[:] = doy_grid_2x25
    unc_Healds = dataset.createVariable('unc_Heald', np.float64, ('day','lat','lon'))
    unc_Healds[:,:,:] = smooth_total_error
    unc_obss = dataset.createVariable('unc_obs', np.float64, ('day','lat','lon'))
    unc_obss[:,:,:] = smooth_retrieval_error
    dataset.close()
    

    # Plots to check output
    for iday in range(np.size(doy_grid_2x25)):
        print(iday)
        fig = plt.figure(3,figsize=(24, 8), dpi=80)
        #x
        m = Basemap(projection='robin',lon_0=0,resolution='c')
        xx, yy = meshgrid(lon_grid_2x25, lat_grid_2x25)
        #
        ax1 = fig.add_axes([0.0+0./3., 0.02, 0.9/3., 0.9/1.])
        m.pcolormesh(xx,yy,ma.masked_invalid(smooth_total_error[iday,:,:]),cmap='CMRmap_r',vmin=0,vmax=35,latlon=True)
        m.drawcoastlines()
        m.colorbar()
        plt.title('std(Y-Hx)')
        #
        ax1 = fig.add_axes([0.0+1./3., 0.02, 0.9/3., 0.9/1.])
        m.pcolormesh(xx,yy,ma.masked_invalid(smooth_retrieval_error[iday,:,:]),cmap='CMRmap_r',vmin=0,vmax=35,latlon=True)
        m.drawcoastlines()
        m.colorbar()
        plt.title('retrieval unc')
        #
        ax1 = fig.add_axes([0.0+2./3., 0.02, 0.9/3., 0.9/1.])
        m.pcolormesh(xx,yy,ma.masked_invalid((smooth_retrieval_error[iday,:,:])/smooth_total_error[iday,:,:]),cmap='CMRmap_r',vmin=0,vmax=1)
        m.drawcoastlines()
        m.colorbar()
        plt.title('retrieval / std')
        #
        plt.savefig('uncertainty_Heald/uncertainty_Heald_slide_'+str(iday+1).zfill(2)+'.jpg')
        plt.clf()
        plt.close()


