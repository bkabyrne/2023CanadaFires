# --- import modules ---                    
import numpy as np
import datetime
from netCDF4 import Dataset
import glob, os
from datetime import date
import numpy.ma as ma
from math import pi, cos, radians
import numpy.matlib

def read_obs(nc_file_TROPOMI,nc_file_GFED,nc_file_GFAS,nc_file_QFED,doy):

    f=Dataset(nc_file_TROPOMI,mode='r')
    longitude=f.variables['longitude'][:]
    latitude=f.variables['latitude'][:]
    f.close()

    # ===================================
    if os.path.isfile(nc_file_GFED):
        f=Dataset(nc_file_GFED,mode='r')
        Y_GFED=np.squeeze(f.variables['Y'][:]*1e9)
        Hx_GFED=np.squeeze(f.variables['HX'][:]*1e9)
        f.close()
    else:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    # ---
    if os.path.isfile(nc_file_GFAS):
        f=Dataset(nc_file_GFAS,mode='r')
        Hx_GFAS=np.squeeze(f.variables['HX'][:]*1e9)
        f.close()
    else:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    # ---
    if os.path.isfile(nc_file_QFED):
        f=Dataset(nc_file_QFED,mode='r')
        Hx_QFED=np.squeeze(f.variables['HX'][:]*1e9)
        f.close()
    else:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    # ===================================

    doy_arr = Hx_QFED*0.+doy

    # ---
    if np.size(Hx_QFED) != np.size(latitude):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        return longitude, latitude, Y_GFED, Hx_GFED, Hx_GFAS, Hx_QFED, doy_arr
    # =========================================================================


def create_timeseries_and_write(run_dirs_in,XCO2_data_dir,file_out,month_arr,day_arr):
    
    # ==========================================
    #
    # Function that reads TROPOMI XCO2 co-samples
    # for each day during Apr-Sep simulation and
    # writes it out to a netcdf file
    #
    # ---- inputs ----
    #  - run_dirs_in: directories for the QFED, GFED, and GFAS simulations
    #  - XCO2_data_dir: direcotry for TROPOMI data for given experiment
    #  - file_out: name of output file
    # ----------------
    #
    # ==========================================

    # ---
    for iit in range(273-90-1):
        # ---
        ii = iit + 90
        # ---
        doy = ii
        # ---
        nc_file_QFED = run_dirs_in[0]+'/OBSF/'+XCO2_data_dir+'/2023/'+str(int(month_arr[ii])).zfill(2)+'/'+str(int(day_arr[ii])).zfill(2)+'.nc'
        nc_file_GFED = run_dirs_in[1]+'/OBSF/'+XCO2_data_dir+'/2023/'+str(int(month_arr[ii])).zfill(2)+'/'+str(int(day_arr[ii])).zfill(2)+'.nc'
        nc_file_GFAS = run_dirs_in[2]+'/OBSF/'+XCO2_data_dir+'/2023/'+str(int(month_arr[ii])).zfill(2)+'/'+str(int(day_arr[ii])).zfill(2)+'.nc'
        # ---
        nc_file_TROPOMI = '/nobackup/bbyrne1/'+XCO2_data_dir+'/2023/'+str(int(month_arr[ii])).zfill(2)+'/'+str(int(day_arr[ii])).zfill(2)+'.nc'
        # ---
        print(nc_file_GFED)
        # ---
        longitudet, latitudet, Y_GFEDt, Hx_GFEDt, Hx_GFASt, Hx_QFEDt, doy_arrt = read_obs(nc_file_TROPOMI,nc_file_GFED,nc_file_GFAS,nc_file_QFED,doy)
        if 'longitude' in locals():
            longitude = np.append(longitude,longitudet)
            latitude = np.append(latitude,latitudet)
            Y_GFED = np.append(Y_GFED,Y_GFEDt)
            Hx_GFED = np.append(Hx_GFED,Hx_GFEDt)
            Hx_GFAS = np.append(Hx_GFAS,Hx_GFASt)
            Hx_QFED = np.append(Hx_QFED,Hx_QFEDt)
            doy_arr = np.append(doy_arr,doy_arrt)
        else:
            longitude = longitudet.copy()
            latitude = latitudet.copy()
            Y_GFED = Y_GFEDt.copy()
            Hx_GFED = Hx_GFEDt.copy()
            Hx_GFAS = Hx_GFASt.copy()
            Hx_QFED = Hx_QFEDt.copy()
            doy_arr = doy_arrt.copy()
    # ---
    print(file_out)
    dataset = Dataset(file_out,'w')
    nSamples = dataset.createDimension('nSamples',np.size(longitude))
    longitudes = dataset.createVariable('longitude', np.float64, ('nSamples',))
    longitudes[:]=longitude
    latitudes = dataset.createVariable('latitude', np.float64, ('nSamples',))
    latitudes[:]=latitude
    Y_GFEDs = dataset.createVariable('Y_GFED', np.float64, ('nSamples',))
    Y_GFEDs[:]=Y_GFED
    Hx_GFEDs = dataset.createVariable('Hx_GFED', np.float64, ('nSamples',))
    Hx_GFEDs[:]=Hx_GFED
    Hx_GFASs = dataset.createVariable('Hx_GFAS', np.float64, ('nSamples',))
    Hx_GFASs[:]=Hx_GFAS
    Hx_QFEDs = dataset.createVariable('Hx_QFED', np.float64, ('nSamples',))
    Hx_QFEDs[:]=Hx_QFED
    doy_arrs = dataset.createVariable('doy_arr', np.float64, ('nSamples',))
    doy_arrs[:]=doy_arr
    dataset.close()
    # =========================================================================

    
if __name__ == "__main__":
    
    # =======================================
    # Set-up time arrays
    days_in_month = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
    days_in_year = 365
    #
    n=0
    month_arr = np.zeros(days_in_year)
    day_arr = np.zeros(days_in_year)
    for i in range(12):
        for j in range(days_in_month[i]):
            month_arr[n] = int(i + 1)
            day_arr[n] = int(j + 1)
            n=n+1
    # =======================================

    base_dir = '/u/bbyrne1/python_codes/Canada_Fires_2023/Byrne_etal_codes/plot_figures/data_for_figures/'
    # =======================================
    run_dirs_in = ['/nobackup/bbyrne1/GHGF-CMS-3day-COinv-2023/Run_COinv_QFED_2023',
                   '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-2023/Run_COinv_GFED_2023',
                   '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-2023/Run_COinv_GFAS_2023']
    XCO2_data_dir = 'TROPOMI_XCO_2x25'
    file_out = base_dir+'TROPOMI_CanadaFire_posterior_YHx_3day.nc'
    create_timeseries_and_write(run_dirs_in,XCO2_data_dir,file_out,month_arr,day_arr)
    # ---------------------------------------
    run_dirs_in = ['/nobackup/bbyrne1/GHGF-CMS-3day-COinv-2023/Run_COinv_rep_QFED_2023',
                   '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-2023/Run_COinv_rep_GFED_2023',
                   '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-2023/Run_COinv_rep_GFAS_2023']
    XCO2_data_dir = 'TROPOMIrep_XCO_2x25'
    file_out = base_dir+'TROPOMI_CanadaFire_posterior_YHx_3day_rep.nc'
    create_timeseries_and_write(run_dirs_in,XCO2_data_dir,file_out,month_arr,day_arr)
    # ---------------------------------------
    run_dirs_in = ['/nobackup/bbyrne1/GHGF-CMS-7day-COinv-2023/Run_COinv_QFED_2023',
                   '/nobackup/bbyrne1/GHGF-CMS-7day-COinv-2023/Run_COinv_GFED_2023',
                   '/nobackup/bbyrne1/GHGF-CMS-7day-COinv-2023/Run_COinv_GFAS_2023']
    XCO2_data_dir = 'TROPOMI_XCO_2x25'
    file_out = base_dir+'TROPOMI_CanadaFire_posterior_YHx_7day.nc'
    create_timeseries_and_write(run_dirs_in,XCO2_data_dir,file_out,month_arr,day_arr)
    # ---------------------------------------
    run_dirs_in = ['/nobackup/bbyrne1/GHGF-CMS-7day-COinv-2023/Run_COinv_rep_QFED_2023',
                   '/nobackup/bbyrne1/GHGF-CMS-7day-COinv-2023/Run_COinv_rep_GFED_2023',
                   '/nobackup/bbyrne1/GHGF-CMS-7day-COinv-2023/Run_COinv_rep_GFAS_2023']
    XCO2_data_dir = 'TROPOMIrep_XCO_2x25'
    file_out = base_dir+'TROPOMI_CanadaFire_posterior_YHx_7day_rep.nc'
    create_timeseries_and_write(run_dirs_in,XCO2_data_dir,file_out,month_arr,day_arr)
    # =======================================
    

    # =======================================
    run_dirs_in = ['/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/QFED_post_3day',
                   '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/GFED_post_3day',
                   '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/GFAS_post_3day']
    XCO2_data_dir = 'TROPOMI_XCO_2x25'
    file_out = base_dir+'TROPOMI_CanadaFire_posterior_YHx_3day_injh.nc'
    create_timeseries_and_write(run_dirs_in,XCO2_data_dir,file_out,month_arr,day_arr)
    # ---------------------------------------
    run_dirs_in = ['/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/QFED_rep_post_3day',
                   '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/GFED_rep_post_3day',
                   '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/GFAS_rep_post_3day']
    XCO2_data_dir = 'TROPOMI_XCO_2x25'
    file_out = base_dir+'TROPOMI_CanadaFire_posterior_YHx_3day_rep_injh.nc'
    create_timeseries_and_write(run_dirs_in,XCO2_data_dir,file_out,month_arr,day_arr)
    # ---------------------------------------
    run_dirs_in = ['/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/QFED_post_7day',
                   '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/GFED_post_7day',
                   '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/GFAS_post_7day']
    XCO2_data_dir = 'TROPOMI_XCO_2x25'
    file_out = base_dir+'TROPOMI_CanadaFire_posterior_YHx_7day_injh.nc'
    create_timeseries_and_write(run_dirs_in,XCO2_data_dir,file_out,month_arr,day_arr)
    # ---------------------------------------
    run_dirs_in = ['/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/QFED_rep_post_7day',
                   '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/GFED_rep_post_7day',
                   '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/GFAS_rep_post_7day']
    XCO2_data_dir = 'TROPOMI_XCO_2x25'
    file_out = base_dir+'TROPOMI_CanadaFire_posterior_YHx_7day_rep_injh.nc'
    create_timeseries_and_write(run_dirs_in,XCO2_data_dir,file_out,month_arr,day_arr)
    # =======================================

    
    # =======================================
    run_dirs_in = ['/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/QFED_prior_3day',
                   '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/GFED_prior_3day',
                   '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/GFAS_prior_3day']
    XCO2_data_dir = 'TROPOMI_XCO_2x25'
    file_out = base_dir+'TROPOMI_CanadaFire_prior_YHx_3day_injh.nc'
    create_timeseries_and_write(run_dirs_in,XCO2_data_dir,file_out,month_arr,day_arr)
    # ---------------------------------------
    run_dirs_in = ['/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/QFED_rep_prior_3day',
                   '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/GFED_rep_prior_3day',
                   '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/GFAS_rep_prior_3day']
    XCO2_data_dir = 'TROPOMI_XCO_2x25'
    file_out = base_dir+'TROPOMI_CanadaFire_prior_YHx_3day_rep_injh.nc'
    create_timeseries_and_write(run_dirs_in,XCO2_data_dir,file_out,month_arr,day_arr)
    # ---------------------------------------
    run_dirs_in = ['/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/QFED_prior_7day',
                   '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/GFED_prior_7day',
                   '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/GFAS_prior_7day']
    XCO2_data_dir = 'TROPOMI_XCO_2x25'
    file_out = base_dir+'TROPOMI_CanadaFire_prior_YHx_7day_injh.nc'
    create_timeseries_and_write(run_dirs_in,XCO2_data_dir,file_out,month_arr,day_arr)
    # ---------------------------------------
    run_dirs_in = ['/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/QFED_rep_prior_7day',
                   '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/GFED_rep_prior_7day',
                   '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/GFAS_rep_prior_7day']
    XCO2_data_dir = 'TROPOMI_XCO_2x25'
    file_out = base_dir+'TROPOMI_CanadaFire_prior_YHx_7day_rep_injh.nc'
    create_timeseries_and_write(run_dirs_in,XCO2_data_dir,file_out,month_arr,day_arr)
    # =======================================

