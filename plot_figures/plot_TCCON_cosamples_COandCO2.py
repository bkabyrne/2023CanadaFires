# --- import modules ---
from mpl_toolkits.basemap import Basemap, cm
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import glob, os 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import interpolate

#         
# *******************************************************     
# -------- plot_TCCON_cosamples_COandCO2.py
#                                                             
# This code processes data and plots Figures S8 and S9                
#                                                             
# contact: Brendan Byrne                                      
# email: brendan.k.byrne@jpl.nasa.gov                         
#                                                             
# *******************************************************
#

def read_TCCON(dir_name,site,scale):
    
    if 'HX_all' in locals():
        del Y_all
        del HX_all
        del doy_all

    days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 29, 31, 30, 31])
    doy_i = 0
    for mtht in range(5):
        mth=mtht+4
        for ddd in range(days_in_month[mth]):
            #
            nc_file = '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-2023/'+dir_name+'/OBSF/'+site+'/2023/'+str(mth+1).zfill(2)+'/'+str(ddd+1).zfill(2)+'.nc'
            #
            if os.path.exists(nc_file):
                #
                f=Dataset(nc_file,mode='r')
                lon=f.variables['longitude'][:]
                lat=f.variables['latitude'][:]
                Y=f.variables['Y'][:] * scale
                HX=f.variables['HX'][:] * scale
                f.close()
                #
                if 'HX_all' in locals():
                    Y_all = np.append(Y_all,Y)
                    HX_all = np.append(HX_all,HX)
                    doy_all = np.append(doy_all,Y*0.+doy_i)
                else:
                    Y_all = Y
                    HX_all = HX
                    doy_all = Y*0.+doy_i
                #
            doy_i = doy_i + 1

    return Y_all, HX_all, doy_all

def read_TCCON_obs(site):
    
    if 'time_all' in locals():
        del time_all

    days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 29, 31, 30, 31])
    doy_i = 31+28+31+30
    for mtht in range(5):
        mth=mtht+4
        for ddd in range(days_in_month[mth]):
            #
            nc_file = '/nobackup/bbyrne1/'+site+'/2023/'+str(mth+1).zfill(2)+'/'+str(ddd+1).zfill(2)+'.nc'
            print(nc_file)
            #
            if os.path.exists(nc_file):
                #
                f=Dataset(nc_file,mode='r')
                time=f.variables['time'][:]
                f.close()
                #
                time_year = doy_i + time/24.
                #
                if 'time_all' in locals():
                    time_all = np.append(time_all,time_year)
                else:
                    time_all = time_year
                #
            doy_i = doy_i + 1

    return time_all

time_vec = read_TCCON_obs('TCCON_PA_GGG2020_XCO')
# ====================================================
Y_Post_QFED_PA, HX_Post_QFED_PA, doy_Post_QFED_PA = read_TCCON('Post_COinv_QFED_AprSep','TCCON_PA_GGG2020_XCO',1e9)
Y_Post_GFAS_PA, HX_Post_GFAS_PA, doy_Post_GFAS_PA = read_TCCON('Post_COinv_GFAS_AprSep','TCCON_PA_GGG2020_XCO',1e9)
Y_Post_GFED_PA, HX_Post_GFED_PA, doy_Post_GFED_PA = read_TCCON('Post_COinv_GFED_AprSep','TCCON_PA_GGG2020_XCO',1e9)
# ====================================================
Y_Post_QFED_PA_CO2, HX_Post_QFED_PA_CO2, doy_Post_QFED_PA = read_TCCON('FWD-CO2-QFED-Post','TCCON_PA_GGG2020_XCO2',1e6)
Y_Post_GFED_PA_CO2, HX_Post_GFED_PA_CO2, doy_Post_GFED_PA = read_TCCON('FWD-CO2-GFED-Post','TCCON_PA_GGG2020_XCO2',1e6)
Y_Post_GFAS_PA_CO2, HX_Post_GFAS_PA_CO2, doy_Post_GFAS_PA = read_TCCON('FWD-CO2-GFAS-Post','TCCON_PA_GGG2020_XCO2',1e6)
Y_Post_noFlux_PA_CO2, HX_Post_noFlux_PA_CO2, doy_Post_noFlux_PA = read_TCCON('FWD-CO2-noFlux','TCCON_PA_GGG2020_XCO2',1e6)
Y_Post_Bckd_PA_CO2, HX_Post_Bckd_PA_CO2, doy_Post_Bckd_PA = read_TCCON('FWD-CO2-Bckd','TCCON_PA_GGG2020_XCO2',1e6)

# Isolate the Biomass burning signal
HXminusBckd_QFED_PA_CO2 = HX_Post_QFED_PA_CO2 - HX_Post_noFlux_PA_CO2
HXminusBckd_GFED_PA_CO2 = HX_Post_GFED_PA_CO2 - HX_Post_noFlux_PA_CO2
HXminusBckd_GFAS_PA_CO2 = HX_Post_GFAS_PA_CO2 - HX_Post_noFlux_PA_CO2
# Add to background
HX_total_QFED_PA_CO2t = HXminusBckd_QFED_PA_CO2 + HX_Post_Bckd_PA_CO2
HX_total_GFED_PA_CO2t = HXminusBckd_GFED_PA_CO2 + HX_Post_Bckd_PA_CO2
HX_total_GFAS_PA_CO2t = HXminusBckd_GFAS_PA_CO2 + HX_Post_Bckd_PA_CO2
# Remove mean data-model bias
HX_total_QFED_PA_CO2 = HX_total_QFED_PA_CO2t - np.mean(HX_total_QFED_PA_CO2t) + np.mean(Y_Post_Bckd_PA_CO2)
HX_total_GFED_PA_CO2 = HX_total_GFED_PA_CO2t - np.mean(HX_total_GFED_PA_CO2t) + np.mean(Y_Post_Bckd_PA_CO2)
HX_total_GFAS_PA_CO2 = HX_total_GFAS_PA_CO2t - np.mean(HX_total_GFAS_PA_CO2t) + np.mean(Y_Post_Bckd_PA_CO2)

fig = plt.figure(1,figsize=(5,4),dpi=300)
#
ax1 = fig.add_axes([0.1, 1.05/2., 0.87, 0.8/2.])
plt.plot(time_vec,Y_Post_QFED_PA,'k.')
plt.plot(time_vec,HX_Post_QFED_PA,'r.')
plt.plot(time_vec,HX_Post_GFAS_PA,'g.')
plt.plot(time_vec,HX_Post_GFED_PA,'b.')
plt.ylim([70,300])
plt.title('Park Falls XCO (ppb)')
#
ax1 = fig.add_axes([0.1, 0.1/2., 0.87, 0.8/2.])
plt.plot(time_vec,Y_Post_QFED_PA-HX_Post_QFED_PA,'r.')
plt.plot(time_vec,Y_Post_QFED_PA-HX_Post_GFAS_PA,'g.')
plt.plot(time_vec,Y_Post_QFED_PA-HX_Post_GFED_PA,'b.')
plt.ylim([-75,75])
#
plt.savefig('timeseries_PA_CO_20231122.png')

fig = plt.figure(2,figsize=(5,4),dpi=300)
#
ax1 = fig.add_axes([0.1, 1.05/2., 0.87, 0.8/2.])
plt.plot(time_vec,Y_Post_QFED_PA_CO2,'k.')
plt.plot(time_vec,HX_total_QFED_PA_CO2,'r.')
plt.plot(time_vec,HX_total_GFAS_PA_CO2,'g.')
plt.plot(time_vec,HX_total_GFED_PA_CO2,'b.')
plt.ylim([410,425])
plt.title('Park Falls XCO2 (ppm)')
#
ax1 = fig.add_axes([0.1, 0.1/2., 0.87, 0.8/2.])
plt.plot(time_vec,Y_Post_QFED_PA_CO2-HX_total_QFED_PA_CO2,'r.')
plt.plot(time_vec,Y_Post_QFED_PA_CO2-HX_total_GFAS_PA_CO2,'g.')
plt.plot(time_vec,Y_Post_QFED_PA_CO2-HX_total_GFED_PA_CO2,'b.')
plt.ylim([-5,5])
#
plt.savefig('timeseries_PA_CO2_20231122.png')

file_out='/u/bbyrne1/TCCON_PA_cosamples_20231122.nc'
print(file_out)
dataset = Dataset(file_out,'w')
nSamples = dataset.createDimension('nSamples',np.size(time_vec))
#                                                                                                                                                                                                                                   
times = dataset.createVariable('time', np.float64, ('nSamples',))
times[:] = time_vec
times.units = 'day of year'
#                                                                                                                                                                                                                                   
Y_XCOs = dataset.createVariable('Y_XCO', np.float64, ('nSamples',))
Y_XCOs[:] = Y_Post_QFED_PA
Y_XCOs.units = 'ppb'
Y_XCOs.long_name = 'TCCON XCO'
#
Hx_QFED_XCOs = dataset.createVariable('Hx_QFED_XCO', np.float64, ('nSamples',))
Hx_QFED_XCOs[:] = HX_Post_QFED_PA
Hx_QFED_XCOs.units = 'ppb'
Hx_QFED_XCOs.long_name = 'Simulated posterior XCO w/ QFED prior'
#
Hx_GFED_XCOs = dataset.createVariable('Hx_GFED_XCO', np.float64, ('nSamples',))
Hx_GFED_XCOs[:] = HX_Post_GFED_PA
Hx_GFED_XCOs.units = 'ppb'
Hx_GFED_XCOs.long_name = 'Simulated posterior XCO w/ GFED prior'
#
Hx_GFAS_XCOs = dataset.createVariable('Hx_GFAS_XCO', np.float64, ('nSamples',))
Hx_GFAS_XCOs[:] = HX_Post_GFAS_PA
Hx_GFAS_XCOs.units = 'ppb'
Hx_GFAS_XCOs.long_name = 'Simulated posterior XCO w/ GFAS prior'
#
Y_XCO2s = dataset.createVariable('Y_XCO2', np.float64, ('nSamples',))
Y_XCO2s[:] = Y_Post_QFED_PA_CO2
Y_XCO2s.units = 'ppm'
Y_XCO2s.long_name = 'TCCON XCO2'
#
Hx_QFED_XCO2s = dataset.createVariable('Hx_QFED_XCO2', np.float64, ('nSamples',))
Hx_QFED_XCO2s[:] = HX_total_QFED_PA_CO2
Hx_QFED_XCO2s.units = 'ppm'
Hx_QFED_XCO2s.long_name = 'Simulated posterior XCO2 w/ QFED prior'
#
Hx_GFED_XCO2s = dataset.createVariable('Hx_GFED_XCO2', np.float64, ('nSamples',))
Hx_GFED_XCO2s[:] = HX_total_GFED_PA_CO2
Hx_GFED_XCO2s.units = 'ppm'
Hx_GFED_XCO2s.long_name = 'Simulated posterior XCO2 w/ GFED prior'
#
Hx_GFAS_XCO2s = dataset.createVariable('Hx_GFAS_XCO2', np.float64, ('nSamples',))
Hx_GFAS_XCO2s[:] = HX_total_GFAS_PA_CO2
Hx_GFAS_XCO2s.units = 'ppm'
Hx_GFAS_XCO2s.long_name = 'Simulated posterior XCO2 w/ GFAS prior'
#
dataset.close()
















time_vec = read_TCCON_obs('TCCON_ETL_GGG2020_XCO')
# ====================================================
Y_Post_QFED_ETL, HX_Post_QFED_ETL, doy_Post_QFED_ETL = read_TCCON('Post_COinv_QFED_AprSep','TCCON_ETL_GGG2020_XCO',1e9)
Y_Post_GFAS_ETL, HX_Post_GFAS_ETL, doy_Post_GFAS_ETL = read_TCCON('Post_COinv_GFAS_AprSep','TCCON_ETL_GGG2020_XCO',1e9)
Y_Post_GFED_ETL, HX_Post_GFED_ETL, doy_Post_GFED_ETL = read_TCCON('Post_COinv_GFED_AprSep','TCCON_ETL_GGG2020_XCO',1e9)
# ====================================================
Y_Post_QFED_ETL_CO2, HX_Post_QFED_ETL_CO2, doy_Post_QFED_ETL = read_TCCON('FWD-CO2-QFED-Post','TCCON_ETL_GGG2020_XCO2',1e6)
Y_Post_GFED_ETL_CO2, HX_Post_GFED_ETL_CO2, doy_Post_GFED_ETL = read_TCCON('FWD-CO2-GFED-Post','TCCON_ETL_GGG2020_XCO2',1e6)
Y_Post_GFAS_ETL_CO2, HX_Post_GFAS_ETL_CO2, doy_Post_GFAS_ETL = read_TCCON('FWD-CO2-GFAS-Post','TCCON_ETL_GGG2020_XCO2',1e6)
Y_Post_noFlux_ETL_CO2, HX_Post_noFlux_ETL_CO2, doy_Post_noFlux_ETL = read_TCCON('FWD-CO2-noFlux','TCCON_ETL_GGG2020_XCO2',1e6)
Y_Post_Bckd_ETL_CO2, HX_Post_Bckd_ETL_CO2, doy_Post_Bckd_ETL = read_TCCON('FWD-CO2-Bckd','TCCON_ETL_GGG2020_XCO2',1e6)

# Isolate the Biomass burning signal
HXminusBckd_QFED_ETL_CO2 = HX_Post_QFED_ETL_CO2 - HX_Post_noFlux_ETL_CO2
HXminusBckd_GFED_ETL_CO2 = HX_Post_GFED_ETL_CO2 - HX_Post_noFlux_ETL_CO2
HXminusBckd_GFAS_ETL_CO2 = HX_Post_GFAS_ETL_CO2 - HX_Post_noFlux_ETL_CO2
# Add to background
HX_total_QFED_ETL_CO2t = HXminusBckd_QFED_ETL_CO2 + HX_Post_Bckd_ETL_CO2
HX_total_GFED_ETL_CO2t = HXminusBckd_GFED_ETL_CO2 + HX_Post_Bckd_ETL_CO2
HX_total_GFAS_ETL_CO2t = HXminusBckd_GFAS_ETL_CO2 + HX_Post_Bckd_ETL_CO2
# Remove mean data-model bias
HX_total_QFED_ETL_CO2 = HX_total_QFED_ETL_CO2t - np.mean(HX_total_QFED_ETL_CO2t) + np.mean(Y_Post_Bckd_ETL_CO2)
HX_total_GFED_ETL_CO2 = HX_total_GFED_ETL_CO2t - np.mean(HX_total_GFED_ETL_CO2t) + np.mean(Y_Post_Bckd_ETL_CO2)
HX_total_GFAS_ETL_CO2 = HX_total_GFAS_ETL_CO2t - np.mean(HX_total_GFAS_ETL_CO2t) + np.mean(Y_Post_Bckd_ETL_CO2)

fig = plt.figure(11,figsize=(5,4),dpi=300)
#
ax1 = fig.add_axes([0.1, 1.05/2., 0.87, 0.8/2.])
plt.plot(time_vec,Y_Post_QFED_ETL,'k.')
plt.plot(time_vec,HX_Post_QFED_ETL,'r.')
plt.plot(time_vec,HX_Post_GFAS_ETL,'g.')
plt.plot(time_vec,HX_Post_GFED_ETL,'b.')
plt.ylim([70,300])
plt.title('ETL XCO (ppb)')
#
ax1 = fig.add_axes([0.1, 0.1/2., 0.87, 0.8/2.])
plt.plot(time_vec,Y_Post_QFED_ETL-HX_Post_QFED_ETL,'r.')
plt.plot(time_vec,Y_Post_QFED_ETL-HX_Post_GFAS_ETL,'g.')
plt.plot(time_vec,Y_Post_QFED_ETL-HX_Post_GFED_ETL,'b.')
plt.ylim([-75,75])
#
plt.savefig('timeseries_ETL_CO_20231122.png')

fig = plt.figure(12,figsize=(5,4),dpi=300)
#
ax1 = fig.add_axes([0.1, 1.05/2., 0.87, 0.8/2.])
plt.plot(time_vec,Y_Post_QFED_ETL_CO2,'k.')
plt.plot(time_vec,HX_total_QFED_ETL_CO2,'r.')
plt.plot(time_vec,HX_total_GFAS_ETL_CO2,'g.')
plt.plot(time_vec,HX_total_GFED_ETL_CO2,'b.')
plt.ylim([410,425])
plt.title('ETL XCO2 (ppm)')
#
ax1 = fig.add_axes([0.1, 0.1/2., 0.87, 0.8/2.])
plt.plot(time_vec,Y_Post_QFED_ETL_CO2-HX_total_QFED_ETL_CO2,'r.')
plt.plot(time_vec,Y_Post_QFED_ETL_CO2-HX_total_GFAS_ETL_CO2,'g.')
plt.plot(time_vec,Y_Post_QFED_ETL_CO2-HX_total_GFED_ETL_CO2,'b.')
plt.ylim([-5,5])
#
plt.savefig('timeseries_ETL_CO2_20231122.png')

file_out='/u/bbyrne1/TCCON_ETL_cosamples_20231122.nc'
print(file_out)
dataset = Dataset(file_out,'w')
nSamples = dataset.createDimension('nSamples',np.size(time_vec))
#                                                                                                                                                                                                                                   
times = dataset.createVariable('time', np.float64, ('nSamples',))
times[:] = time_vec
times.units = 'day of year'
#                                                                                                                                                                                                                                   
Y_XCOs = dataset.createVariable('Y_XCO', np.float64, ('nSamples',))
Y_XCOs[:] = Y_Post_QFED_ETL
Y_XCOs.units = 'ppb'
Y_XCOs.long_name = 'TCCON XCO'
#
Hx_QFED_XCOs = dataset.createVariable('Hx_QFED_XCO', np.float64, ('nSamples',))
Hx_QFED_XCOs[:] = HX_Post_QFED_ETL
Hx_QFED_XCOs.units = 'ppb'
Hx_QFED_XCOs.long_name = 'Simulated posterior XCO w/ QFED prior'
#
Hx_GFED_XCOs = dataset.createVariable('Hx_GFED_XCO', np.float64, ('nSamples',))
Hx_GFED_XCOs[:] = HX_Post_GFED_ETL
Hx_GFED_XCOs.units = 'ppb'
Hx_GFED_XCOs.long_name = 'Simulated posterior XCO w/ GFED prior'
#
Hx_GFAS_XCOs = dataset.createVariable('Hx_GFAS_XCO', np.float64, ('nSamples',))
Hx_GFAS_XCOs[:] = HX_Post_GFAS_ETL
Hx_GFAS_XCOs.units = 'ppb'
Hx_GFAS_XCOs.long_name = 'Simulated posterior XCO w/ GFAS prior'
#
Y_XCO2s = dataset.createVariable('Y_XCO2', np.float64, ('nSamples',))
Y_XCO2s[:] = Y_Post_QFED_ETL_CO2
Y_XCO2s.units = 'ppm'
Y_XCO2s.long_name = 'TCCON XCO2'
#
Hx_QFED_XCO2s = dataset.createVariable('Hx_QFED_XCO2', np.float64, ('nSamples',))
Hx_QFED_XCO2s[:] = HX_total_QFED_ETL_CO2
Hx_QFED_XCO2s.units = 'ppm'
Hx_QFED_XCO2s.long_name = 'Simulated posterior XCO2 w/ QFED prior'
#
Hx_GFED_XCO2s = dataset.createVariable('Hx_GFED_XCO2', np.float64, ('nSamples',))
Hx_GFED_XCO2s[:] = HX_total_GFED_ETL_CO2
Hx_GFED_XCO2s.units = 'ppm'
Hx_GFED_XCO2s.long_name = 'Simulated posterior XCO2 w/ GFED prior'
#
Hx_GFAS_XCO2s = dataset.createVariable('Hx_GFAS_XCO2', np.float64, ('nSamples',))
Hx_GFAS_XCO2s[:] = HX_total_GFAS_ETL_CO2
Hx_GFAS_XCO2s.units = 'ppm'
Hx_GFAS_XCO2s.long_name = 'Simulated posterior XCO2 w/ GFAS prior'
#
dataset.close()
