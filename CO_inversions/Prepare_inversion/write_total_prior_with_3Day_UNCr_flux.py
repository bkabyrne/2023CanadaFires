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

# ######################################################
#
#  This program writes the prior fluxes used for the 3-day
#  inversions. The component emissions are combined and the
#  uncertainties are constructed.
#
# ######################################################

def read_combine_fluxes(inventory,year,month,day):
    #
    # =========================================
    # Read and combine fire, FF & biogenic fluxes 
    # =========================================
    #
    # -- Read Fire emissions
    if inventory == 'GFED':
        nc_BBprior ='/nobackup/bbyrne1/Flux_2x25_CO/BiomassBurn/GFED41s/'+str(year).zfill(4)+'/'+str(month).zfill(2)+'/'+str(day).zfill(2)+'.nc'
        f=Dataset(nc_BBprior,mode='r')
        CO_flux_BB = f.variables['CO_Flux'][:] 
        f.close()
    else:
        # Diurnal information is from GFED                                                                           
        nc_GFED_diurnal_3hr = '/nobackupp17/bbyrne1/GFED41s_2x25_diurnal_scale/2022/'+str(month).zfill(2)+'/'+str(day).zfill(2)+'.nc'
        f=Dataset(nc_GFED_diurnal_3hr,mode='r')
        GFED_diurnal_3hr = f.variables['diurnal'][:]
        f.close()
        if (year % 4) == 0:
            GFED_diurnal_3hr = np.append(GFED_diurnal_3hr[0:361,:,:],GFED_diurnal_3hr[360:365,:,:],axis=0)
        # Daily fire emissions                                                                                       
        nc_BBprior = '/nobackup/bbyrne1/Flux_2x25_CO/BiomassBurn/'+inventory+'/'+str(year).zfill(4)+'/'+str(month).zfill(2)+'/'+str(day).zfill(2)+'.nc'
        f=Dataset(nc_BBprior,mode='r')
        CO_flux_BB_daily = f.variables['CO_Flux'][:]
        f.close()
        #
        CO_flux_BB = GFED_diurnal_3hr*np.repeat(CO_flux_BB_daily[np.newaxis, :, :], 8, axis=0)
    #
    # -- Read Fossil Fuel emissions
    nc_FFprior ='/nobackup/bbyrne1/Flux_2x25_CO/FossilFuel/CEDSdaily/'+str(year).zfill(4)+'/'+str(month).zfill(2)+'/'+str(day).zfill(2)+'.nc'
    f=Dataset(nc_FFprior,mode='r')
    CO_flux_FF = np.repeat(f.variables['CO_Flux'][:][np.newaxis,:,:],8,axis=0)
    f.close()
    #
    # -- Read Biogenic emissions
    nc_BioPrior ='/nobackup/bbyrne1/Flux_2x25_CO/Biogenic_units/'+str(year).zfill(4)+'/'+str(month).zfill(2)+'/'+str(day).zfill(2)+'.nc'
    f=Dataset(nc_BioPrior,mode='r')
    CO_flux_Biogenic = np.repeat(f.variables['CO_Flux'][:][np.newaxis,:,:],8,axis=0)
    f.close()        
    #
    if np.sum(CO_flux_BB)>5:
        print(inventory+' on '+str(year).zfill(4)+'/'+str(month).zfill(2)+'/'+str(day).zfill(2))
        print('Sum FF: %.4f Sum BB: %.4f' %
              (np.sum(CO_flux_FF), np.sum(CO_flux_BB) ) )

            
    # -- Calculate total flux
    total_flux = CO_flux_BB + CO_flux_FF + CO_flux_Biogenic
    #
    return total_flux
    # =============================================

def calculate_uncertainty(total_flux):
    #
    # =========================================
    # Calculate SF uncertainty
    #
    # Expecting input array from April 1 to Sep 30!!!
    #
    # =========================================
    #
    # We will use 3-Day grouping for optimizations                    
    prior_grouped = np.zeros((61,91,144))
    for ind in range(61): # each ind is a temporal grouping of 3 day
        prior_grouped[ind,:,:] = np.mean(np.mean(total_flux[(ind)*3:(ind+1)*3,:,:,:],1),0)
        #print(ind)
        #print(np.max(prior_grouped[ind,:,:]))
    #
    prior_state_vec2 = prior_grouped.flatten() # flatten the 3D array
    prior_state_vec3 = prior_state_vec2[np.where(prior_state_vec2>0)] # find fluxes greater than zero

    SF_unc1 = (np.mean(prior_state_vec3)*10.) / prior_grouped # uncertainty is "mean flux * 10" (note this is scale factor uncertainty)                                
    SF_unc1[np.where(np.isfinite(SF_unc1)==0)]=1 # non-finite values get 1 scale factor uncertainty                
    SF_unc1[np.where(SF_unc1<0.25)]=0.25 # scale factor uncertainties must be greater or equal to 25%              
    SF_unc1[np.where(SF_unc1>1000)]=1000 # scale factor uncertainty must be less than or equal to 1000x            

    # ==============================================      
    prior_UNC1 = np.zeros((61*3,91,144)) # Map 3day uncertainties to daily values                                   
    for j in range(61):
        for dw in range(3):
            #print(j*3+dw)
            prior_UNC1[j*3+dw,:,:] = SF_unc1[j,:,:]
    #                                                     
    prior_UNC1[np.where(prior_UNC1 == 0)] = 1
    # ==============================================      

    return prior_UNC1


def write_fluxes(inventory,year,month,day,total_flux,unc_SF):
    #
    # =========================================
    # Write out fluxes and SF uncertainty
    # =========================================
    #
    nc_out = '/nobackup/bbyrne1/Flux_2x25_CO/Combined/FF_'+inventory+'_Bio_UNCr_3Day/'+str(year).zfill(4)+'/'+str(month).zfill(2)+'/'+str(day).zfill(2)+'.nc'
    #
    dataset = Dataset(nc_out,'w')
    #
    times = dataset.createDimension('time',8)
    lats = dataset.createDimension('lat',91)
    lons = dataset.createDimension('lon',144)
    #
    postBBs = dataset.createVariable('CO_Flux', np.float64, ('time','lat','lon'))
    postBBs[:,:,:] = total_flux
    postBBs.units = 'kgC/km2/s'
    #
    CO2_UNC1 = dataset.createVariable('Uncertainty',np.float64,('lat','lon'))
    CO2_UNC1[:,:] = unc_SF
    CO2_UNC1.units = 'SF'
    #
    dataset.close()

def write_fluxes_noUNC(inventory,year,month,day,total_flux):
    #
    # =========================================
    # Write out fluxes and SF uncertainty
    # =========================================
    #
    nc_out = '/nobackup/bbyrne1/Flux_2x25_CO/Combined/FF_'+inventory+'_Bio_noUNC_3Day/'+str(year).zfill(4)+'/'+str(month).zfill(2)+'/'+str(day).zfill(2)+'.nc'
    #
    dataset = Dataset(nc_out,'w')
    #
    times = dataset.createDimension('time',8)
    lats = dataset.createDimension('lat',91)
    lons = dataset.createDimension('lon',144)
    #
    postBBs = dataset.createVariable('CO_Flux', np.float64, ('time','lat','lon'))
    postBBs[:,:,:] = total_flux
    postBBs.units = 'kgC/km2/s'
    #
    dataset.close()



# Grid area
grid_area_2x25 = np.array([2.70084e+08,  2.16024e+09,  4.31787e+09,  6.47023e+09,  8.61471e+09,  1.07487e+10,
                           1.28696e+10,  1.49748e+10,  1.70617e+10,  1.91279e+10,  2.11708e+10,  2.31879e+10,
                           2.51767e+10,  2.71348e+10,  2.90599e+10,  3.09496e+10,  3.28016e+10,  3.46136e+10,
                           3.63835e+10,  3.81090e+10,  3.97881e+10,  4.14187e+10,  4.29988e+10,  4.45266e+10,
                           4.60001e+10,  4.74175e+10,  4.87772e+10,  5.00775e+10,  5.13168e+10,  5.24935e+10,
                           5.36063e+10,  5.46538e+10,  5.56346e+10,  5.65477e+10,  5.73920e+10,  5.81662e+10,
                           5.88696e+10,  5.95014e+10,  6.00606e+10,  6.05466e+10,  6.09588e+10,  6.12968e+10,
                           6.15601e+10,  6.17484e+10,  6.18615e+10,  6.18992e+10,  6.18615e+10,  6.17484e+10,
                           6.15601e+10,  6.12968e+10,  6.09588e+10,  6.05466e+10,  6.00606e+10,  5.95014e+10,
                           5.88696e+10,  5.81662e+10,  5.73920e+10,  5.65477e+10,  5.56346e+10,  5.46538e+10,
                           5.36063e+10,  5.24935e+10,  5.13168e+10,  5.00775e+10,  4.87772e+10,  4.74175e+10,
                           4.60001e+10,  4.45266e+10,  4.29988e+10,  4.14187e+10,  3.97881e+10,  3.81090e+10,
                           3.63835e+10,  3.46136e+10,  3.28016e+10,  3.09496e+10,  2.90599e+10,  2.71348e+10,
                           2.51767e+10,  2.31879e+10,  2.11708e+10,  1.91279e+10,  1.70617e+10,  1.49748e+10,
                           1.28696e+10, 1.07487e+10,  8.61471e+09,  6.47023e+09,  4.31787e+09,  2.16024e+09,
                           2.70084e+08])
grid_area_2x25_arr = np.zeros((91,144))
for ii in range(144):
    grid_area_2x25_arr[:,ii] = grid_area_2x25

# Kg/Km2/s * 1000g/kg * 1km/1000m * 1km/1000m * (60*60*24)s/day 
#(60.*60.*24.)/1000. 

inventory_array = ['GFED','GFAS','QFED']
#inventory_array = ['GFAS']

for inventory in inventory_array:
    print(inventory)
    # Loop over years
    for yyyy in range(2019,2023+1):
        print(yyyy)
        
        # Check days in months for the year
        if (yyyy % 4) == 0:
            #print(str(yyyy).zfill(4)+' is a leap year')
            days_in_months = np.array([31,29,31,30,31,30,31,31,30,31,30,31])
        else:
            days_in_months = np.array([31,28,31,30,31,30,31,31,30,31,30,31])

        # Read and combine the Fire, FF and biogenic fluxes
        total_flux = np.zeros((np.sum(days_in_months[3:9]),8, 91, 144))
        n1=0
        for mm in range(3,9):
            for dd in range(days_in_months[mm]):
                #
                total_flux[n1,:,:,:] = read_combine_fluxes(inventory,yyyy,mm+1,dd+1)
                #
                n1=n1+1

        # Calculate uncertainties on the fluxes
        unc_SF = calculate_uncertainty(total_flux)

    
        # Write the fluxes
        n1=0
        for mm in range(3,9): # Apr-Sep
            for dd in range(days_in_months[mm]):
                #
                write_fluxes(inventory,yyyy,mm+1,dd+1,total_flux[n1,:,:,:],unc_SF[n1,:,:])
                write_fluxes_noUNC(inventory,yyyy,mm+1,dd+1,total_flux[n1,:,:,:])
                #
                n1=n1+1            
        # -----------------------
        
for i in range(30):
    ncfile = '/nobackup/bbyrne1/Flux_2x25_CO/Combined/FF_GFED_Bio_UNCr_3Day/2023/04/'+str(i+1).zfill(2)+'.nc'
    f = Dataset(ncfile,'r')
    UNC = f.variables['Uncertainty'][:]
    print(np.max(UNC))
    print(np.min(UNC))
    f.close()
