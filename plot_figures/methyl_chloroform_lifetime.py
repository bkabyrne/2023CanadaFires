import xarray as xr
import numpy as np
import os
from datetime import datetime, timedelta
from math import pi, cos, radians
import matplotlib.pyplot as plt

"""
Program to calculate the lifetime of Methyl Chloroform given an OH and temperature field. 
We should expect a lifetime of 5-6 years.

---------
From GEOS-CHEM WIKI (https://wiki.seas.harvard.edu/geos-chem/index.php/Methyl_chloroform_lifetime):

The CH3CCl3 lifetime is the destruction rate of CH3CCl3 by OH. 
For the reaction OH + CH3CCl3, the reaction rate from JPL2006 is:

    k = 1.64e-12 * exp(-1520 / T)

The lifetime of CH3CCl3 is:

    1 / (k * [OH])

If OH is 1e6 molecule/cm^3 and temperature is 298K, the lifetime of CH3CCl3 would be 1e8 seconds, which equals to 3.2 years.
---------

For the global mean lifetime, I calculated the airmass-weighted mean OH fields and reaction rate. Note that calculating lifetimes for individual gridcells doesnt work because 1/OH -> infinity when OH is low.

"""

def define_constants():
    """                                                                                                             
    Defines atmospheric constants used for calculations.                                                            
                                                                                                                    
    Returns:                                                                                                        
    - dict: A dictionary containing atmospheric constants and arrays for pressure calculations.                     
    """
    # === Constants ===                                                                                             
    atm_const = {
        'gravity': 9.8, # m/s2                                                                                      
        'AIR_MW': 28./1000., # kg/mol                                                                               
        'R_dryair': 287.058  # J kg-1 K-1 = m2 s-2 K-1 # Specific gas constant for dry air                          
    }
    # === Stuff for calculating GEOS-Chem pressure ===                                                              
    #(surface)                                                                                                      
    Ap = np.array([0.000000e+00, 4.804826e-02, 6.593752e+00, 1.313480e+01, 1.961311e+01, 2.609201e+01,
                   3.257081e+01, 3.898201e+01, 4.533901e+01, 5.169611e+01, 5.805321e+01, 6.436264e+01,
                   7.062198e+01, 7.883422e+01, 8.909992e+01, 9.936521e+01, 1.091817e+02, 1.189586e+02,
                   1.286959e+02, 1.429100e+02, 1.562600e+02, 1.696090e+02, 1.816190e+02, 1.930970e+02,
                   2.032590e+02, 2.121500e+02, 2.187760e+02, 2.238980e+02, 2.243630e+02, 2.168650e+02,
                   2.011920e+02, 1.769300e+02, 1.503930e+02, 1.278370e+02, 1.086630e+02, 9.236572e+01,
                   7.851231e+01, 6.660341e+01, 5.638791e+01, 4.764391e+01, 4.017541e+01, 3.381001e+01,
                   2.836781e+01, 2.373041e+01, 1.979160e+01, 1.645710e+01, 1.364340e+01, 1.127690e+01,
                   9.292942e+00, 7.619842e+00, 6.216801e+00, 5.046801e+00, 4.076571e+00, 3.276431e+00,
                   2.620211e+00, 2.084970e+00, 1.650790e+00, 1.300510e+00, 1.019440e+00, 7.951341e-01,
                   6.167791e-01, 4.758061e-01, 3.650411e-01, 2.785261e-01, 2.113490e-01, 1.594950e-01,
                   1.197030e-01, 8.934502e-02, 6.600001e-02, 4.758501e-02, 3.270000e-02, 2.000000e-02,
                   1.000000e-02])
    #(top of atmosphere)                                                                                            
    #(surface)                                                                                                      
    Bp = np.array([1.000000e+00, 9.849520e-01, 9.634060e-01, 9.418650e-01, 9.203870e-01, 8.989080e-01,
                   8.774290e-01, 8.560180e-01, 8.346609e-01, 8.133039e-01, 7.919469e-01, 7.706375e-01,
                   7.493782e-01, 7.211660e-01, 6.858999e-01, 6.506349e-01, 6.158184e-01, 5.810415e-01,
                   5.463042e-01, 4.945902e-01, 4.437402e-01, 3.928911e-01, 3.433811e-01, 2.944031e-01,
                   2.467411e-01, 2.003501e-01, 1.562241e-01, 1.136021e-01, 6.372006e-02, 2.801004e-02,
                   6.960025e-03, 8.175413e-09, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                   0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                   0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                   0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                   0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                   0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                   0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                   0.000000e+00])
    #(top of atmosphere)
    #
    atm_const['Ap_repeated'] = np.tile(Ap[np.newaxis, :, np.newaxis, np.newaxis], (8, 1, 91, 144))
    atm_const['Bp_repeated'] = np.tile(Bp[np.newaxis, :, np.newaxis, np.newaxis], (8, 1, 91, 144))

    # ==========
    MERRA2_atm = xr.open_dataset('/nobackup/bbyrne1/MERRA2/2x2.5/2023/01/MERRA2.20230101.I3.2x25.nc4')
    xres = 2.5
    yres = 2.0
    earth_radius = 6371009 # in meters                    
    lat_dist0 = pi * earth_radius / 180.0
    y = MERRA2_atm['lon'].values*0. + yres * lat_dist0
    x= MERRA2_atm['lat'].values*0.
    for i in range(np.size(MERRA2_atm['lat'].values)):
        x[i]= xres * lat_dist0 * cos(radians(MERRA2_atm['lat'].values[i]))
    area_res = np.zeros((np.size(x),np.size(y)))
    for i in range(np.size(y)):
        for j in range(np.size(x)):
            area_res[j,i] = np.abs(x[j]*y[i])
    # ==========
    atm_const['area_repeated'] = np.tile(area_res[np.newaxis, np.newaxis, :, :], (8, 47, 1, 1))
    
    return atm_const
    # ------------------------------------------------------------- 

def calculate_airmass(atm_const, MERRA2_atm):

    '''

    Hydrostatic Balance:
        dm = dP * A / g  

    '''
    PS = np.tile(MERRA2_atm['PS'].values[:,np.newaxis, :, :], (1, 48, 1, 1))
    
    # Pressure at edge and middle points of layers                                                                   
    Pedge = atm_const['Ap_repeated'][:,0:48,:,:] * 100 + (atm_const['Bp_repeated'][:,0:48,:,:] * PS)
    Pdiff = Pedge[:,1:,:,:] - Pedge[:,:-1,:,:]

    airmass = (Pdiff[:,0:47,:,:] * atm_const['area_repeated']) / atm_const['gravity']

    return airmass
    
def calculate_CH3CCl3_daily_lifetime(temperature_arr, OH_arr,airmass):
    """
    Calculate the CH3CCl3 lifetime for a given day in seconds.

    Parameters:
    temperature (xarray.DataArray): Temperature [K] with dimensions (time, lev, lat, lon)
    OH_fields (xarray.DataArray): OH fields [molecule/cm^3] with dimensions (time, lev, lat, lon)
    airmass (numpy array): mass of air in gridcell

    Returns:
    float: Mean CH3CCl3 lifetime across all grid cells
    """
    #Set zero OH to NAN
    OH_arr[np.where(OH_arr==0)] = np.nan
    # Find non-NAN OH
    II = np.where(np.isfinite(OH_arr)==1)

    k = 1.64e-12 * np.exp(-1520.0 / temperature_arr)
    # Global mean reaction rate [mass-weighted]
    k_mean = np.sum(k[II] * airmass[II]) / np.sum(airmass[II])
    
    # Global mean OH [mass-weighted]
    OH_arr_mean = np.sum(OH_arr[II] * airmass[II]) / np.sum(airmass[II])

    # Calculate lifetime for global mean field
    lifetime_mean = ( 1 / (k_mean * OH_arr_mean) ) / (60.*60.*24.*365.25) # years

    # Return mean lifetime across all grid cells
    return OH_arr_mean, lifetime_mean

if __name__ == '__main__':

    # define constants
    atm_const = define_constants()
    
    # Initialize arrays to store lifetimes
    Kazu_lifetime = np.zeros(365)
    GC_lifetime = np.zeros(365)
    Kazu_OH = np.zeros(365)
    GC_OH = np.zeros(365)

    # Iterate over each day in 2023
    start_date = datetime(2023, 1, 1)
    for day_of_year in range(365):
        current_date = start_date + timedelta(days=day_of_year)
        date_str = current_date.strftime('%Y%m%d')
        print(date_str)
        
        # Load MERRA2 data
        nc_file = f'/nobackup/bbyrne1/MERRA2/2x2.5/2023/{current_date.strftime("%m")}/MERRA2.{date_str}.I3.2x25.nc4'
        if not os.path.exists(nc_file):
            raise FileNotFoundError(f"File not found: {nc_file}")

        MERRA2_atm = xr.open_dataset(nc_file)
        temperature = MERRA2_atm['T']

        airmass = calculate_airmass(atm_const, MERRA2_atm)

        # Load OH fields data
        OH_filename = f'/nobackupp17/bbyrne1/OH_fields_2x25/2023/{current_date.strftime("%m")}/{current_date.strftime("%d")}.nc'
        if not os.path.exists(OH_filename):
            raise FileNotFoundError(f"File not found: {OH_filename}")

        OH_data = xr.open_dataset(OH_filename)
        # Convert OH fields to molecules/cm^3
        OH_data_converted = OH_data['OH'] * (1000.0 * 6.022e23 / 17) * 1.0e-15
        OH_3hr = np.zeros((8,47,91,144))
        for i in range(8):
            OH_3hr[i,:,:,:] = np.mean(OH_data_converted[i*3:(i+1)*3,:,:,:],0)
        # Calculate CH3CCl3 lifetime
        Kazu_OH[day_of_year], Kazu_lifetime[day_of_year] = calculate_CH3CCl3_daily_lifetime(temperature.values[:, 0:30, :, :], OH_3hr[:, 0:30, :, :],airmass[:, 0:30, :, :])

        # Load GC OH fields data
        GC_OH_filename = f'/nobackupp17/bbyrne1/GC_OH_fields_2x25/2023/{current_date.strftime("%m")}/{current_date.strftime("%d")}.nc'
        if not os.path.exists(GC_OH_filename):
            raise FileNotFoundError(f"File not found: {GC_OH_filename}")

        GC_OH_data = xr.open_dataset(GC_OH_filename)
        # Convert OH fields to molecules/cm^3
        GC_OH_data_converted = GC_OH_data['OH'] * (1000.0 * 6.022e23 / 17) * 1.0e-15
        GC_OH_3hr = np.zeros((8,47,91,144))
        for i in range(8):
            GC_OH_3hr[i,:,:,:] = np.mean(GC_OH_data_converted[i*3:(i+1)*3,:,:,:],0)
        # Calculate CH3CCl3 lifetime
        GC_OH[day_of_year], GC_lifetime[day_of_year] = calculate_CH3CCl3_daily_lifetime(temperature.values[:, 0:30, :, :], GC_OH_3hr[:, 0:30, :, :],airmass[:, 0:30, :, :])

    # Print the calculated lifetimes
    #print("Kazu Lifetime: ", Kazu_lifetime)
    #print("GC Lifetime: ", GC_lifetime)


# 60 sec/min
# 60 min/hr
# 24 hr/day
# 365.25 day/yr

covert_s_to_year = 1. / (60.*60.*24.*365.25)

#nc_file ='/nobackup/bbyrne1/MERRA2/2x2.5/'+TROPOMI_ob['year']+'/'+TROPOMI_ob['month']+'/MERRA2.'+TROPOMI_ob['year']+TROPOMI_ob['month']+TROPOMI_ob['day']+'.I3.2x25.nc4'
#GCKazu_OH_fields_base = '/nobackupp17/bbyrne1/GC_OH_fields_2x25/'
fig = plt.figure(1, figsize=(5,4), dpi=300)
l1 = plt.plot(np.arange(365)+1,Kazu_OH * 1e-5,'b',linewidth=2)
l2 = plt.plot(np.arange(365)+1,GC_OH * 1e-5,'r',linewidth=2)
plt.legend([l1[0],l2[0]],('Kazu','GEOS-Chem'))
plt.savefig('daily_global_mean_OH.png')
