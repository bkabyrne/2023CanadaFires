from mpl_toolkits.basemap import Basemap, cm                                                                   
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import datetime
from netCDF4 import Dataset
import glob, os
import xarray as xr

# Function that creates L2# observation files for CMS-Flux from the TROPOMI CO L2 product.
# This version of the function is specific to the 2x2.5 version of CMS-Flux



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
    atm_const['Ap'] = np.array([0.000000e+00, 4.804826e-02, 6.593752e+00, 1.313480e+01, 1.961311e+01, 2.609201e+01,
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
    atm_const['Bp'] = np.array([1.000000e+00, 9.849520e-01, 9.634060e-01, 9.418650e-01, 9.203870e-01, 8.989080e-01,
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
    return atm_const
    # -------------------------------------------------------------

    
def is_data_within_latlon_grid(latitude, longitude, MERRA2_grid):
    """
    Check if there are any data points within specified latitude and longitude bounds.

    Parameters:
    - latitude (array-like): Array of latitude values to check.
    - longitude (array-like): Array of longitude values to check.
    - MERRA2_grid (dict): Dictionary containing 'lat' and 'lon' keys with their respective min and max values.

    Returns:
    - bool: True if any data points are within both the latitude and longitude bounds, False otherwise.
    """
    is_within_lat = np.any((latitude > MERRA2_grid['lat'].min()) & (latitude < MERRA2_grid['lat'].max()))
    is_within_lon = np.any((longitude > MERRA2_grid['lon'].min()) & (longitude < MERRA2_grid['lon'].max()))
    
    return is_within_lat and is_within_lon
    # -------------------------------------------------------------


def subselect_TROPOMI_data(dict_out, dict_in, time_i, scanline_i, ground_pixel_i):
    """
    Extracts a subset of TROPOMI data based on specified indices.

    Parameters:
    - dict_in (dict): Dictionary containing subselected TROPOMI time data arrays.
    - dict_in (dict): Dictionary containing TROPOMI data arrays.
    - time_i (int): Index for the time dimension.
    - scanline_i (int): Index for the scanline dimension.
    - ground_pixel_i (int): Index for the ground pixel dimension.

    Returns:
    - dict: A dictionary containing the subset of TROPOMI data and additional computed layers.
    """

    # Subselect data for given indices
    dict_out['qa_value'] = dict_in['qa_value'][time_i, scanline_i, ground_pixel_i]
    dict_out['latitude'] = dict_in['latitude'][time_i, scanline_i, ground_pixel_i]
    dict_out['longitude'] = dict_in['longitude'][time_i, scanline_i, ground_pixel_i]
    dict_out['carbonmonoxide_total_column'] = dict_in['carbonmonoxide_total_column'][time_i, scanline_i, ground_pixel_i]
    dict_out['carbonmonoxide_total_column_precision'] = dict_in['carbonmonoxide_total_column_precision'][time_i, scanline_i, ground_pixel_i]
    dict_out['surface_pressure'] = dict_in['surface_pressure'][time_i, scanline_i, ground_pixel_i]
    dict_out['pressure_levels'] = dict_in['pressure_levels'][time_i, scanline_i, ground_pixel_i, :]
    dict_out['column_averaging_kernel'] = dict_in['column_averaging_kernel'][time_i, scanline_i, ground_pixel_i, :]


    # Update bottom layer of pressure levels to match surface pressure
    dict_out['pressure_levels'][49] = dict_out['surface_pressure']

    # Calculate mid-point pressures between layers
    dict_out['pressure_levels_layers'] = (dict_out['pressure_levels'][0:49] + dict_out['pressure_levels'][1:50]) / 2.
    dict_out['column_averaging_kernel_layers'] = dict_out['column_averaging_kernel'][1:50]

    return dict_out
    # -------------------------------------------------------------


def map_TROPOMI_ob_to_model_grid(atm_const, TROPOMI_ob, MERRA2_atm):
    """
    Maps TROPOMI observations to a model grid using atmospheric constants and MERRA2 atmospheric data.

    Parameters:
    - atm_const (dict): Dictionary containing atmospheric constants like gravity and air molecular weight.
    - TROPOMI_ob (dict): Dictionary containing TROPOMI observations data.
    - MERRA2_atm (xarray): xarray containing MERRA2 atmospheric data.

    Returns:
    - dict: Updated TROPOMI_ob dictionary with mapped data including mole fractions and pressure weighting functions.

     ======= Equations ======= 
    
     Hydrostatin Balance:
       m * g = P * A
    
     Specific humidity
       m_dry = m_tot * ( 1 - q )
    
     Hypsometric equation:
     z_2 - z_1 = (R *T / g) * ln(P_1/P_2) 
    
     Therefore,
       m_dry / A = P * ( 1 - q) / g
 
    """

    # Find closest indices in MERRA2 grid
    lon_ind = np.argmin(np.abs(MERRA2_atm['lon'].values - TROPOMI_ob['longitude']))
    lat_ind = np.argmin(np.abs(MERRA2_atm['lat'].values - TROPOMI_ob['latitude']))
    hour = TROPOMI_ob['hour']  # Ensure 'time' is provided in hours
    time_ind = int(np.floor(hour / 3.))

    # Retrieve relevant atmospheric data from MERRA2
    P = MERRA2_atm['PS'][time_ind, lat_ind, lon_ind].values
    q_prof = MERRA2_atm['QV'][time_ind, :, lat_ind, lon_ind].values
    T = MERRA2_atm['T'][time_ind, :, lat_ind, lon_ind].values

    # Pressure at edge and middle points of layers
    Pedge = atm_const['Ap'] * 100 + (atm_const['Bp'] * P)
    Pmid = (Pedge[:-1] + Pedge[1:]) / 2

    # Reverse arrays for interpolation
    T_rev = T[::-1]
    q_rev = q_prof[::-1]
    P_rev = Pmid[::-1]

    # Interpolate temperature and specific humidity
    T_interp = np.interp(TROPOMI_ob['pressure_levels_layers'], P_rev, T_rev)
    q_interp = np.interp(TROPOMI_ob['pressure_levels_layers'], P_rev, q_rev)

    # Calculate dry air mass per area
    delta_P = np.diff(TROPOMI_ob['pressure_levels'])
    P_dry_profile = delta_P * (1 - q_interp)
    P_dry = np.sum(P_dry_profile)
    m_dry_per_A = P_dry / atm_const['gravity']

    # Convert to moles
    mol_dry_per_A = m_dry_per_A / atm_const['AIR_MW']

    # Calculate mole fraction and its precision
    TROPOMI_ob['mole_frac'] = TROPOMI_ob['carbonmonoxide_total_column'] / mol_dry_per_A
    TROPOMI_ob['mole_frac_precision'] = TROPOMI_ob['carbonmonoxide_total_column_precision'] / mol_dry_per_A

    # Calculate pressure weighting function
    TROPOMI_ob['Pressure_Weighting_Function'] = P_dry_profile / P_dry

    # Calculate height differences using hypsometric equation
    delta_h = (atm_const['R_dryair'] * T_interp / atm_const['gravity']) * np.log(TROPOMI_ob['pressure_levels'][1:50] / TROPOMI_ob['pressure_levels'][:-1])

    # Convert averaging kernels
    AK_convert = 1 / delta_h * (P_dry_profile / P_dry)
    TROPOMI_ob['AK_convert_full'] = AK_convert

    return TROPOMI_ob
    # -------------------------------------------------------------


def calculate_TROPOMI_superobs(Daily_TROPOMI_data, MERRA2_grid):

    '''
    Generates super-obs for each gridcell and hour

    Inputs:
      - Daily_TROPOMI_data: Dictionary with daily TROPOMI data
      - MERRA2_grid: Grid with MERRA-2 data

    Outputs:
      - Daily_TROPOMI_superobs: Dictionairy with super-obs
    '''

    # Temporary arrays for the data
    hour_aggt = np.zeros(40000)
    latitude_ob_aggt = np.zeros(40000)
    longitude_ob_aggt = np.zeros(40000)
    mole_frac_aggt = np.zeros(40000)
    mole_frac_precision_aggt = np.zeros(40000)
    pressure_levels_aggt = np.zeros((49,40000))
    column_averaging_kernel_aggt = np.zeros((49,40000))
    Pressure_Weighting_Function_aggt = np.zeros((49,40000))

    n=0
    hour_day = np.arange(24)+0.5
    # Loop over longitudes
    for lonGC in MERRA2_grid['lon'].values:
        # Sub-select longitude range
        IND = np.where(np.logical_and(Daily_TROPOMI_data['longitude']>=lonGC-2.5/2.,Daily_TROPOMI_data['longitude']<lonGC+2.5/2.))
        if np.size(IND)>0:
            latitude_x = Daily_TROPOMI_data['latitude'][IND]
            hour_x = Daily_TROPOMI_data['hour'][IND]
            mole_frac_x = Daily_TROPOMI_data['mole_frac'][IND]
            mole_frac_precision_x = Daily_TROPOMI_data['mole_frac_precision'][IND]
            column_averaging_kernel_x = Daily_TROPOMI_data['column_averaging_kernel'][:,IND[0]]
            pressure_levels_x = Daily_TROPOMI_data['pressure_levels'][:,IND[0]]
            Pressure_Weighting_Function_x = Daily_TROPOMI_data['Pressure_Weighting_Function'][:,IND[0]]
            # Loop over latitudes            
            for latGC in MERRA2_grid['lat'].values:
                # Sub-select latitude range
                IND = np.where(np.logical_and(latitude_x>=latGC-2./2.,latitude_x<latGC+2./2.))
                if np.size(IND)>0:
                    mole_frac_xx = mole_frac_x[IND]
                    mole_frac_precision_xx = mole_frac_precision_x[IND]
                    hour_xx = hour_x[IND]
                    column_averaging_kernel_xx = column_averaging_kernel_x[:,IND[0]]
                    pressure_levels_xx = pressure_levels_x[:,IND[0]]
                    Pressure_Weighting_Function_xx = Pressure_Weighting_Function_x[:,IND[0]]
                    # Loop over hours
                    for hour_int in hour_day:
                        # Sub-select hour
                        IND = np.where(np.logical_and(hour_xx>=hour_int-0.5,hour_xx<hour_int+0.5))
                        if np.size(IND)>0:
                            longitudexxx = lonGC
                            latitudexxx = latGC
                            hourxxx = hour_int
                            mole_frac_xxx = np.mean(mole_frac_xx[IND])
                            mole_frac_precision_xxx = np.mean(mole_frac_precision_xx[IND])
                            column_averaging_kernel_xxx = np.mean(column_averaging_kernel_xx[:,IND[0]],1)
                            pressure_levels_xxx = np.mean(pressure_levels_xx[:,IND[0]],1)
                            Pressure_Weighting_Function_xxx = np.mean(Pressure_Weighting_Function_xx[:,IND[0]],1)
                            n=n+1
                            # Add super-ob to array
                            hour_aggt[ind_agg] = hourxxx
                            latitude_ob_aggt[ind_agg] = latitudexxx
                            longitude_ob_aggt[ind_agg] = longitudexxx
                            mole_frac_aggt[ind_agg] = mole_frac_xxx
                            mole_frac_precision_aggt[ind_agg] = mole_frac_precision_xxx
                            pressure_levels_aggt[:,ind_agg] = pressure_levels_xxx
                            column_averaging_kernel_aggt[:,ind_agg] = column_averaging_kernel_xxx
                            Pressure_Weighting_Function_aggt[:,ind_agg] = Pressure_Weighting_Function_xxx 
                            ind_agg = ind_agg + 1

    # Restructure array for output
    column_averaging_kernel_agg2_TEMP = np.transpose(column_averaging_kernel_aggt[:,0:ind_agg])
    column_averaging_kernel_agg2 = column_averaging_kernel_agg2_TEMP[:,::-1]
    pressure_levels_agg2_TEMP = np.transpose(pressure_levels_aggt[:,0:ind_agg]) / 100. # hPa
    pressure_levels_agg2 = pressure_levels_agg2_TEMP[:,::-1]
    Pressure_Weighting_Function_agg2_TEMP = np.transpose(Pressure_Weighting_Function_aggt[:,0:ind_agg])
    Pressure_Weighting_Function_agg2 = Pressure_Weighting_Function_agg2_TEMP[:,::-1]
    
    # Truncate to all data in the given day
    Daily_TROPOMI_superobs =  {
        'hour': hour_aggt[0:ind_agg],
        'latitude': latitude_ob_aggt[0:ind_agg],
        'longitude': longitude_ob_aggt[0:ind_agg],
        'mole_frac': mole_frac_aggt[0:ind_agg],
        'mole_frac_precision': mole_frac_precision_aggt[0:ind_agg],
        'pressure_levels': pressure_levels_agg2,
        'column_averaging_kernel': column_averaging_kernel_agg2,
        'Pressure_Weighting_Function': Pressure_Weighting_Function_agg2
    }

    return Daily_TROPOMI_superobs
    # -------------------------------------------------------------


def make_TROPOMI_obs (YEAR, MONTH_START, MONTH_END, DATA_TYPE, DATA_DIRECTORY, DIRECTORY_OUT):

    '''
    Function that calculates and writes TROPOMI super-obs over a given period

    Inputs:
      - YEAR: Year to perform operation
      - MONTH_START: Month to start operation
      - MONTH_START: Month to end operation
      - DATA_TYPE: Type of TROPOMI CO retrieval (either 'OFFL' or 'RPRO')
      - DATA_DIRECTORY: location of TROPOMI data
      - DIRECTORY_OUT: directory to write data to
        
    ################################# GENERAL APPROACH ##################################
    
      *  AVERAGING KERNAL COMES WITH UNITS OF "m" SO THAT AK*CO HAS UNIITS m*mol/m3 = mol/m2
         I HAVE CONVERTED TO PROPER UNITS BY MULTIPLYING BY "(1 / delta z)*(delta P / P_surf)",
         WHERE "delta z" WAS CALCULATED USING THE HYPSOMETRIC EQUATION
    
      *  I HAVE USED THE PRESSURE LEVELS FROM THE RETRIEVAL BUT CALCULATED MOLE FRACTION
         USING GEOS-CHEM PRESSURE. SHOULD PROBABLY THINK ABOUT WHETHER THIS IS CONSISTENT.
    
      *  I HAVE SET THE PRIOR INFO TO 0 SINCE TROPOMOI RETRIVALS EXCLUDE IMPACT OF PRIOR.
    
    ######################################################################################

    '''

    atm_const = define_constants()
    
    nc_file ='/nobackup/mlee7/GEOS/GEOS_2x2.5/MERRA2/2019/11/MERRA2.20191101.I3.2x25.nc4'
    MERRA2_grid = xr.open_dataset(nc_file)

    month_array = np.arange(MONTH_END-MONTH_START)+MONTH_START-1
    day_in_month = np.array([31,28,31,30,31,30,31,31,30,31,30,31])

    # Iterate over days
    for mth_loop in month_array:
        time_i=0
        year_old='0000'
        month_old='00'
        day_old='00'
        for day_of_month in range(day_in_month[mth_loop]):
            
            print('mth: ')
            print(mth_loop+1)
            print('day_of_montht: ')
            print(day_of_month+1)
            print(' ============= ')
            
            # Create large arrays to put data in
            hour_outt =  np.zeros(10000000)
            latitude_ob_outt =  np.zeros(10000000)
            longitude_ob_outt =  np.zeros(10000000)
            mole_frac_outt =  np.zeros(10000000)
            mole_frac_precision_outt =  np.zeros(10000000)
            pressure_levels_outt = np.zeros((49,10000000))
            column_averaging_kernel_outt = np.zeros((49,10000000))
            Pressure_Weighting_Function_outt = np.zeros((49,10000000))
                        
            ind_out = 0
            ind_agg = 0

            # Loop over TROPOMI data
            b = glob.glob(DATA_DIRECTORY+"/S5P_"+DATA_TYPE+"_L2__CO_____"+str(YEAR).zfill(2)+str(mth_loop+1).zfill(2)+str(day_of_month+1).zfill(2)+"*.nc")
            a = np.sort(b)
            for nc_file in a:
                print(nc_file)

                # Make sure data is within lat/lon grid
                f=Dataset(nc_file,mode='r')
                latitude = f.groups['PRODUCT']['latitude'][:]
                longitude = f.groups['PRODUCT']['longitude'][:]
                if is_data_within_latlon_grid(latitude, longitude, MERRA2_grid):

                    # Read data if within grid
                    TROPOMI_obs_file = {
                        'latitude': latitude,
                        'longitude': longitude,
                        'scanline': f.groups['PRODUCT']['scanline'][:],
                        'ground_pixel': f.groups['PRODUCT']['ground_pixel'][:],
                        'time': f.groups['PRODUCT']['time'][:],
                        'corner': f.groups['PRODUCT']['corner'][:],
                        'layer': f.groups['PRODUCT']['layer'][:],
                        'delta_time': f.groups['PRODUCT']['delta_time'][:],
                        'time_utc': f.groups['PRODUCT']['time_utc'][:],
                        'qa_value': f.groups['PRODUCT']['qa_value'][:],
                        'carbonmonoxide_total_column': f.groups['PRODUCT']['carbonmonoxide_total_column'][:], # mol m-2
                        'carbonmonoxide_total_column_precision': f.groups['PRODUCT']['carbonmonoxide_total_column_precision'][:],
                        'pressure_levels': f.groups['PRODUCT']['SUPPORT_DATA']['DETAILED_RESULTS']['pressure_levels'][:], # Pa  -- this is bottom of layers
                        'column_averaging_kernel': f.groups['PRODUCT']['SUPPORT_DATA']['DETAILED_RESULTS']['column_averaging_kernel'][:] * 1000., # m -- this is for layers
                        'surface_pressure': f.groups['PRODUCT']['SUPPORT_DATA']['INPUT_DATA']['surface_pressure'][:] # m
                        }

                    # Loop over scan lines
                    for scanline_i in range(np.size(TROPOMI_obs_file['scanline'])):

                        # Make sure data is within lat/lon grid
                        if is_data_within_latlon_grid(TROPOMI_obs_file['latitude'][time_i, scanline_i, :], TROPOMI_obs_file['longitude'][time_i, scanline_i, :], MERRA2_grid):

                            # Loop over ground pixel
                            for ground_pixel_i in range(np.size(TROPOMI_obs_file['ground_pixel'])):

                                # 'TROPOMI_ob' is sub-selected data for this observation
                                TROPOMI_ob = {
                                    'year': str(TROPOMI_obs_file['time_utc'][time_i,scanline_i][0:4]), # (time, scanline)
                                    'month': str(TROPOMI_obs_file['time_utc'][time_i,scanline_i][5:7]),
                                    'day': str(TROPOMI_obs_file['time_utc'][time_i,scanline_i][8:10]),
                                    'hour': int(str(TROPOMI_obs_file['time_utc'][time_i,scanline_i][11:13])),
                                    'minute': int(str(TROPOMI_obs_file['time_utc'][time_i,scanline_i][14:16]))
                                    }

                                # Ensure also the same day
                                if TROPOMI_ob['day'] == str(day_of_month+1).zfill(2):

                                    if TROPOMI_ob['year'] != str(YEAR).zfill(2):
                                        raise ValueError(f"Error: Year mismatch. Expected {str(YEAR).zfill(2)}, got {year}.")
                                    
                                    if TROPOMI_ob['month'] != str(mth_loop+1).zfill(2):
                                        raise ValueError(f"Error: Month mismatch. Expected {str(mth_loop+1).zfill(2)}, got {month}.")

                                    # Sub-select the observation given time, scaneline and ground pixel indices
                                    TROPOMI_ob = subselect_TROPOMI_data(TROPOMI_ob, TROPOMI_obs_file,time_i, scanline_i, ground_pixel_i)
                                    
                                    # Qa_value          Condition                               Remark
                                    #   1.0      T_aer<0.5 and z_cld<500m      clear-sky and clear-sky like observations
                                    #   0.7      T_aer>=0.5 and z_cld<5000m                mid-levels cloud
                                    #   0.4                                        high clouds, experimental data set
                                    #   0.0                                           corrupted or defectivedata 
                                    if TROPOMI_ob['qa_value'] > 0.5 and is_data_within_latlon_grid(TROPOMI_ob['latitude'], TROPOMI_ob['longitude'], MERRA2_grid):
                                                    
                                        if TROPOMI_ob['year'] != year_old or TROPOMI_ob['month'] != month_old or TROPOMI_ob['day'] != day_old:
                                            print('old date: '+year_old+'/'+month_old+'/'+day_old)
                                            print('Current date: '+TROPOMI_ob['year']+'/'+TROPOMI_ob['month']+'/'+TROPOMI_ob['day'])
                                            nc_file ='/nobackup/bbyrne1/MERRA2/2x2.5/'+TROPOMI_ob['year']+'/'+TROPOMI_ob['month']+'/MERRA2.'+TROPOMI_ob['year']+TROPOMI_ob['month']+TROPOMI_ob['day']+'.I3.2x25.nc4'
                                            print(nc_file)
                                            MERRA2_atm = xr.open_dataset(nc_file)
                                            MERRA2_atm['time'] = MERRA2_atm['time'].astype(float) / 60.0
                                            # Update the old date variables
                                            year_old = TROPOMI_ob['year']
                                            month_old = TROPOMI_ob['month']
                                            day_old = TROPOMI_ob['day']

                                        # Calculate AK and mole fractions for model atmosphere
                                        TROPOMI_ob_GCgrid = map_TROPOMI_ob_to_model_grid(atm_const,TROPOMI_ob,MERRA2_atm)

                                        # Append data
                                        hour_outt[ind_out] = TROPOMI_ob_GCgrid['hour']+TROPOMI_ob_GCgrid['minute']/60.
                                        latitude_ob_outt[ind_out] = TROPOMI_ob_GCgrid['latitude']
                                        longitude_ob_outt[ind_out] = TROPOMI_ob_GCgrid['longitude']
                                        mole_frac_outt[ind_out] = TROPOMI_ob_GCgrid['mole_frac']
                                        mole_frac_precision_outt[ind_out] = TROPOMI_ob_GCgrid['mole_frac_precision']
                                        pressure_levels_outt[:,ind_out] = TROPOMI_ob_GCgrid['pressure_levels_layers']
                                        column_averaging_kernel_outt[:,ind_out] = TROPOMI_ob_GCgrid['AK_convert_full']*TROPOMI_ob_GCgrid['column_averaging_kernel_layers']
                                        Pressure_Weighting_Function_outt[:,ind_out] = TROPOMI_ob_GCgrid['Pressure_Weighting_Function']
                                        ind_out = ind_out+1
            #===============================================================================================================================================

            # Truncate to all data in the given day
            Daily_TROPOMI_data_dict =  {
                'hour': hour_outt[0:ind_out],
                'latitude': latitude_ob_outt[0:ind_out],
                'longitude': longitude_ob_outt[0:ind_out],
                'mole_frac': mole_frac_outt[0:ind_out],
                'mole_frac_precision': mole_frac_precision_outt[0:ind_out],
                'pressure_levels': pressure_levels_outt[:,0:ind_out],
                'column_averaging_kernel': column_averaging_kernel_outt[:,0:ind_out],
                'Pressure_Weighting_Function': Pressure_Weighting_Function_outt[:,0:ind_out],
            }

            print("latitude shape:", Daily_TROPOMI_data_dict['latitude'].shape)
            print("Pressure levels shape:", Daily_TROPOMI_data_dict['pressure_levels'].shape)

    
            # Convert Daily_TROPOMI_data_dict to xarray Dataset for easier handling
            Daily_TROPOMI_data = xr.Dataset({key: (['time'], value) for key, value in Daily_TROPOMI_data_dict.items() if key not in ['pressure_levels', 'column_averaging_kernel', 'Pressure_Weighting_Function']})
            Daily_TROPOMI_data['pressure_levels'] = (('time', 'level'), Daily_TROPOMI_data_dict['pressure_levels'])
            Daily_TROPOMI_data['column_averaging_kernel'] = (('time', 'level'), Daily_TROPOMI_data_dict['column_averaging_kernel'])
            Daily_TROPOMI_data['Pressure_Weighting_Function'] = (('time', 'level'), Daily_TROPOMI_data_dict['Pressure_Weighting_Function'])

            print("... HOW LARGE ARE ARRAYS BEFORE AGGREGATION ...")
            print(np.shape(Daily_TROPOMI_data['mole_frac']))
            print("............................")
            
            Daily_TROPOMI_superobs = calculate_TROPOMI_superobs(Daily_TROPOMI_data, MERRA2_grid)

            print("... HOW LARGE ARE ARRAYS AFTER AGGREGATION ...")
            print(np.shape(Daily_TROPOMI_superobs['mole_frac'].values)) # 8557
            print("............................")
                                
            return Daily_TROPOMI_superobs

            # Write the data
            if np.size(latitude_ob_agg)>0:
                print('Min Lat out = ' + str(np.min(latitude_ob_agg)))
                print('Max Lat out = ' + str(np.max(latitude_ob_agg)))
                print('Min Lon out = ' + str(np.min(longitude_ob_agg)))
                print('Max Lon out = ' + str(np.max(longitude_ob_agg)))


                file_out=DIRECTORY_OUT+year+'/'+str(mth_loop+1).zfill(2)+'/'+str(day_of_month+1).zfill(2)+'.nc' #+str(day_of_month+1).zfill(2)+'.nc'
                print(file_out)
                dataset = Dataset(file_out,'w')
                nSamples = dataset.createDimension('nSamples',np.size(hour_agg))
                maxLevels = dataset.createDimension('maxLevels',np.size(pressure_levels_agg2[0,:]))
                
                longitudes = dataset.createVariable('longitude', np.float64, ('nSamples',))
                longitudes[:]=longitude_ob_agg
                
                latitudes = dataset.createVariable('latitude', np.float64, ('nSamples',))
                latitudes[:]=latitude_ob_agg
                
                modes = dataset.createVariable('mode', np.float64, ('nSamples',))
                modes[:]=hour_agg*0.
                
                times = dataset.createVariable('time', np.float64, ('nSamples',))
                times[:]=hour_agg
                
                pressures = dataset.createVariable('pressure', np.float64, ('nSamples','maxLevels'))
                pressures[:,:]=pressure_levels_agg2
                
                xCOs = dataset.createVariable('xCO', np.float64, ('nSamples',))
                xCOs[:]=mole_frac_agg
                
                xCOaprioris = dataset.createVariable('xCO-apriori', np.float64, ('nSamples',))
                xCOaprioris[:]=mole_frac_agg*0.
                
                xCOpressureWeights = dataset.createVariable('xCO-pressureWeight', np.float64, ('nSamples','maxLevels'))
                xCOpressureWeights[:,:]=Pressure_Weighting_Function_agg2
                
                xCOuncertaintys = dataset.createVariable('xCO-uncertainty', np.float64, ('nSamples',))
                xCOuncertaintys[:]=mole_frac_precision_agg
                
                xCOaveragingKernels = dataset.createVariable('xCO-averagingKernel', np.float64, ('nSamples','maxLevels'))
                xCOaveragingKernels[:,:]=column_averaging_kernel_agg2
                
                COaprioris = dataset.createVariable('CO-apriori', np.float64, ('nSamples','maxLevels'))
                COaprioris[:,:]=column_averaging_kernel_agg2*0.
                
                dataset.close()
            
                print("##########################################")
    # -------------------------------------------------------------




if __name__ == "__main__":

    directory_out = './output_8Feb22_2x25/'
    data_directory = '/nobackupp19/bbyrne1/Test_TROPOMI_download/PRODUCT/'
    data_type = 'OFFL'
    year = 2023
    month = 6
    day = 10
    make_TROPOMI_obs(year,month,day,data_type,data_directory.directory_out)
