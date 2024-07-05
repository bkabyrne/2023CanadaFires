from mpl_toolkits.basemap import Basemap, cm                                                                   
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import datetime
from netCDF4 import Dataset
import glob, os

def find_matching_files(directory, date):
    """
    Finds files that contain retrievals occuring on given date
    """

    matching_files = []

    # Define the directory for the given date
    dir1 = os.path.join(directory, date.strftime('%Y'), date.strftime('%m'), date.strftime('%d'))
    
    # Add all files from dir1 (guaranteed to meet the criteria)
    if os.path.exists(dir1):
        for file_name in os.listdir(dir1):
            matching_files.append(os.path.join(dir1, file_name))
    
    # Define the directory for the previous day
    date_before = date - datetime.timedelta(days=1)
    dir2 = os.path.join(directory, date_before.strftime('%Y'), date_before.strftime('%m'), date_before.strftime('%d'))
    
    # Iterate over all files in dir2
    if os.path.exists(dir2):
        for root, _, files in os.walk(dir2):
            for file_name in files:

                # Check substring at index 35
                if len(file_name) >= 43:  # Ensure the filename is long enough
                    date_str_35 = file_name[36:44]

                    # Check if either substring matches the given date
                    if date_str_35 == date.strftime('%Y%m%d'):
                        matching_files.append(os.path.join(root, file_name))

    return matching_files

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


def within_bounds(data, bounds):
    """
    Check if any values in the data fall within the specified bounds.
        
    Parameters:
    data (array-like): Array of data to check.
    bounds (array-like): Array of bounds to check against.
        
    Returns:
    bool: True if any values fall within the bounds, False otherwise.
    """
    return np.any((data >= np.min(bounds)) & (data <= np.max(bounds)))


def check_and_extract_data(nc_file, latGC, lonGC):
    """
    Check if latitude and longitude in the NetCDF file fall within the specified bounds and extract relevant data.
    
    Parameters:
    nc_file (str): Path to the NetCDF file.
    latGC (array-like): Array of latitude bounds for checking.
    lonGC (array-like): Array of longitude bounds for checking.
    
    Returns:
    dict: A dictionary containing extracted data if conditions are met, otherwise None.
    """
    
    with Dataset(nc_file, mode='r') as f:
        # Access latitude and longitude once
        product_group = f['PRODUCT']
        latitude = product_group['latitude'][:]
        longitude = product_group['longitude'][:]

        print(f'Min Lat = {np.min(latitude)}')
        print(f'Max Lat = {np.max(latitude)}')
        print(f'Min Lon = {np.min(longitude)}')
        print(f'Max Lon = {np.max(longitude)}')

        # Check bounds for both latitude and longitude
        if not (within_bounds(latitude, latGC) and within_bounds(longitude, lonGC)):
            return None

        # Extract relevant data if conditions are met
        support_data_group = product_group['SUPPORT_DATA']
        detailed_results_group = support_data_group['DETAILED_RESULTS']
        input_data_group = support_data_group['INPUT_DATA']

        data = {
            'latitude': latitude,
            'longitude': longitude,
            'scanline': product_group['scanline'][:],
            'ground_pixel': product_group['ground_pixel'][:],
            'time': product_group['time'][:],
            'corner': product_group['corner'][:],
            'layer': product_group['layer'][:],
            'delta_time': product_group['delta_time'][:],
            'time_utc': product_group['time_utc'][:],
            'qa_value': product_group['qa_value'][:],
            'carbonmonoxide_total_column': product_group['carbonmonoxide_total_column'][:],
            'carbonmonoxide_total_column_precision': product_group['carbonmonoxide_total_column_precision'][:],
            'pressure_levels': detailed_results_group['pressure_levels'][:],
            'column_averaging_kernel': detailed_results_group['column_averaging_kernel'][:] * 1000.,
            'surface_pressure': input_data_group['surface_pressure'][:],
        }
        
        return data

def calculate_retrieval_molefraction(MERRA2,observation_data,hour,minute,atm_const,array_out,ind_out):
    """
    This function calculates the mole fraction and averaging kernel from the TROPOMI data using MERRA2 reanalysis

    Parameters:
    - MERRA2: Dictionary with MERRA2 reanalysis for the day
    - observation_data: Dictorionary with TROPOMI CO retrieval
    - hour: hour of the day (UTC)
    - minute: minute of the hour (UTC)
    - atm_const: atmospheric constants
    - array_out: Dictionary containing the converted retrievals
    - ind_out: Current index to put new retrieval in array_out
        
    Returns:
    - array_out: Dictionary containing the converted retrievals
    """

    
    # ======= Equations ======= 
    #
    # Hydrostatin Balance:
    #   m * g = P * A
    #
    # Specific humidity
    #   m_dry = m_tot * ( 1 - q )
    #
    # Hypsometric equation:
    # z_2 - z_1 = (R *T / g) * ln(P_1/P_2) 
    #
    # Therefore,
    #   m_dry / A = P * ( 1 - q) / g
    #
    
    lon_ind = np.argmin(np.abs(MERRA2['lon']-observation_data['longitude']))
    lat_ind = np.argmin(np.abs(MERRA2['lat']-observation_data['latitude']))
    time_ind = int(np.floor(hour/3.))
    
    P = MERRA2['PS'][time_ind,lat_ind,lon_ind]
    q_prof = MERRA2['QV'][time_ind,:,lat_ind,lon_ind]
    T = MERRA2['T'][time_ind,:,lat_ind,lon_ind]
    
    # NEED TO FIX. P is Pa but atm_const['Ap'] is hPa
    Pedge = (atm_const['Ap']*100. + ( atm_const['Bp'] * P )) # atm_const['Ap'] is in hPa
    Pmid = (Pedge[0:72]+Pedge[1:73])/2.
    
    # ----- Need to interpolate based on TROPOMI pressure being bottom of levels. ----
    T_rev = T[::-1]
    q_rev = q_prof[::-1]
    P_rev = Pmid[::-1]
    
    T_interp=np.interp(observation_data['pressure_levels_layers'], P_rev, T_rev)
    q_interp=np.interp(observation_data['pressure_levels_layers'], P_rev, q_rev)
    
    
    # ----- Calculate mole dry air per layer ----
    delta_P = observation_data['pressure_levels'][1:50]-observation_data['pressure_levels'][0:49]
    P_dry_profile = delta_P*(1-q_interp)
    P_dry = np.sum(P_dry_profile)
    m_dry_per_A = P_dry / atm_const['gravity']                            
    # mol = kg / atm_const['AIR_MW'] = kg / (kg/mol)
    mol_dry_per_A = m_dry_per_A / atm_const['AIR_MW'] # mol/m2
    
    
    mole_frac = observation_data['carbonmonoxide_total_column'] / mol_dry_per_A
    mole_frac_precision = observation_data['carbonmonoxide_total_column_precision'] / mol_dry_per_A
    
    # PRESSURE WEIGHTING FUNCTION
    Pressure_Weighting_Function = (P_dry_profile)/P_dry
    
    # delta h -- use hypsometric equation
    delta_h = (atm_const['R_dryair']*T_interp/atm_const['gravity']) * np.log(observation_data['pressure_levels'][1:50]/observation_data['pressure_levels'][0:49])
    # convert using 1/delta_h * delta_mol / total_mol
    AK_convert = 1/delta_h * (P_dry_profile)/P_dry
    AK_convert_full = AK_convert
    
    array_out['hour'][ind_out] = hour+minute/60.
    array_out['lat'][ind_out] = observation_data['latitude']
    array_out['lon'][ind_out] = observation_data['longitude']
    array_out['mole_frac'][ind_out] = mole_frac
    array_out['mole_frac_precision'][ind_out] = mole_frac_precision
    array_out['pressure_levels'][:,ind_out] = observation_data['pressure_levels_layers']
    array_out['column_averaging_kernel'][:,ind_out] = AK_convert_full*observation_data['column_averaging_kernel_layers']
    array_out['Pressure_Weighting_Function'][:,ind_out] = Pressure_Weighting_Function

    return array_out
    
def process_retrievals(matching_files, MERRA2, atm_const, date):
    """
    Converts all TROPOMI retrievals for given day into mole fractions using the calculate_retrieval_molefraction function

    Parameters:
    - matching_files: list of files containing TROPOMI retrievals for the given day
    - MERRA2: Dictionary with MERRA2 reanalysis for the day
    - atm_const: atmospheric constants
    - date: date to perform analysis for
        
    Returns:
    - array_out: Dictionary containing the converted retrievals for the enture day
    
    """
    
    # Create large arrays to put data in
    array_out = {
        'hour': np.zeros(10000000),
        'lat': np.zeros(10000000),
        'lon': np.zeros(10000000),
        'mole_frac': np.zeros(10000000),
        'mole_frac_precision': np.zeros(10000000),
        'pressure_levels': np.zeros((49,10000000)),
        'column_averaging_kernel': np.zeros((49,10000000)),
        'Pressure_Weighting_Function': np.zeros((49,10000000))
    }
    
    print(' --- Files for '+date.strftime('%Y%m%d')+' --- ')
    print(matching_files)
    print(' --------------------------------------------- ')
    
    # Calculate mole fractions for all data in given day
    ind_out = 0 # retrieval counter
    for nc_file in matching_files:
        
        print(nc_file)
        TROPOMI_data = check_and_extract_data(nc_file, MERRA2['lat'], MERRA2['lon'])
        
        if TROPOMI_data:    
            # Loop over scal lines
            for scanline_i in TROPOMI_data['scanline']:
                latitude_ob_temp = TROPOMI_data['latitude'][0, scanline_i, :]
                longitude_ob_temp = TROPOMI_data['longitude'][0, scanline_i, :]
                if (within_bounds(latitude_ob_temp, MERRA2['lat']) and within_bounds(longitude_ob_temp, MERRA2['lon'])):
                    
                    for ground_pixel_i in TROPOMI_data['ground_pixel']:
                        
                        day = str(TROPOMI_data['time_utc'][0,scanline_i][8:10])
                        hour = int(str(TROPOMI_data['time_utc'][0,scanline_i][11:13]))
                        minute = int(str(TROPOMI_data['time_utc'][0,scanline_i][14:16]))
                        
                        if day == date.strftime('%d'):
                            
                            observation_data = {
                                'qa_value': TROPOMI_data['qa_value'][0, scanline_i, ground_pixel_i],
                                'latitude': TROPOMI_data['latitude'][0, scanline_i, ground_pixel_i],
                                'longitude': TROPOMI_data['longitude'][0, scanline_i, ground_pixel_i],
                                'carbonmonoxide_total_column': TROPOMI_data['carbonmonoxide_total_column'][0, scanline_i, ground_pixel_i],
                                'carbonmonoxide_total_column_precision': TROPOMI_data['carbonmonoxide_total_column_precision'][0, scanline_i, ground_pixel_i],
                                'surface_pressure': TROPOMI_data['surface_pressure'][0, scanline_i, ground_pixel_i],
                                'pressure_levels': TROPOMI_data['pressure_levels'].data[0, scanline_i, ground_pixel_i, :],
                                'column_averaging_kernel': TROPOMI_data['column_averaging_kernel'].data[0, scanline_i, ground_pixel_i, :]
                            }
                            
                            # Adjusting the pressure_levels to account for surface pressure
                            observation_data['pressure_levels'][49] = observation_data['surface_pressure'] # bottom layer truncated at surface
                            
                            # Calculating pressure levels and column averaging kernel for layers
                            observation_data['pressure_levels_layers'] = (observation_data['pressure_levels'][0:49] + observation_data['pressure_levels'][1:50]) / 2
                            observation_data['column_averaging_kernel_layers'] = observation_data['column_averaging_kernel'][1:50]
                            
                            # Qa_value          Condition                               Remark
                            #   1.0      T_aer<0.5 and z_cld<500m      clear-sky and clear-sky like observations
                            #   0.7      T_aer>=0.5 and z_cld<5000m                mid-levels cloud
                            #   0.4                                        high clouds, experimental data set
                            #   0.0                                           corrupted or defectivedata 
                            
                            if observation_data['qa_value'] > 0.5:
                                if (within_bounds(observation_data['latitude'], MERRA2['lat']) and within_bounds(observation_data['longitude'], MERRA2['lon'])):
                                    
                                    array_out = calculate_retrieval_molefraction(MERRA2,observation_data,hour,minute,atm_const,array_out,ind_out)
                                    ind_out = ind_out+1
                                        
                                        #
                                            
    array_out['hour']=array_out['hour'][0:ind_out]
    array_out['lat']=array_out['lat'][0:ind_out]
    array_out['lon']=array_out['lon'][0:ind_out]
    array_out['mole_frac']=array_out['mole_frac'][0:ind_out]
    array_out['mole_frac_precision']=array_out['mole_frac_precision'][0:ind_out]
    array_out['pressure_levels']=array_out['pressure_levels'][:,0:ind_out]
    array_out['column_averaging_kernel']=array_out['column_averaging_kernel'][:,0:ind_out]
    array_out['Pressure_Weighting_Function']=array_out['Pressure_Weighting_Function'][:,0:ind_out]
    
    return array_out



def calculate_superobs(MERRA2,daily_retrievals):
    """
    Calculates TROPOMI XCO super-obs at 2 x 2.5 spatial resolution

    Parameters:
    - MERRA2: Dictionary with MERRA2 reanalysis for the day
    - daily_retrievals: Dictionary containing the converted retrievals for the entire day

    Returns:
    - data_out: Dictionary of super-obs
    """
    
    XCO_map = np.zeros((np.size(MERRA2['lon']),np.size(MERRA2['lat'])))
    
    hour_aggt = np.zeros(40000)
    latitude_ob_aggt = np.zeros(40000)
    longitude_ob_aggt = np.zeros(40000)
    mole_frac_aggt = np.zeros(40000)
    mole_frac_precision_aggt = np.zeros(40000)
    pressure_levels_aggt = np.zeros((49,40000))
    column_averaging_kernel_aggt = np.zeros((49,40000))
    Pressure_Weighting_Function_aggt = np.zeros((49,40000))    
    ind_agg = 0
    
    n=0
    hour_day = np.arange(24)+0.5
    for i in range(np.size(MERRA2['lon'])):
        IND = np.where(np.logical_and(daily_retrievals['lon']>=MERRA2['lon'][i]-2.5/2.,daily_retrievals['lon']<MERRA2['lon'][i]+2.5/2.))
        if np.size(IND)>0:
            latitude_ob_outx = daily_retrievals['lat'][IND]
            hour_outx = daily_retrievals['hour'][IND]
            mole_frac_outx = daily_retrievals['mole_frac'][IND]
            mole_frac_precision_outx = daily_retrievals['mole_frac_precision'][IND]
            column_averaging_kernel_outx = daily_retrievals['column_averaging_kernel'][:,IND[0]]
            pressure_levels_outx = daily_retrievals['pressure_levels'][:,IND[0]]
            Pressure_Weighting_Function_outx = daily_retrievals['Pressure_Weighting_Function'][:,IND[0]]
            for j in range(np.size(MERRA2['lat'])):
                IND = np.where(np.logical_and(latitude_ob_outx>=MERRA2['lat'][j]-2./2.,latitude_ob_outx<MERRA2['lat'][j]+2./2.))
                if np.size(IND)>0:
                    mole_frac_outxx = mole_frac_outx[IND]
                    mole_frac_precision_outxx = mole_frac_precision_outx[IND]
                    hour_outxx = hour_outx[IND]
                    column_averaging_kernel_outxx = column_averaging_kernel_outx[:,IND[0]]
                    pressure_levels_outxx = pressure_levels_outx[:,IND[0]]
                    Pressure_Weighting_Function_outxx = Pressure_Weighting_Function_outx[:,IND[0]]
                    XCO_map[i,j] = np.mean(mole_frac_outxx)
                    for k in range(np.size(hour_day)):
                        IND = np.where(np.logical_and(hour_outxx>=hour_day[k]-0.5,hour_outxx<hour_day[k]+0.5))
                        if np.size(IND)>0:
                            longitudexxx = MERRA2['lon'][i]
                            latitudexxx = MERRA2['lat'][j]
                            hourxxx = hour_day[k]
                            mole_frac_outxxx = np.mean(mole_frac_outxx[IND])
                            mole_frac_precision_outxxx = np.mean(mole_frac_precision_outxx[IND])
                            column_averaging_kernel_outxxx = np.mean(column_averaging_kernel_outxx[:,IND[0]],1)
                            pressure_levels_outxxx = np.mean(pressure_levels_outxx[:,IND[0]],1)
                            Pressure_Weighting_Function_outxxx = np.mean(Pressure_Weighting_Function_outxx[:,IND[0]],1)
                            n=n+1
                            
                            hour_aggt[ind_agg] = hourxxx
                            latitude_ob_aggt[ind_agg] = latitudexxx
                            longitude_ob_aggt[ind_agg] = longitudexxx
                            mole_frac_aggt[ind_agg] = mole_frac_outxxx
                            mole_frac_precision_aggt[ind_agg] = mole_frac_precision_outxxx
                            pressure_levels_aggt[:,ind_agg] = pressure_levels_outxxx
                            column_averaging_kernel_aggt[:,ind_agg] = column_averaging_kernel_outxxx#/np.sum(column_averaging_kernel_outxxx)
                            Pressure_Weighting_Function_aggt[:,ind_agg] = Pressure_Weighting_Function_outxxx 
                            ind_agg = ind_agg + 1
                            
    data_out = {
        'hour': hour_aggt[0:ind_agg],
        'lat': latitude_ob_aggt[0:ind_agg],
        'lon': longitude_ob_aggt[0:ind_agg],
        'mole_frac': mole_frac_aggt[0:ind_agg],
        'mole_frac_precision': mole_frac_precision_aggt[0:ind_agg],
        'pressure_levels': pressure_levels_aggt[:,0:ind_agg],
        'column_averaging_kernel': column_averaging_kernel_aggt[:,0:ind_agg],
        'Pressure_Weighting_Function': Pressure_Weighting_Function_aggt[:,0:ind_agg]
    }
    
    return data_out
    

def write_superobs(superobs,dir_out,date):
    """
    Writes the TROPOMI super-obs to an L2# file for assimilation

    Parameters:
    - superobs: Dictionary of super-obs
    - dir_out: directory to write data
    - date: date of observations
    """
    column_averaging_kernel_agg2_TEMP = np.transpose(superobs['column_averaging_kernel'])
    pressure_levels_agg2_TEMP = np.transpose(superobs['pressure_levels']) / 100. # hPa
    Pressure_Weighting_Function_agg2_TEMP = np.transpose(superobs['Pressure_Weighting_Function'])
    
    column_averaging_kernel_agg2 = column_averaging_kernel_agg2_TEMP[:,::-1]
    pressure_levels_agg2 = pressure_levels_agg2_TEMP[:,::-1]
    Pressure_Weighting_Function_agg2 = Pressure_Weighting_Function_agg2_TEMP[:,::-1]
    
    file_out=dir_out+date.strftime('%Y')+'/'+date.strftime('%m')+'/'+date.strftime('%d')+'.nc' 
    print(file_out)
    dataset = Dataset(file_out,'w')
    nSamples = dataset.createDimension('nSamples',np.size(superobs['hour']))
    maxLevels = dataset.createDimension('maxLevels',np.size(pressure_levels_agg2[0,:]))
    
    longitudes = dataset.createVariable('longitude', np.float64, ('nSamples',))
    longitudes[:]=superobs['lon']
    
    latitudes = dataset.createVariable('latitude', np.float64, ('nSamples',))
    latitudes[:]=superobs['lat']
    
    modes = dataset.createVariable('mode', np.float64, ('nSamples',))
    modes[:]=superobs['hour']*0.
    
    times = dataset.createVariable('time', np.float64, ('nSamples',))
    times[:]=superobs['hour']
    
    pressures = dataset.createVariable('pressure', np.float64, ('nSamples','maxLevels'))
    pressures[:,:]=pressure_levels_agg2
    
    xCOs = dataset.createVariable('xCO', np.float64, ('nSamples',))
    xCOs[:]=superobs['mole_frac']
    
    xCOaprioris = dataset.createVariable('xCO-apriori', np.float64, ('nSamples',))
    xCOaprioris[:]=superobs['mole_frac']*0.
    
    xCOpressureWeights = dataset.createVariable('xCO-pressureWeight', np.float64, ('nSamples','maxLevels'))
    xCOpressureWeights[:,:]=Pressure_Weighting_Function_agg2
    
    xCOuncertaintys = dataset.createVariable('xCO-uncertainty', np.float64, ('nSamples',))
    xCOuncertaintys[:]=superobs['mole_frac_precision']
    
    xCOaveragingKernels = dataset.createVariable('xCO-averagingKernel', np.float64, ('nSamples','maxLevels'))
    xCOaveragingKernels[:,:]=column_averaging_kernel_agg2
    
    COaprioris = dataset.createVariable('CO-apriori', np.float64, ('nSamples','maxLevels'))
    COaprioris[:,:]=column_averaging_kernel_agg2*0.
    
    dataset.close()


def make_TROPOMI_obs (data_directory, dir_out, date):
    """
    Function that creates L2# observation files for CMS-Flux from the TROPOMI CO L2 product for a given date
    This version of the function is specific to the 2x2.5 version of CMS-Flux.
    """

    ################################## GENERAL APPROACH ##################################
    #
    #  *  AVERAGING KERNAL COMES WITH UNITS OF "m" SO THAT AK*CO HAS UNIITS m*mol/m3 = mol/m2
    #     I HAVE TRIED TO CONVERT TO PROPER UNITS BY MULTIPLYING BY "(1 / delta z)*(delta P / P_surf)",
    #     WHERE "delta z" WAS CALCULATED USING THE HYPSOMETRIC EQUATION
    #
    #  *  I HAVE USED THE PRESSURE LEVELS FROM THE RETRIEVAL BUT CALCULATED MOLE FRACTION
    #     USING GEOS-CHEM PRESSURE. SHOULD PROBABLY THINK ABOUT WHETHER THIS IS CONSISTENT.
    #
    #  *  I HAVE SET THE PRIOR INFO TO 0 SINCE TROPOMOI RETRIVALS EXCLUDE IMPACT OF PRIOR.
    #
    #######################################################################################
    
    atm_const = define_constants()

    # Read MERRA2 data
    nc_file ='/nobackup/bbyrne1/MERRA2/2x2.5/'+date.strftime('%Y')+'/'+date.strftime('%m')+'/MERRA2.'+date.strftime('%Y')+date.strftime('%m')+date.strftime('%d')+'.I3.2x25.nc4'
    print(nc_file)
    f=Dataset(nc_file,mode='r')
    MERRA2 = {
        'lat': f.variables['lat'][:],
        'lon': f.variables['lon'][:],
        'time': f.variables['time'][:]/60.,
        'PS': f.variables['PS'][:], # surface pressue (Pa = kg m-1 s-2)
        'QV': f.variables['QV'][:], # specific humidity ( kg / kg )
        'T': f.variables['T'][:] # air temperature (K) 
    }

    # Find observations for given date
    matching_files = find_matching_files(data_directory, date)

    # Read all retrievals for day and convert to mole-fractions
    daily_retrievals = process_retrievals(matching_files, MERRA2, atm_const, date)
            
    print("... HOW LARGE ARE ARRAYS BEFORE AGGREGATION ...")
    print(np.shape(daily_retrievals['lat']))
    print("............................")

    # Calculate super-obs at 2 x 2.5 degree resolution
    superobs = calculate_superobs(MERRA2,daily_retrievals)
    
    print("... HOW LARGE ARE ARRAYS AFTER AGGREGATION ...")
    print(np.shape(superobs['lat'])) # 8557
    print("............................")
    
    # Write super-obs (if any exist)    
    if np.size(superobs['lat'])>0:
        write_superobs(superobs,dir_out,date)


if __name__ == "__main__":
    
    input_date = datetime.datetime.strptime("20221231", "%Y%m%d")
    data_directory='/nobackupp19/bbyrne1/Test_TROPOMI_download/PRODUCT/'
    dir_out='test/'

    make_TROPOMI_obs(data_directory,dir_out,input_date)
