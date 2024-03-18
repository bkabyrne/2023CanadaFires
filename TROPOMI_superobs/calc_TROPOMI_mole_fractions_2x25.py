from mpl_toolkits.basemap import Basemap, cm                                                                   
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import datetime
from netCDF4 import Dataset
import glob, os

# Function that creates L2# observation files for CMS-Flux from the TROPOMI CO L2 product.
# This version of the function is specific to the 2x2.5 version of CMS-Flux

def make_TROPOMI_obs (YEAR, MONTH_START, MONTH_END):

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
    
    # ########### USER INPUT ########### 
    
    data_directory='/nobackupp19/bbyrne1/Test_TROPOMI_download/PRODUCT/'#TROPOMI_CO_20220202/'#/nobackup/bbyrne1/TROPOMI_CO_20220202/'#/u/bbyrne1/TROPOMI_CO'#/u/bbyrne1/TROPOMI_data_Jan'#/u/bbyrne1/TROPOMI_CO'
    
    # ################################## 
    
    # === Stuff for calculating GEOS-Chem pressure ===
    #(surface)
    Ap =np.array([0.000000e+00, 4.804826e-02, 6.593752e+00, 1.313480e+01, 1.961311e+01, 2.609201e+01,
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
    
    
    
    time_i=0
    year_old='0000'
    month_old='00'
    day_old='00'
    
    nc_file ='/nobackup/mlee7/GEOS/GEOS_2x2.5/MERRA2/2019/11/MERRA2.20191101.I3.2x25.nc4'
    f=Dataset(nc_file,mode='r')
    lonGC=f.variables['lon'][:]
    latGC=f.variables['lat'][:]
    
    
    #group: PRODUCT {    
    #  dimensions:       
    #        scanline = 4172 ;
    #        ground_pixel = 215 ;
    #        corner = 4 ;
    #        time = 1 ;
    #        layer = 50 ;
    #        int scanline(scanline) ;
    #                scanline:units = "1" ;
    #                scanline:axis = "Y" ;
    #                scanline:long_name = "along-track dimension index" ;
    #                scanline:comment = "This coordinate variable defines the indices along track; index starts at 0" ;
    #        int ground_pixel(ground_pixel) ;
    #                ground_pixel:units = "1" ;
    #                ground_pixel:axis = "X" ;
    #                ground_pixel:long_name = "across-track dimension index" ;
    #                ground_pixel:comment = "This coordinate variable defines the indices across track, from west to east; index starts at 0" ;
    #        int time(time) ;
    #                time:units = "seconds since 2010-01-01 00:00:00" ;
    #                time:standard_name = "time" ;
    #                time:axis = "T" ;
    #                time:long_name = "reference time for the measurements" ;
    #                time:comment = "The time in this variable corresponds to the time in the time_reference global attribute" ;
    #        int corner(corner) ;
    #                corner:units = "1" ;
    #                corner:long_name = "pixel corner index" ;
    #                corner:comment = "This coordinate variable defines the indices for the pixel corners; index starts at 0 (counter-clockwise, starting from south-western corner of the pixel in ascending part of the orbit)" ;
    #        float layer(layer) ;
    #                layer:units = "m" ;
    #                layer:standard_name = "height" ;
    #                layer:long_name = "Height above topographic surface" ;
    #                layer:axis = "Z" ;
    #        int delta_time(time, scanline) ;
    #                delta_time:long_name = "offset of start time of measurement relative to time_reference" ;
    #                delta_time:units = "milliseconds since 2019-12-31 00:00:00" ;
    #        string time_utc(time, scanline) ;
    #                time_utc:long_name = "Time of observation as ISO 8601 date-time string" ;
    #                string time_utc:_FillValue = "" ;
    #        ubyte qa_value(time, scanline, ground_pixel) ;
    #                qa_value:long_name = "data quality value" ;
    #                qa_value:comment = "A continuous quality descriptor, varying between 0 (no data) and 1 (full quality data). Recommend to ignore data with qa_value < 0.5" ;
    #                qa_value:coordinates = "longitude latitude" ;
    #        float latitude(time, scanline, ground_pixel) ;
    #                latitude:long_name = "pixel center latitude" ;
    #                latitude:units = "degrees_north" ;
    #                latitude:standard_name = "latitude" ;
    #        float longitude(time, scanline, ground_pixel) ;
    #                longitude:long_name = "pixel center longitude" ;
    #                longitude:units = "degrees_east" ;
    #                longitude:standard_name = "longitude" ;
    #        float carbonmonoxide_total_column(time, scanline, ground_pixel) ;
    #                carbonmonoxide_total_column:units = "mol m-2" ;
    #                carbonmonoxide_total_column:standard_name = "atmosphere_mole_content_of_carbon_monoxide" ;
    #                carbonmonoxide_total_column:long_name = "Vertically integrated CO column" ;
    #                carbonmonoxide_total_column:coordinates = "longitude latitude" ;
    #                carbonmonoxide_total_column:ancillary_variables = "carbonmonoxide_total_column_precision" ;
    #        float carbonmonoxide_total_column_precision(time, scanline, ground_pixel) ;
    #                carbonmonoxide_total_column_precision:units = "mol m-2" ;
    #                carbonmonoxide_total_column_precision:standard_name = "atmosphere_mole_content_of_carbon_monoxide standard_error" ;
    #                carbonmonoxide_total_column_precision:long_name = "Standard error of the vertically integrated CO column" ;
    #                carbonmonoxide_total_column_precision:coordinates = "longitude latitude" ;
    
    #    group: DETAILED_RESULTS {
    #        float pressure_levels(time, scanline, ground_pixel, layer) ;
    #                pressure_levels:positive = "down" ;
    #                pressure_levels:units = "Pa" ;
    #                pressure_levels:standard_name = "air_pressure" ;
    #                pressure_levels:long_name = "Pressure at bottom of layer" ;
    #        float water_total_column(time, scanline, ground_pixel) ;
    #                water_total_column:units = "mol m-2" ;
    #                water_total_column:standard_name = "atmosphere_mole_content_of_water_vapor" ;
    #                water_total_column:long_name = "Vertically integrated H2O column" ;
    #                water_total_column:coordinates = "/PRODUCT/longitude /PRODUCT/latitude" ;
    #                water_total_column:ancillary_variables = "water_total_column_precision" ;
    #        float column_averaging_kernel(time, scanline, ground_pixel, layer) ;
    #                column_averaging_kernel:units = "m" ;
    #                column_averaging_kernel:long_name = "CO column averaging kernel" ;
    #                column_averaging_kernel:coordinates = "/PRODUCT/longitude /PRODUCT/latitude" ;


    month_array = np.arange(MONTH_END-MONTH_START)+MONTH_START-1
    day_in_month = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
    for mth_loop in month_array:
        #mth_loop = mth_loopt+0
        time_i=0
        year_old='0000'
        month_old='00'
        day_old='00'
        for day_of_montht in range(day_in_month[mth_loop]):
            
            print('mth: ')
            print(mth_loop)
            print('day_of_montht: ')
            print(day_of_montht)
            print(' ============= ')
            
            day_of_month = day_of_montht+0
            # Create large arrays to put data in
            year_outt = np.zeros(10000000)
            month_outt =  np.zeros(10000000)
            day_outt =  np.zeros(10000000)
            hour_outt =  np.zeros(10000000)
            minute_outt =  np.zeros(10000000)
            latitude_ob_outt =  np.zeros(10000000)
            longitude_ob_outt =  np.zeros(10000000)
            mole_frac_outt =  np.zeros(10000000)
            mole_frac_precision_outt =  np.zeros(10000000)
            pressure_levels_outt = np.zeros((49,10000000))
            column_averaging_kernel_outt = np.zeros((49,10000000))
            Pressure_Weighting_Function_outt = np.zeros((49,10000000))
            
            
            hour_aggt = np.zeros(40000)
            latitude_ob_aggt = np.zeros(40000)
            longitude_ob_aggt = np.zeros(40000)
            mole_frac_aggt = np.zeros(40000)
            mole_frac_precision_aggt = np.zeros(40000)
            pressure_levels_aggt = np.zeros((49,40000))
            column_averaging_kernel_aggt = np.zeros((49,40000))
            Pressure_Weighting_Function_aggt = np.zeros((49,40000))
            
            
            ind_out = 0
            ind_agg = 0


            # Loop over TROPOMI data
            b = glob.glob(data_directory+"/S5P_OFFL_L2__CO_____"+str(YEAR).zfill(2)+str(mth_loop+1).zfill(2)+str(day_of_month+1).zfill(2)+"*.nc")
            #b = glob.glob(data_directory+"S5P_RPRO_L2__CO_____"+str(YEAR).zfill(2)+str(mth_loop+1).zfill(2)+str(day_of_month+1).zfill(2)+"*.nc")
            a = np.sort(b)
            for nc_file in a:
                print(nc_file)
                f=Dataset(nc_file,mode='r')
                latitude = f.groups['PRODUCT']['latitude'][:]
                print('Min Lat = ' + str(np.min(latitude)))
                print('Max Lat = ' + str(np.max(latitude)))
                if np.sum(np.logical_and(latitude>np.min(latGC),latitude<np.max(latGC)))>0:
                    longitude = f.groups['PRODUCT']['longitude'][:]
                    print('Min Lon = ' + str(np.min(longitude)))
                    print('Max Lon = ' + str(np.max(longitude)))
                    if np.sum(np.logical_and(longitude>np.min(lonGC),longitude<np.max(lonGC)))>0:
                        scanline = f.groups['PRODUCT']['scanline'][:]
                        ground_pixel = f.groups['PRODUCT']['ground_pixel'][:]
                        time = f.groups['PRODUCT']['time'][:]
                        corner = f.groups['PRODUCT']['corner'][:]
                        layer = f.groups['PRODUCT']['layer'][:]
                        delta_time = f.groups['PRODUCT']['delta_time'][:]
                        time_utc = f.groups['PRODUCT']['time_utc'][:]
                        qa_value = f.groups['PRODUCT']['qa_value'][:]
                        carbonmonoxide_total_column = f.groups['PRODUCT']['carbonmonoxide_total_column'][:] # mol m-2
                        carbonmonoxide_total_column_precision = f.groups['PRODUCT']['carbonmonoxide_total_column_precision'][:]
                        pressure_levels = f.groups['PRODUCT']['SUPPORT_DATA']['DETAILED_RESULTS']['pressure_levels'][:] # Pa  ----- this is bottom of layers
                        column_averaging_kernel = f.groups['PRODUCT']['SUPPORT_DATA']['DETAILED_RESULTS']['column_averaging_kernel'][:] * 1000. # m ----- this is for layers
                        surface_pressure = f.groups['PRODUCT']['SUPPORT_DATA']['INPUT_DATA']['surface_pressure'][:] # m
                        
                        # Loop over scal lines
                        for i in range(np.size(scanline)):
                            scanline_i=i
                            latitude_ob_temp = latitude[time_i, scanline_i, :]
                            if np.sum(np.logical_and(latitude_ob_temp>np.min(latGC),latitude_ob_temp<np.max(latGC)))>0:
                                longitude_ob_temp = longitude[time_i, scanline_i, :]
                                if np.sum(np.logical_and(longitude_ob_temp>np.min(lonGC),longitude_ob_temp<np.max(lonGC)))>0:
                                    for j in range(np.size(ground_pixel)):
                                        ground_pixel_i=j
                                
                                        year = str(time_utc[time_i,scanline_i][0:4]) # (time, scanline)
                                        month = str(time_utc[time_i,scanline_i][5:7])
                                        day = str(time_utc[time_i,scanline_i][8:10])
                                        hour = int(str(time_utc[time_i,scanline_i][11:13]))
                                        minute = int(str(time_utc[time_i,scanline_i][14:16]))

                                        if day == str(day_of_month+1).zfill(2):

                                            if year != str(YEAR).zfill(2):
                                                stophere
                                            if month != str(mth_loop+1).zfill(2):
                                                stophere                                   


                                            qa_value_ob = qa_value[time_i, scanline_i, ground_pixel_i]
                                            latitude_ob = latitude[time_i, scanline_i, ground_pixel_i]
                                            longitude_ob = longitude[time_i, scanline_i, ground_pixel_i]
                                            carbonmonoxide_total_column_ob = carbonmonoxide_total_column[time_i, scanline_i, ground_pixel_i]
                                            carbonmonoxide_total_column_precision_ob = carbonmonoxide_total_column_precision[time_i, scanline_i, ground_pixel_i]
                                            surface_pressure_ob = surface_pressure[time_i, scanline_i, ground_pixel_i]
                                            pressure_levels_ob = pressure_levels.data[time_i, scanline_i, ground_pixel_i,:]
                                            pressure_levels_ob[49] = surface_pressure_ob # bottom layer truncated at surface
                                            column_averaging_kernel_ob = column_averaging_kernel.data[time_i, scanline_i, ground_pixel_i,:]
                                            
                                            pressure_levels_ob_layers = (pressure_levels_ob[0:49]+pressure_levels_ob[1:50])/2.
                                            column_averaging_kernel_ob_layers = column_averaging_kernel_ob[1:50]
                                        
                                            # Qa_value          Condition                               Remark
                                            #   1.0      T_aer<0.5 and z_cld<500m      clear-sky and clear-sky like observations
                                            #   0.7      T_aer>=0.5 and z_cld<5000m                mid-levels cloud
                                            #   0.4                                        high clouds, experimental data set
                                            #   0.0                                           corrupted or defectivedata 
                                        
                                            if qa_value_ob > 0.5:
                                                if np.logical_and(longitude_ob>np.min(lonGC),longitude_ob<np.max(lonGC)):
                                                    if np.logical_and(latitude_ob>np.min(latGC),latitude_ob<np.max(latGC)):
                                                    
                                            
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
                                                        
                                                        # === Constants ===
                                                        g = 9.8 # m/s2
                                                        AIR_MW = 28./1000. # kg/mol
                                                        AVO = 6.022e23 # molec/mol
                                                        R = 287.058  # J kg-1 K-1 = m2 s-2 K-1 # Specific gas constant for dry air
                                                    
                                                
                                                        if year!=year_old or month!=month_old or day!=day_old:
                                                            print('old date: '+year_old+'/'+month_old+'/'+day_old)
                                                            print('Current date: '+year+'/'+month+'/'+day)
                                                            nc_file ='/nobackup/bbyrne1/MERRA2/2x2.5/'+year+'/'+month+'/MERRA2.'+year+month+day+'.I3.2x25.nc4'
                                                            print(nc_file)
                                                            f=Dataset(nc_file,mode='r')
                                                            lonGC=f.variables['lon'][:]
                                                            latGC=f.variables['lat'][:]
                                                            timeGC=f.variables['time'][:]/60.
                                                            PGC=f.variables['PS'][:] # surface pressue (Pa = kg m-1 s-2)
                                                            qGC=f.variables['QV'][:] # specific humidity ( kg / kg )
                                                            TGC=f.variables['T'][:] # air temperature (K) 
                                                            year_old=year
                                                            month_old=month
                                                            day_old=day
                                                        
                                                        lon_ind = np.argmin(np.abs(lonGC-longitude_ob))
                                                        lat_ind = np.argmin(np.abs(latGC-latitude_ob))
                                                        time_ind = int(np.floor(hour/3.))
                                                    
                                                        P = PGC[time_ind,lat_ind,lon_ind]
                                                    
                                                        q_prof = qGC[time_ind,:,lat_ind,lon_ind]
                                                        
                                                        T = TGC[time_ind,:,lat_ind,lon_ind]
                                                        #T_mean = (T[0:72]+T[1:73])/2.
                                                    
                                                        # NEED TO FIX. P is Pa but Ap is hPa
                                                        Pedge = (Ap*100. + ( Bp * P )) # Ap is in hPa
                                                        Pmid = (Pedge[0:72]+Pedge[1:73])/2.
                                                    
                                                        # ----- Need to interpolate based on TROPOMI pressure being bottom of levels. ----
                                                        T_rev = T[::-1]
                                                        q_rev = q_prof[::-1]
                                                        P_rev = Pmid[::-1]
                                                        
                                                        T_interp=np.interp(pressure_levels_ob_layers, P_rev, T_rev)
                                                        q_interp=np.interp(pressure_levels_ob_layers, P_rev, q_rev)
                                                        
                                                        
                                                        # ----- Calculate mole dry air per layer ----
                                                        delta_P = pressure_levels_ob[1:50]-pressure_levels_ob[0:49]
                                                        P_dry_profile = delta_P*(1-q_interp)
                                                        P_dry = np.sum(P_dry_profile)
                                                        m_dry_per_A = P_dry / g                            
                                                        # mol = kg / AIR_MW = kg / (kg/mol)
                                                        mol_dry_per_A = m_dry_per_A / AIR_MW # mol/m2
                                                        
                                                        
                                                        mole_frac = carbonmonoxide_total_column_ob / mol_dry_per_A
                                                        mole_frac_precision = carbonmonoxide_total_column_precision_ob / mol_dry_per_A
                                                        
                                                        # PRESSURE WEIGHTING FUNCTION
                                                        Pressure_Weighting_Function = (P_dry_profile)/P_dry
                                                        
                                                        # delta h -- use hypsometric equation
                                                        delta_h = (R*T_interp/g)*np.log(pressure_levels_ob[1:50]/pressure_levels_ob[0:49])
                                                        # convert using 1/delta_h * delta_mol / total_mol
                                                        AK_convert = 1/delta_h * (P_dry_profile)/P_dry
                                                        AK_convert_full = AK_convert
                                                    
                                                        year_outt[ind_out] = year
                                                        month_outt[ind_out] = month
                                                        day_outt[ind_out] = day
                                                        hour_outt[ind_out] = hour+minute/60.
                                                        minute_outt[ind_out] = minute
                                                        latitude_ob_outt[ind_out] = latitude_ob
                                                        longitude_ob_outt[ind_out] = longitude_ob
                                                        mole_frac_outt[ind_out] = mole_frac
                                                        mole_frac_precision_outt[ind_out] = mole_frac_precision
                                                        pressure_levels_outt[:,ind_out] = pressure_levels_ob_layers
                                                        column_averaging_kernel_outt[:,ind_out] = AK_convert_full*column_averaging_kernel_ob_layers
                                                        Pressure_Weighting_Function_outt[:,ind_out] = Pressure_Weighting_Function
                                                        ind_out = ind_out+1
            #
                                                    
            year_out=year_outt[0:ind_out]
            month_out= month_outt[0:ind_out]
            day_out=day_outt[0:ind_out]
            hour_out=hour_outt[0:ind_out]
            minute_out=minute_outt[0:ind_out]
            latitude_ob_out=latitude_ob_outt[0:ind_out]
            longitude_ob_out=longitude_ob_outt[0:ind_out]
            mole_frac_out=mole_frac_outt[0:ind_out]
            mole_frac_precision_out=mole_frac_precision_outt[0:ind_out]
            pressure_levels_out=pressure_levels_outt[:,0:ind_out]
            column_averaging_kernel_out=column_averaging_kernel_outt[:,0:ind_out]
            Pressure_Weighting_Function_out=Pressure_Weighting_Function_outt[:,0:ind_out]
            
            
            print("... HOW LARGE ARE ARRAYS BEFORE AGGREGATION ...")
            print(np.shape(mole_frac_out))
            print("............................")
            
            
            XCO_map = np.zeros((np.size(lonGC),np.size(latGC)))
                                                
                                                
            n=0
            hour_day = np.arange(24)+0.5
            for i in range(np.size(lonGC)):
                IND = np.where(np.logical_and(longitude_ob_out>=lonGC[i]-2.5/2.,longitude_ob_out<lonGC[i]+2.5/2.))
                if np.size(IND)>0:
                    latitude_ob_outx = latitude_ob_out[IND]
                    hour_outx = hour_out[IND]
                    mole_frac_outx = mole_frac_out[IND]
                    mole_frac_precision_outx = mole_frac_precision_out[IND]
                    column_averaging_kernel_outx = column_averaging_kernel_out[:,IND[0]]
                    pressure_levels_outx = pressure_levels_out[:,IND[0]]
                    Pressure_Weighting_Function_outx = Pressure_Weighting_Function_out[:,IND[0]]
                    for j in range(np.size(latGC)):
                        IND = np.where(np.logical_and(latitude_ob_outx>=latGC[j]-2./2.,latitude_ob_outx<latGC[j]+2./2.))
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
                                    longitudexxx = lonGC[i]
                                    latitudexxx = latGC[j]
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

            hour_agg=hour_aggt[0:ind_agg]
            latitude_ob_agg=latitude_ob_aggt[0:ind_agg]
            longitude_ob_agg=longitude_ob_aggt[0:ind_agg]
            mole_frac_agg=mole_frac_aggt[0:ind_agg]
            mole_frac_precision_agg=mole_frac_precision_aggt[0:ind_agg]
            pressure_levels_agg=pressure_levels_aggt[:,0:ind_agg]
            column_averaging_kernel_agg=column_averaging_kernel_aggt[:,0:ind_agg]
            Pressure_Weighting_Function_agg=Pressure_Weighting_Function_aggt[:,0:ind_agg]

                            
            print("... HOW LARGE ARE ARRAYS AFTER AGGREGATION ...")
            print(np.shape(mole_frac_agg)) # 8557
            print("............................")
                                
                                
            column_averaging_kernel_agg2_TEMP = np.transpose(column_averaging_kernel_agg)
            pressure_levels_agg2_TEMP = np.transpose(pressure_levels_agg) / 100. # hPa
            Pressure_Weighting_Function_agg2_TEMP = np.transpose(Pressure_Weighting_Function_agg)
            
            column_averaging_kernel_agg2 = column_averaging_kernel_agg2_TEMP[:,::-1]
            pressure_levels_agg2 = pressure_levels_agg2_TEMP[:,::-1]
            Pressure_Weighting_Function_agg2 = Pressure_Weighting_Function_agg2_TEMP[:,::-1]


            if np.size(latitude_ob_agg)>0:
                print('Min Lat out = ' + str(np.min(latitude_ob_agg)))
                print('Max Lat out = ' + str(np.max(latitude_ob_agg)))
                print('Min Lon out = ' + str(np.min(longitude_ob_agg)))
                print('Max Lon out = ' + str(np.max(longitude_ob_agg)))
                
                file_out='./output_8Feb22_2x25/'+year+'/'+str(mth_loop+1).zfill(2)+'/'+str(day_of_month+1).zfill(2)+'.nc' #+str(day_of_month+1).zfill(2)+'.nc'
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
