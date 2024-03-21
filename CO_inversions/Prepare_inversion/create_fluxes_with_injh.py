import numpy as np
from netCDF4 import Dataset
import glob, os
from math import pi, cos, radians
from pylab import *

# ===================================================================================
#
#  Maps fire emissions to injection height from GFAS IS4fires
#
# ===================================================================================


def define_Ap_Bp_for_pressure():
    #
    # ==================================================
    #  Defines Ap and Bp that are needed for the vertical grid
    # ==================================================
    #
    # Need Ap and Bp for calculating pressure
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
    # Repeat the array over two new dimensions
    Ap_reshaped = Ap.reshape((73, 1, 1))  # Reshape to (72, 1, 1)
    Ap_tiled = np.tile(Ap_reshaped, (1, 91, 144))  # Tile over lat and lon dimensions
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
    Bp_reshaped = Bp.reshape((73, 1, 1))  # Reshape to (72, 1, 1)
    Bp_tiled = np.tile(Bp_reshaped, (1, 91, 144))  # Tile over lat and lon dimensions
    #
    return Ap_tiled, Bp_tiled
    # ==============================================================


def calculate_model_height(year,month,day,Ap_tiled,Bp_tiled):
    #
    # ==================================================
    #  Calculated the vertical grid in meters
    # ==================================================
    #
    # === Constants ===                     
    g = 9.8 # m/s2                          
    AIR_MW = 28./1000. # kg/mol             
    AVO = 6.022e23 # molec/mol              
    R = 287.058  # J kg-1 K-1 = m2 s-2 K-1 Specific gas constant for dry air  
    # -----------------------------------------------
    #
    nc_file ='/nobackup/bbyrne1/MERRA2/2x2.5/'+str(year).zfill(4)+'/'+str(month).zfill(2)+'/MERRA2.'+str(year).zfill(4)+str(month).zfill(2)+str(day).zfill(2)+'.I3.2x25.nc4'
    print(nc_file)
    f=Dataset(nc_file,mode='r')
    lonGC=f.variables['lon'][:]
    latGC=f.variables['lat'][:]
    PGC=f.variables['PS'][:] # surface pressue (Pa = kg m-1 s-2) 
    TGC=f.variables['T'][:] # air temperature (K)
    f.close()
    P = np.mean(PGC,0)
    P_reshaped = P.reshape((1, 91, 144))  # Reshape to (1, lat, lon)
    P_tiled = np.tile(P_reshaped, (73, 1, 1))  # Tile over the first dimension
    T = np.mean(TGC,0)
    # ==================================================
    #
    # Pedge(I,J,L) = Ap(L) + [ Bp(L) * Psurface(I,J) ]
    Pedge = (Ap_tiled*100. + ( Bp_tiled * P_tiled ))
    #
    # Hypsometric equation:                 
    # z_2 - z_1 = (R *T / g) * ln(P_1/P_2) 
    height = (R*T/g)*np.log(P_tiled[0:72,:,:]/Pedge[0:72,:,:])
    #
    return height
    # ==============================================================


def write_injh_file(year,month,day,inj_level_arr):
    #
    # ==================================================
    #  Writes out the injection level of the model atmosphere
    # ==================================================    
    #
    # Write out fluxes                                                                                                                          
    nc_out = '/u/bbyrne1/inj_level/'+str(year).zfill(4)+'/'+str(month).zfill(2)+'/'+str(day).zfill(2)+'.nc'
    print(nc_out)
    #                                                                                                                                           
    dataset = Dataset(nc_out,'w')
    #                                                                                                                                           
    lats = dataset.createDimension('lat',inj_level_arr.shape[0])
    lons = dataset.createDimension('lon',inj_level_arr.shape[1])
    #                                                                                                                                           
    injhLs = dataset.createVariable('injh_level', np.float64, ('lat','lon'))
    injhLs[:,:] = inj_level_arr
    injhLs.units = 'level'
    #                                                                                                                                           
    dataset.close()
    # ==============================================================


if __name__ == "__main__":

    # Year for calculation
    year = 2023
    # Days in each month
    days_in_months = [31,28,31,30,31,30,31,31,30,31,30,31]
    
    # Arrays giving the vertical structure (hybrid sigma-pressure)
    Ap, Bp = define_Ap_Bp_for_pressure()

    # Loop over months and days
    for month in range(4,10):
        for day in range(days_in_months[month-1]):

            # Calculate model level heights for day
            height = calculate_model_height(year,month,day+1,Ap,Bp)

            # Read in IS4fires Injection heights (in meters)
            nc_out = '/nobackup/bbyrne1/Flux_2x25_CO/BiomassBurn/injh/'+str(year).zfill(4)+'/'+str(month).zfill(2)+'/'+str(day+1).zfill(2)+'.nc'
            f = Dataset(nc_out,'r')
            injh = f.variables['injh'][:] # m
            f.close()

            # Calculate injection level given IS4fires and model heights
            injh_level = np.zeros((91,144))
            for i in range(91):
                for j in range(144):
                    injh_level[i,j] = np.argmax(injh[i,j]<=height[:,i,j])

            # Write out injection level
            write_injh_file(year,month,day+1,injh_level)
