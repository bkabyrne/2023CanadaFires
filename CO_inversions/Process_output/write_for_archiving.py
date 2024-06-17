import xarray as xr
import pandas as pd
import numpy as np
import os

'''

This code archives the CO and CO2 flux emission estimates for Byrne et al. (2024) in a user friendly format.

'''


def create_xarray(ds, time):
    dsarr = xr.Dataset(
        {
            'grid_area': (['latitude', 'longitude'], ds['grid_area'].values)
        },
        coords={
            'time': time[0:365],
            'latitude': ds['latitude'].values,
            'longitude': ds['longitude'].values
        }
    )
    dsarr['grid_area'].attrs['units'] = 'm2'
    return dsarr

def create_xarray_monte_carlo(ds, time):
    dsarr = xr.Dataset(
        {
            'grid_area': (['latitude', 'longitude'], ds['grid_area'].values)
        },
        coords={
            'ens_member': np.arange(40)+1,
            'time': time[0:365],
            'latitude': ds['latitude'].values,
            'longitude': ds['longitude'].values
        }
    )
    dsarr['grid_area'].attrs['units'] = 'm2'
    return dsarr

def write_archived_data(year):
    
    Rep_all = ['', '_rep']
    Rep_clear_all = ['noTROPOMIrepUnc', 'TROPOMIrepUnc']
    Rep_all_long = ['TROPOMI XCO uncertainties without representativeness errors', 'TROPOMI XCO uncertainties with representativeness errors']
    Prior_all = ['GFED', 'GFAS', 'QFED']
    opt_all = ['3day', '7day']
    opt_all_long = ['near constant uncertainty and 3 day optimization', '200% uncertainty and 7 day optimization']
    
    first_iteration = True
    base_dir = '/u/bbyrne1/python_codes/Canada_Fires_2023/Byrne_etal_codes/plot_figures/data_for_figures/'
    
    nn = 1
    for j, Rep in enumerate(Rep_all):
        for Prior in Prior_all:
            for k, opt in enumerate(opt_all):
                nc_file = base_dir + f'TROPOMI{Rep}_{Prior}_COinv_2x25_{year}_fire_{opt}.nc'
                if not os.path.exists(nc_file):
                    print(f"File does not exist: {nc_file}")
                    continue

                print(f"Processing file: {nc_file}")
                ds = xr.open_dataset(nc_file)
                ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})
                    
                if first_iteration:
                    time = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
                    ds_all_CO2 = create_xarray(ds, time)
                    ds_all_CO = create_xarray(ds, time)
                    first_iteration = False

                ds_all_CO2[f'Prior_{Prior}'] = ds['CO2_prior']
                ds_all_CO2[f'Prior_{Prior}'].attrs = {
                    'units': 'gC m-2 day-1',
                    'description': f'Prior fire CO2 flux estimated by {Prior}'
                }
                ds_all_CO2[f'Posterior_exp{str(nn).zfill(2)}'] = ds['CO2_post']
                ds_all_CO2[f'Posterior_exp{str(nn).zfill(2)}'].attrs = {
                    'units': 'gC m-2 day-1',
                    'long_name': f'Posterior_{Prior}_{opt}_{Rep_clear_all[j]}',
                    'description': (
                        'Posterior fire CO2 flux estimated by a TROPOMI XCO inversion, with the following settings:\n'
                        f'- Prior fire emissions from {Prior}\n'
                        f'- {opt_all_long[k]}\n'
                        f'- {Rep_all_long[j]}'
                    )
                }
                ds_all_CO[f'Prior_{Prior}'] = ds['CO_prior']
                ds_all_CO[f'Prior_{Prior}'].attrs = {
                    'units': 'gC m-2 day-1',
                    'description': f'Prior fire CO flux estimated by {Prior}'
                }
                ds_all_CO[f'Posterior_exp{str(nn).zfill(2)}'] = ds['CO_post']
                ds_all_CO[f'Posterior_exp{str(nn).zfill(2)}'].attrs = {
                    'units': 'gC m-2 day-1',
                    'long_name': f'Posterior_{Prior}_{opt}_{Rep_clear_all[j]}',
                    'description': (
                        'Posterior fire CO flux estimated by a TROPOMI XCO inversion, with the following settings:\n'
                        f'- Prior fire emissions from {Prior}\n'
                        f'- {opt_all_long[k]}\n'
                        f'- {Rep_all_long[j]}'
                    )
                }
                nn += 1
                
    # Reorder the dataset
    reordered_list= [
        'grid_area',
        'Prior_GFED',
        'Prior_GFAS',
        'Prior_QFED',
        'Posterior_exp01',
        'Posterior_exp02',
        'Posterior_exp03',
        'Posterior_exp04',
        'Posterior_exp05',
        'Posterior_exp06',
        'Posterior_exp07',
        'Posterior_exp08',
        'Posterior_exp09',
        'Posterior_exp10',
        'Posterior_exp11',
        'Posterior_exp12',
        'Posterior_exp13',
        'Posterior_exp14',
        'Posterior_exp15',
        'Posterior_exp16',
        'Posterior_exp17',
        'Posterior_exp18',
        'Posterior_exp19',
        'Posterior_exp20',
        'Posterior_exp21',
        'Posterior_exp22',
        'Posterior_exp23',
        'Posterior_exp24'
    ]
    actual_vars = list(ds_all_CO2.data_vars)
    filtered_reordered_list = [var for var in reordered_list if var in actual_vars]
    ds_all_CO2_reorder = ds_all_CO2[filtered_reordered_list]
    ds_all_CO_reorder = ds_all_CO[filtered_reordered_list]
    
    # Check which time entries do not have all zero values
    non_zero_times = (ds_all_CO2_reorder['Posterior_exp01'].values != 0).any(axis=(1, 2))
    
    # Subselect the dataset for these times
    ds_all_CO2_reorder_non_zero = ds_all_CO2_reorder.sel(time=ds_all_CO2_reorder.time[non_zero_times])
    ds_all_CO2_reorder_non_zero.attrs['description'] = "This dataset contains daily grided (2 x 2.5) fire CO2 emissions from a variety of TROPOMI XCO inversion experiments performed with CMS-Flux for Byrne et al. (2024). This netcdf file contains fluxes for the year "+str(year).zfill(4)
    file_path = "Archived_data/Fire_CO2_emissions_during_"+str(year).zfill(4)+".nc"
    ds_all_CO2_reorder_non_zero.to_netcdf(file_path)
    
    ds_all_CO_reorder_non_zero = ds_all_CO_reorder.sel(time=ds_all_CO_reorder.time[non_zero_times])
    ds_all_CO_reorder_non_zero.attrs['description'] = "This dataset contains daily grided (2 x 2.5) fire CO emissions from a variety of TROPOMI XCO inversion experiments performed with CMS-Flux for Byrne et al. (2024). This netcdf file contains fluxes for the year "+str(year).zfill(4)
    file_path = "Archived_data/Fire_CO_emissions_during_"+str(year).zfill(4)+".nc"
    ds_all_CO_reorder_non_zero.to_netcdf(file_path)


def write_archived_MonteCarlo_data(year):
    
    Rep_all = ['']
    Rep_clear_all = ['noTROPOMIrepUnc', 'TROPOMIrepUnc']
    Rep_all_long = ['TROPOMI XCO uncertainties without representativeness errors', 'TROPOMI XCO uncertainties with representativeness errors']
    Prior_all = ['GFED','GFAS','QFED']
    opt_all = ['7day']
    opt_all_long = ['near constant uncertainty and 3 day optimization', '200% uncertainty and 7 day optimization']
    
    first_iteration = True
    base_dir = '/u/bbyrne1/python_codes/Canada_Fires_2023/Byrne_etal_codes/plot_figures/data_for_figures/'
    
    nn = 1
    for j, Rep in enumerate(Rep_all):
        for Prior in Prior_all:
            for k, opt in enumerate(opt_all):
                nc_file = base_dir + f'TROPOMI{Rep}_{Prior}_COinv_2x25_{year}_fire_{opt}_MonteCarlo.nc'
                if not os.path.exists(nc_file):
                    print(f"File does not exist: {nc_file}")
                    continue

                print(f"Processing file: {nc_file}")
                ds = xr.open_dataset(nc_file)
                ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})
                
                if first_iteration:
                    time = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
                    ds_all_CO2 = create_xarray_monte_carlo(ds, time)
                    ds_all_CO = create_xarray_monte_carlo(ds, time)
                    first_iteration = False

                ds_all_CO2[f'Prior_{Prior}'] = ds['CO2_prior']
                ds_all_CO2[f'Prior_{Prior}'].attrs = {
                    'units': 'gC m-2 day-1',
                    'description': f'Perturbed prior fire CO2 flux estimated by {Prior}'
                }
                ds_all_CO2[f'Posterior_exp{str(nn).zfill(2)}'] = ds['CO2_post']
                ds_all_CO2[f'Posterior_exp{str(nn).zfill(2)}'].attrs = {
                    'units': 'gC m-2 day-1',
                    'long_name': f'Posterior_{Prior}_{opt}_{Rep_clear_all[j]}',
                    'description': (
                        'Posterior fire CO2 flux estimated by a TROPOMI XCO inversion with perturbed prior scaling factors and the following settings:\n'
                        f'- Prior fire emissions from {Prior}\n'
                        f'- {opt_all_long[k]}\n'
                        f'- {Rep_all_long[j]}'
                    )
                }
                ds_all_CO[f'Prior_{Prior}'] = ds['CO_prior']
                ds_all_CO[f'Prior_{Prior}'].attrs = {
                    'units': 'gC m-2 day-1',
                    'description': f'Perturbed prior fire CO flux estimated by {Prior}'
                }
                ds_all_CO[f'Posterior_exp{str(nn).zfill(2)}'] = ds['CO_post']
                ds_all_CO[f'Posterior_exp{str(nn).zfill(2)}'].attrs = {
                    'units': 'gC m-2 day-1',
                    'long_name': f'Posterior_{Prior}_{opt}_{Rep_clear_all[j]}',
                    'description': (
                        'Posterior fire CO flux estimated by a TROPOMI XCO inversion with perturbed prior scaling factors and the following settings:\n'
                        f'- Prior fire emissions from {Prior}\n'
                        f'- {opt_all_long[k]}\n'
                        f'- {Rep_all_long[j]}'
                    )
                }
                nn += 1
                
    # Reorder the dataset
    reordered_list= [
        'grid_area',
        'Prior_GFED',
        'Prior_GFAS',
        'Prior_QFED',
        'Posterior_exp01',
        'Posterior_exp02',
        'Posterior_exp03',
        'Posterior_exp04',
        'Posterior_exp05',
        'Posterior_exp06',
        'Posterior_exp07',
        'Posterior_exp08',
        'Posterior_exp09',
        'Posterior_exp10',
        'Posterior_exp11',
        'Posterior_exp12',
        'Posterior_exp13',
        'Posterior_exp14',
        'Posterior_exp15',
        'Posterior_exp16',
        'Posterior_exp17',
        'Posterior_exp18',
        'Posterior_exp19',
        'Posterior_exp20',
        'Posterior_exp21',
        'Posterior_exp22',
        'Posterior_exp23',
        'Posterior_exp24'
    ]
    actual_vars = list(ds_all_CO2.data_vars)
    filtered_reordered_list = [var for var in reordered_list if var in actual_vars]
    ds_all_CO2_reorder = ds_all_CO2[filtered_reordered_list]
    ds_all_CO_reorder = ds_all_CO[filtered_reordered_list]
    
    # Check which time entries do not have all zero values
    non_zero_times = (ds_all_CO2_reorder['Posterior_exp01'].values != 0).any(axis=(0,2, 3))
    
    # Subselect the dataset for these times
    ds_all_CO2_reorder_non_zero = ds_all_CO2_reorder.sel(time=ds_all_CO2_reorder.time[non_zero_times])
    ds_all_CO2_reorder_non_zero.attrs['description'] = "This dataset contains the Monte Carlo ensemble of daily grided (2 x 2.5) fire CO2 emissions for TROPOMI XCO inversion experiments performed with CMS-Flux for Byrne et al. (2024). This netcdf file contains fluxes for the year "+str(year).zfill(4)
    file_path = "Archived_data/MonteCarlo_Fire_CO2_emissions_during_"+str(year).zfill(4)+".nc"
    ds_all_CO2_reorder_non_zero.to_netcdf(file_path)
    
    ds_all_CO_reorder_non_zero = ds_all_CO_reorder.sel(time=ds_all_CO_reorder.time[non_zero_times])
    ds_all_CO_reorder_non_zero.attrs['description'] = "This dataset contains the Monte Carlo ensemble of daily grided (2 x 2.5) fire CO emissions from a variety of TROPOMI XCO inversion experiments performed with CMS-Flux for Byrne et al. (2024). This netcdf file contains fluxes for the year "+str(year).zfill(4)
    file_path = "Archived_data/MonteCarlo_Fire_CO_emissions_during_"+str(year).zfill(4)+".nc"
    ds_all_CO_reorder_non_zero.to_netcdf(file_path)

for year in range(2019,2024):
    write_archived_data(year)

write_archived_MonteCarlo_data(2023)
