import cdsapi
import numpy as np


def download_cams_global_biomass_emissions(year,month,day):

    # ######################################################
    #
    # Download GFAS fire emissions dataset
    #
    # ######################################################

    date1 = str(year).zfill(4)+'-'+str(month).zfill(2)+'-'+str(day).zfill(2)    

    # Replace 'YOUR_CDS_API_KEY' with your actual CDS API key if required.
    c = cdsapi.Client(url='https://ads.atmosphere.copernicus.eu/api/v2', key='')

    # Request parameters for GFAS data
    request_params = {
        'variable': [
            'injection_height', 'wildfire_flux_of_carbon_dioxide', 'wildfire_flux_of_carbon_monoxide',
        ],
        'date': date1+'/'+date1,  # Replace with your desired date range
        'format': 'netcdf',
    }

    # Replace 'download.nc' with the desired output filename
    output_file = str(year).zfill(4)+'/'+str(month).zfill(2)+'/'+str(year).zfill(4)+str(month).zfill(2)+str(day).zfill(2)+'.nc'

    try:
        # Send the request to retrieve the data
        c.retrieve('cams-global-fire-emissions-gfas', request_params, output_file)
        print("Download complete.")
    except Exception as e:
        print("An error occurred while downloading the data:", e)

if __name__ == "__main__":

    year_arr = np.array([2023])
    for year in year_arr:
        if np.logical_or(year == 2012,year == 2008):
            days_in_month = np.array([31,29,31,30,31,30,31,31,30,31,30,31])
        else:
            days_in_month = np.array([0,0,0,0,0,0,0,0,30,31,30,31])
        for monthi in range(12):
            month=monthi+1
            for dayi in range(days_in_month[monthi]):
                download_cams_global_biomass_emissions(year,month,dayi+1)

