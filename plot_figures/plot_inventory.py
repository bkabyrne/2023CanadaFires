from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

'''
------- plot_inventory

Makes Fig. 4 from the paper (Inventory_Fires.png)

'''

def process_flux(file_name, forest_mask):
    '''                                                                        
    Calculate May-Sep CO+CO2 emissions over masked region.                     
    '''
    print(f"Processing file: {file_name}")
    with Dataset(file_name, 'r') as f:
        CO_post_per_area = f.variables['CO_post'][:]  # gC/m2                  
        CO2_post_per_area = f.variables['CO2_post'][:]  # gC/m2                
        area = f.variables['grid_area'][:]  # m2                               

    # Managed forest emissions                                                 
    CO_and_CO2_post = np.sum(CO_post_per_area[120:273] + CO2_post_per_area[120:273], axis=0) * forest_mask
    post_total_emissions = np.sum(CO_and_CO2_post * area)

    return post_total_emissions

def calculate_posterior_managed_forest_emissions():
    '''                                                                        
    Calculate emissions over managed forests for each inversion.               
    '''

    # Read forest mask data                                                    
    with Dataset('./data_for_figures/Canada_forest_mask_2x25.nc', 'r') as f:
        Forest_mask_2x25 = f.variables['mask'][:]

    # Read managed and unmanaged mask data                                     
    with Dataset('Canada_managed_mask_2x25.nc', 'r') as f:
        managed = f.variables['managed'][:]

    managed_forest_mask = Forest_mask_2x25 * managed

    # File names for processing                                                
    file_names = [
        './data_for_figures/TROPOMI_GFED_COinv_2x25_2023_fire_3day.nc',
        './data_for_figures/TROPOMI_GFAS_COinv_2x25_2023_fire_3day.nc',
        './data_for_figures/TROPOMI_QFED_COinv_2x25_2023_fire_3day.nc',
        './data_for_figures/TROPOMI_rep_GFED_COinv_2x25_2023_fire_3day.nc',
        './data_for_figures/TROPOMI_rep_GFAS_COinv_2x25_2023_fire_3day.nc',
        './data_for_figures/TROPOMI_rep_QFED_COinv_2x25_2023_fire_3day.nc',
        './data_for_figures/TROPOMI_GFED_COinv_2x25_2023_fire_7day.nc',
        './data_for_figures/TROPOMI_GFAS_COinv_2x25_2023_fire_7day.nc',
        './data_for_figures/TROPOMI_QFED_COinv_2x25_2023_fire_7day.nc',
        './data_for_figures/TROPOMI_rep_GFED_COinv_2x25_2023_fire_7day.nc',
        './data_for_figures/TROPOMI_rep_GFAS_COinv_2x25_2023_fire_7day.nc',
        './data_for_figures/TROPOMI_rep_QFED_COinv_2x25_2023_fire_7day.nc',
        './data_for_figures/TROPOMI_OH_GFED_COinv_2x25_2023_fire_3day.nc',
        './data_for_figures/TROPOMI_OH_GFAS_COinv_2x25_2023_fire_3day.nc',
        './data_for_figures/TROPOMI_OH_QFED_COinv_2x25_2023_fire_3day.nc',
        './data_for_figures/TROPOMI_OH_rep_GFED_COinv_2x25_2023_fire_3day.nc',
        './data_for_figures/TROPOMI_OH_rep_GFAS_COinv_2x25_2023_fire_3day.nc',
        './data_for_figures/TROPOMI_OH_rep_QFED_COinv_2x25_2023_fire_3day.nc',
        './data_for_figures/TROPOMI_OH_GFED_COinv_2x25_2023_fire_7day.nc',
        './data_for_figures/TROPOMI_OH_GFAS_COinv_2x25_2023_fire_7day.nc',
        './data_for_figures/TROPOMI_OH_QFED_COinv_2x25_2023_fire_7day.nc',
        './data_for_figures/TROPOMI_OH_rep_GFED_COinv_2x25_2023_fire_7day.nc',
        './data_for_figures/TROPOMI_OH_rep_GFAS_COinv_2x25_2023_fire_7day.nc',
        './data_for_figures/TROPOMI_OH_rep_QFED_COinv_2x25_2023_fire_7day.nc'
    ]

    posterior_flux = np.zeros(len(file_names))

    for i, file_name in enumerate(file_names):
        posterior_flux[i] = process_flux(file_name, managed_forest_mask)

    return posterior_flux

if __name__ == '__main__':

    posterior_flux = calculate_posterior_managed_forest_emissions()
    Top_down_mean = np.mean(posterior_flux)*1e-12 # TgC
    Top_down_min = np.min(posterior_flux)*1e-12
    Top_down_max = np.max(posterior_flux)*1e-12

    # Define the data
    year =	np.array([2016,2017,2018,2019,2020,2021])
    forest_sink = np.array([-37.2,-37.1,-36.5,-37.4,-35.9,-36.3])
    HWP = np.array([37.3,37.2,38.0,35.5,35.0,34.8])
    natural_disturbance = np.array([32.7,65.5,73.6,46.6,5.7,84.5])
    total_co2 = np.array([152.1,154.5,157.4,157.8,142.6,146.5])

    # Plot the data
    fig = plt.figure(2,figsize=(6*0.95,4*0.95),dpi=300)
    ax1 = fig.add_axes([0.15,0.1,0.8,0.8])
    l1=plt.plot(year,HWP,color='peru')
    plt.plot(year,HWP,'o',color='peru',markersize=4)
    plt.text(2016.15,30,'Harvested Wood Products',color='peru',ha='left',va='top')
    l2=plt.plot(year,forest_sink,color='green')
    plt.plot(year,forest_sink,'o',color='green',markersize=4)
    plt.text(2016.15,-42,'Forest Land',color='green',ha='left',va='top')
    l3=plt.plot(year,natural_disturbance,'--',color='red')
    plt.plot(year,natural_disturbance,'o',color='red',markersize=4)
    plt.text(2017.15,80,'Natural disturbances',color='red',ha='left',va='bottom')
    plt.ylabel('carbon flux (TgC)')
    l4=plt.plot(year,total_co2,color='grey')
    plt.plot(year,total_co2,'o',color='grey',markersize=4)
    plt.text(2016.15,160,'Total CO$_2$ emissions',color='grey',ha='left',va='bottom')
    # Canadian Fires
    #plt.plot(2023,421.3,'ko')
    #l5=plt.plot([2023,2023],[387.5,460.8],'k')
    #plt.plot([2022.8,2023.2],[460.8,460.8],'k')
    #plt.plot([2022.8,2023.2],[387.5,387.5],'k')
    plt.plot(2023,Top_down_mean,'ko')
    l5=plt.plot([2023,2023],[Top_down_min,Top_down_max],'k')
    plt.plot([2022.8,2023.2],[Top_down_max,Top_down_max],'k')
    plt.plot([2022.8,2023.2],[Top_down_min,Top_down_min],'k')
    plt.plot([2015,2024],[0,0],'k:')
    plt.text(2020.15,425,'2023 Canadian Fires',color='k',ha='left',va='bottom')
    plt.text(2020.15,395,'(managed lands)',color='k',ha='left',va='bottom')
    plt.xlim([2015.5,2023.5])
    plt.ylim([-70,475])
    plt.savefig('Figures/Inventory_Fires.png')
    plt.clf()
    plt.close()
