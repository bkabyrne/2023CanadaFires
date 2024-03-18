import csv
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import datetime
import glob, os
from scipy import stats
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Polygon
from math import pi, cos, radians
import numpy.matlib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap, maskoceans
from pylab import *

# *******************************************************   
# -------- plot_Fire_emissions.py                        
#                                                           
# This code processes data and plots Figures 1 and S1   
#                                                           
# contact: Brendan Byrne                                    
# email: brendan.k.byrne@jpl.nasa.gov                       
#                                                           
# ******************************************************* 



def calc_forest_total(flux,areain,Forest_maskin):
    #
    # ==============================
    # Sum flux over forested area
    # ==============================
    #
    flux_temp = flux * areain
    flux_temp[np.where(Forest_maskin==0)] = np.nan
    flux_total = np.nansum(flux_temp)
    #
    return flux_total



def MaySep_forest_fluxes(year,file_name,Forest_mask_2x25):
    #
    # =====================================
    #
    # Reads Posterior fluxes and calculates the total
    # CO+CO2 emissions over forests.
    # ---
    #  Calls calc_forest_fires().
    #
    # =====================================
    #
    print(file_name)
    #
    f = Dataset(file_name,'r')
    CO2_post = f.variables['CO2_post'][:] # gC/m2/day
    CO_post = f.variables['CO_post'][:] # gC/m2/day
    area_2x25 = f.variables['grid_area'][:] # m2
    f.close()
    #
    # =============
    # May-Sep totals
    if np.logical_and(year % 4 == 0, year % 100 != 0):
        CO2_post_MaySep = np.nansum(CO2_post[121-1+1:274-1+1,:,:],0)
        CO_post_MaySep = np.nansum(CO_post[121-1+1:274-1+1,:,:],0)
    else:
        CO2_post_MaySep = np.nansum(CO2_post[121-1+1:274-1,:,:],0)
        CO_post_MaySep = np.nansum(CO_post[121-1+1:274-1,:,:],0)
    # =============
    #
    CO2andCO_post_MaySep = CO2_post_MaySep + CO_post_MaySep
    CO2andCO_post_MaySep_forest = calc_forest_total(CO2andCO_post_MaySep,area_2x25,Forest_mask_2x25)
    #s
    return CO2andCO_post_MaySep_forest

def MaySep_forest_fluxes_MonteCarlo(year,file_name,Forest_mask_2x25):
    #
    # =====================================
    #
    # Reads Posterior fluxes and calculates the total
    # CO+CO2 emissions over forests for Monte Carlo runs
    # ---
    #  Calls calc_forest_fires().
    #
    # =====================================
    #
    print(file_name)
    #
    f = Dataset(file_name,'r')
    CO2_post = f.variables['CO2_post'][:] # gC/m2/day
    CO_post = f.variables['CO_post'][:] # gC/m2/day
    area_2x25 = f.variables['grid_area'][:] # m2
    f.close()
    #
    # =============
    # May-Sep totals
    if np.logical_and(year % 4 == 0, year % 100 != 0):
        CO2_post_MaySep = np.nansum(CO2_post[:,121-1+1:274-1+1,:,:],1)
        CO_post_MaySep = np.nansum(CO_post[:,121-1+1:274-1+1,:,:],1)
    else:
        CO2_post_MaySep = np.nansum(CO2_post[:,121-1+1:274-1,:,:],1)
        CO_post_MaySep = np.nansum(CO_post[:,121-1+1:274-1,:,:],1)
    # =============
    #
    CO2andCO_post_MaySep = CO2_post_MaySep + CO_post_MaySep
    CO2andCO_post_MaySep_forest = np.zeros(40)
    for i in range(40):
        CO2andCO_post_MaySep_forest[i] = calc_forest_total(CO2andCO_post_MaySep[i,:,:],area_2x25,Forest_mask_2x25)
    #s
    return CO2andCO_post_MaySep_forest


def MaySep_forest_fluxes_prior(file_name,Forest_mask_2x25):
    #
    # =====================================
    #
    # Reads Prior ONLY fluxes and calculates the total
    # CO+CO2 emissions over forests.
    # ---
    #  Calls calc_forest_fires().
    #
    # =====================================
    #
    print(file_name)
    #
    f = Dataset(file_name,'r')
    CO_Flux_per_area = f.variables['CO_Flux'][:] # gC/m2
    CO2_Flux_per_area = f.variables['CO2_Flux'][:] # gC/m2
    area_2x25 = f.variables['area'][:] # m2
    f.close()
    total_Flux_per_area = CO2_Flux_per_area + CO_Flux_per_area
    prior_MaySep_vec = np.zeros(21)
    for ii in range(21):
        prior_MaySep_vec[ii] = calc_forest_total(total_Flux_per_area[ii,:,:],area_2x25,Forest_mask_2x25)
    #
    return prior_MaySep_vec


def plot_box_whiskers(x,array,colorN,alphaN,hatch_flag):
    if hatch_flag==1:
        plt.fill_between([x+0.4,x-0.4],[np.percentile(array,25),np.percentile(array,25)],[np.percentile(array,75),np.percentile(array,75)],color=colorN,alpha=alphaN,edgecolor=colorN,hatch="//")
    else:
        plt.fill_between([x+0.4,x-0.4],[np.percentile(array,25),np.percentile(array,25)],[np.percentile(array,75),np.percentile(array,75)],color=colorN,alpha=alphaN,linewidth=0.0)
    plt.plot([x+0.4,x-0.4],[np.max(array),np.max(array)],color=colorN,solid_capstyle='butt',alpha=alphaN)
    plt.plot([x,x],[np.percentile(array,75),np.max(array)],color=colorN,solid_capstyle='butt',alpha=alphaN)
    plt.plot([x-0.4,x+0.4],[np.median(array),np.median(array)],color=colorN,solid_capstyle='butt',alpha=alphaN)
    plt.plot([x,x],[np.percentile(array,25),np.min(array)],color=colorN,solid_capstyle='butt',alpha=alphaN)
    plt.plot([x+0.4,x-0.4],[np.min(array),np.min(array)],color=colorN,solid_capstyle='butt',alpha=alphaN)



#
#
# ===== Read data and calculate mean over Canadian forests =====
#
nc_out = './data_for_figures/Canada_forest_mask_2x25.nc'
f = Dataset(nc_out,'r')
lon_MERRA2_2x25 = f.variables['lon'][:]
lat_MERRA2_2x25 = f.variables['lat'][:]
Forest_mask_2x25 = f.variables['mask'][:]
f.close()
#
# ----------------------------------------------------------------------------------------
#
#  Read Prior/Posterior fluxes and calculate annual fire emissions over forests
#
# Calculate GFED posterior May-Sep Fire emissions for each year for the MOPITT inversions
post_MaySep_MOPITT_vec = np.zeros(12)
for ii in range(12):
    yyyy = ii + 2010
    nc_out = './data_for_figures/MOPITT_COinv_2x25_'+str(yyyy).zfill(4)+'_COandCO2.nc'
    post_MaySep_MOPITT_vec[ii] = MaySep_forest_fluxes(yyyy,nc_out,Forest_mask_2x25)
# Calculate GFED posterior May-Sep Fire emissions for each year for the TROPOMI inversions
post_MaySep_TROPOMI_vec = np.zeros(5)
for ii in range(5):
    yyyy = ii + 2019
    nc_out = './data_for_figures/TROPOMI_COinv_2x25_'+str(yyyy).zfill(4)+'_COandCO2.nc'
    post_MaySep_TROPOMI_vec[ii] = MaySep_forest_fluxes(yyyy,nc_out,Forest_mask_2x25)
# Calculate QFED posterior May-Sep Fire emissions for each year for the TROPOMI inversions
nc_out = './data_for_figures/TROPOMI_QFED_COinv_2x25_2023_COandCO2.nc'
post_MaySep_TROPOMI_QFED_vec = MaySep_forest_fluxes(2023,nc_out,Forest_mask_2x25)
# Calculate GFAS posterior May-Sep Fire emissions for each year for the TROPOMI inversions
nc_out = './data_for_figures/TROPOMI_GFAS_COinv_2x25_2023_COandCO2.nc'
post_MaySep_TROPOMI_GFAS_vec = MaySep_forest_fluxes(2023,nc_out,Forest_mask_2x25)
#


# ---------------------------------------------
post_MaySep_TROPOMI_GFED_3day_vec = np.zeros(5)
for i in range(5):
    nc_file = './data_for_figures/TROPOMI_GFED_COinv_2x25_'+str(i+2019)+'_fire_3day.nc'
    post_MaySep_TROPOMI_GFED_3day_vec[i] = MaySep_forest_fluxes(i+2019,nc_file,Forest_mask_2x25)
post_MaySep_TROPOMI_QFED_3day_vec = np.zeros(5)
for i in range(5):
    nc_file = './data_for_figures/TROPOMI_QFED_COinv_2x25_'+str(i+2019)+'_fire_3day.nc'
    post_MaySep_TROPOMI_QFED_3day_vec[i] = MaySep_forest_fluxes(i+2019,nc_file,Forest_mask_2x25)
post_MaySep_TROPOMI_GFAS_3day_vec = np.zeros(5)
for i in range(5):
    nc_file = './data_for_figures/TROPOMI_GFAS_COinv_2x25_'+str(i+2019)+'_fire_3day.nc'
    post_MaySep_TROPOMI_GFAS_3day_vec[i] = MaySep_forest_fluxes(i+2019,nc_file,Forest_mask_2x25)
# ---------------------------------------------
post_MaySep_TROPOMI_rep_GFED_3day_vec = np.zeros(5)
for i in range(5):
    nc_file = './data_for_figures/TROPOMI_rep_GFED_COinv_2x25_'+str(i+2019)+'_fire_3day.nc'
    post_MaySep_TROPOMI_rep_GFED_3day_vec[i] = MaySep_forest_fluxes(i+2019,nc_file,Forest_mask_2x25)
post_MaySep_TROPOMI_rep_QFED_3day_vec = np.zeros(5)
for i in range(5):
    nc_file = './data_for_figures/TROPOMI_rep_QFED_COinv_2x25_'+str(i+2019)+'_fire_3day.nc'
    post_MaySep_TROPOMI_rep_QFED_3day_vec[i] = MaySep_forest_fluxes(i+2019,nc_file,Forest_mask_2x25)
post_MaySep_TROPOMI_rep_GFAS_3day_vec = np.zeros(5)
for i in range(5):
    nc_file = './data_for_figures/TROPOMI_rep_GFAS_COinv_2x25_'+str(i+2019)+'_fire_3day.nc'
    post_MaySep_TROPOMI_rep_GFAS_3day_vec[i] = MaySep_forest_fluxes(i+2019,nc_file,Forest_mask_2x25)


# ---------------------------------------------
post_MaySep_TROPOMI_GFED_7day_vec = np.zeros(5)
for i in range(5):
    nc_file = './data_for_figures/TROPOMI_GFED_COinv_2x25_'+str(i+2019)+'_fire_7day.nc'
    post_MaySep_TROPOMI_GFED_7day_vec[i] = MaySep_forest_fluxes(i+2019,nc_file,Forest_mask_2x25)
post_MaySep_TROPOMI_QFED_7day_vec = np.zeros(5)
for i in range(5):
    nc_file = './data_for_figures/TROPOMI_QFED_COinv_2x25_'+str(i+2019)+'_fire_7day.nc'
    post_MaySep_TROPOMI_QFED_7day_vec[i] = MaySep_forest_fluxes(i+2019,nc_file,Forest_mask_2x25)
post_MaySep_TROPOMI_GFAS_7day_vec = np.zeros(5)
for i in range(5):
    nc_file = './data_for_figures/TROPOMI_GFAS_COinv_2x25_'+str(i+2019)+'_fire_7day.nc'
    post_MaySep_TROPOMI_GFAS_7day_vec[i] = MaySep_forest_fluxes(i+2019,nc_file,Forest_mask_2x25)
# ---------------------------------------------
post_MaySep_TROPOMI_rep_GFED_7day_vec = np.zeros(5)
for i in range(5):
    nc_file = './data_for_figures/TROPOMI_rep_GFED_COinv_2x25_'+str(i+2019)+'_fire_7day.nc'
    post_MaySep_TROPOMI_rep_GFED_7day_vec[i] = MaySep_forest_fluxes(i+2019,nc_file,Forest_mask_2x25)
post_MaySep_TROPOMI_rep_QFED_7day_vec = np.zeros(5)
for i in range(5):
    nc_file = './data_for_figures/TROPOMI_rep_QFED_COinv_2x25_'+str(i+2019)+'_fire_7day.nc'
    post_MaySep_TROPOMI_rep_QFED_7day_vec[i] = MaySep_forest_fluxes(i+2019,nc_file,Forest_mask_2x25)
post_MaySep_TROPOMI_rep_GFAS_7day_vec = np.zeros(5)
for i in range(5):
    nc_file = './data_for_figures/TROPOMI_rep_GFAS_COinv_2x25_'+str(i+2019)+'_fire_7day.nc'
    post_MaySep_TROPOMI_rep_GFAS_7day_vec[i] = MaySep_forest_fluxes(i+2019,nc_file,Forest_mask_2x25)


# ----------- Monte Carlo estimates
nc_file = './data_for_figures/TROPOMI_GFED_COinv_2x25_2023_fire_3day_MonthCarlo.nc'
post_MaySep_TROPOMI_rep_GFED_7day_MonteCarlo = MaySep_forest_fluxes_MonteCarlo(2023,nc_file,Forest_mask_2x25)
#
nc_file = './data_for_figures/TROPOMI_QFED_COinv_2x25_2023_fire_3day_MonthCarlo.nc'
post_MaySep_TROPOMI_rep_QFED_7day_MonteCarlo = MaySep_forest_fluxes_MonteCarlo(2023,nc_file,Forest_mask_2x25)
#
nc_file = './data_for_figures/TROPOMI_GFAS_COinv_2x25_2023_fire_3day_MonthCarlo.nc'
post_MaySep_TROPOMI_rep_GFAS_7day_MonteCarlo = MaySep_forest_fluxes_MonteCarlo(2023,nc_file,Forest_mask_2x25)
# -----------


#
# ----------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------
# Calculate GFED prior May-Sep Fire emissions for full 21-year record
file_name = './data_for_figures/GFED_monthly_MERRA2_regrid_MaytoSep_2x25.nc'
prior_GFED_MaySep_vec = MaySep_forest_fluxes_prior(file_name,Forest_mask_2x25)
# Calculate QFED prior May-Sep Fire emissions for full 21-year record
file_name = './data_for_figures/QFED_monthly_MERRA2_regrid_MaytoSep_2x25.nc'
prior_QFED_MaySep_vec = MaySep_forest_fluxes_prior(file_name,Forest_mask_2x25)
# Calculate GFAS prior May-Sep Fire emissions for full 21-year record
file_name = './data_for_figures/GFAS_monthly_MERRA2_regrid_MaytoSep_2x25.nc'
prior_GFAS_MaySep_vec = MaySep_forest_fluxes_prior(file_name,Forest_mask_2x25)
# ----------------------------------------------------------------------------------------



#
#
# ===== Plot Figure S1 =====
#
#
cmap_test1 = cm.get_cmap('cividis')
colors = cmap_test1(np.linspace(0, 1, 5))

fig = plt.figure(25, figsize=(7,2), dpi=300)
ax1 = fig.add_axes([0.125, 0.1, .85, 0.875])
plt.fill_between([4,5],[0,0],[900,900],color='b',alpha=0.05,linewidth=0.0)
plt.fill_between([6,7],[0,0],[900,900],color='b',alpha=0.05,linewidth=0.0)
plt.fill_between([8,9],[0,0],[900,900],color='b',alpha=0.05,linewidth=0.0)
plt.fill_between([10,11],[0,0],[900,900],color='b',alpha=0.05,linewidth=0.0)
plt.fill_between([12,13],[0,0],[900,900],color='b',alpha=0.05,linewidth=0.0)
plt.fill_between([14,15],[0,0],[900,900],color='b',alpha=0.05,linewidth=0.0)
plt.fill_between([16,17],[0,0],[900,900],color='b',alpha=0.05,linewidth=0.0)
plt.fill_between([18,19],[0,0],[900,900],color='b',alpha=0.05,linewidth=0.0)
plt.fill_between([20,21],[0,0],[900,900],color='b',alpha=0.05,linewidth=0.0)
plt.fill_between([22,23],[0,0],[900,900],color='b',alpha=0.05,linewidth=0.0)
l1=plt.bar(np.arange(21)+3+0.5/3.,prior_GFED_MaySep_vec*1e-12,width=1./3.,color=colors[1],alpha=0.7)
l2=plt.bar(np.arange(21)+3+1.5/3.,prior_GFAS_MaySep_vec*1e-12,width=1./3.,color=colors[2],alpha=0.7)
l3=plt.bar(np.arange(21)+3+2.5/3.,prior_QFED_MaySep_vec*1e-12,width=1./3.,color=colors[3],alpha=0.7)
l4=plt.plot(np.arange(12)+10+0.4/3.,post_MaySep_MOPITT_vec*1e-12,'o',markerfacecolor='powderblue',markersize=4,markeredgecolor='k',markeredgewidth=.5)
l5=plt.plot(np.arange(5)+19+0.6/3.,post_MaySep_TROPOMI_vec*1e-12,'d',markerfacecolor='peachpuff',markersize=4,markeredgecolor='k',markeredgewidth=.5)
l6=plt.plot(np.arange(1)+23+1.6/3.,post_MaySep_TROPOMI_GFAS_vec*1e-12,'v',markerfacecolor='peachpuff',markersize=4,markeredgecolor='k',markeredgewidth=.5)
l7=plt.plot(np.arange(1)+23+2.6/3.,post_MaySep_TROPOMI_QFED_vec*1e-12,'^',markerfacecolor='peachpuff',markersize=4,markeredgecolor='k',markeredgewidth=.5)
#
plt.plot(np.arange(5)+19+0.2/3.,post_MaySep_TROPOMI_GFED_3day_vec*1e-12,'*',markerfacecolor='orange',markersize=4,markeredgecolor='k',markeredgewidth=.5)
plt.plot(np.arange(5)+19+1.2/3.,post_MaySep_TROPOMI_GFAS_3day_vec*1e-12,'*',markerfacecolor='orange',markersize=4,markeredgecolor='k',markeredgewidth=.5)
plt.plot(np.arange(5)+19+2.2/3.,post_MaySep_TROPOMI_QFED_3day_vec*1e-12,'*',markerfacecolor='orange',markersize=4,markeredgecolor='k',markeredgewidth=.5)
#
plt.plot(np.arange(5)+19+0.8/3.,post_MaySep_TROPOMI_rep_GFED_3day_vec*1e-12,'<',markerfacecolor='green',markersize=4,markeredgecolor='k',markeredgewidth=.5)
plt.plot(np.arange(5)+19+1.8/3.,post_MaySep_TROPOMI_rep_GFAS_3day_vec*1e-12,'<',markerfacecolor='green',markersize=4,markeredgecolor='k',markeredgewidth=.5)
plt.plot(np.arange(5)+19+2.8/3.,post_MaySep_TROPOMI_rep_QFED_3day_vec*1e-12,'<',markerfacecolor='green',markersize=4,markeredgecolor='k',markeredgewidth=.5)
#
plt.plot(np.arange(5)+19+0.2/3.,post_MaySep_TROPOMI_GFED_7day_vec*1e-12,'s',markerfacecolor='blue',markersize=4,markeredgecolor='k',markeredgewidth=.5)
plt.plot(np.arange(5)+19+1.2/3.,post_MaySep_TROPOMI_GFAS_7day_vec*1e-12,'s',markerfacecolor='blue',markersize=4,markeredgecolor='k',markeredgewidth=.5)
plt.plot(np.arange(5)+19+2.2/3.,post_MaySep_TROPOMI_QFED_7day_vec*1e-12,'s',markerfacecolor='blue',markersize=4,markeredgecolor='k',markeredgewidth=.5)
#
plt.plot(np.arange(5)+19+0.8/3.,post_MaySep_TROPOMI_rep_GFED_7day_vec*1e-12,'>',markerfacecolor='red',markersize=4,markeredgecolor='k',markeredgewidth=.5)
plt.plot(np.arange(5)+19+1.8/3.,post_MaySep_TROPOMI_rep_GFAS_7day_vec*1e-12,'>',markerfacecolor='red',markersize=4,markeredgecolor='k',markeredgewidth=.5)
plt.plot(np.arange(5)+19+2.8/3.,post_MaySep_TROPOMI_rep_QFED_7day_vec*1e-12,'>',markerfacecolor='red',markersize=4,markeredgecolor='k',markeredgewidth=.5)
#
plt.xlim([3,24])
plt.xticks(np.arange(21)+3)
plt.text(3.5,-15,'03',ha='center',va='top')
plt.text(4.5,-15,'04',ha='center',va='top')
plt.text(5.5,-15,'05',ha='center',va='top')
plt.text(6.5,-15,'06',ha='center',va='top')
plt.text(7.5,-15,'07',ha='center',va='top')
plt.text(8.5,-15,'08',ha='center',va='top')
plt.text(9.5,-15,'09',ha='center',va='top')
plt.text(10.5,-15,'10',ha='center',va='top')
plt.text(11.5,-15,'11',ha='center',va='top')
plt.text(12.5,-15,'12',ha='center',va='top')
plt.text(13.5,-15,'13',ha='center',va='top')
plt.text(14.5,-15,'14',ha='center',va='top')
plt.text(15.5,-15,'15',ha='center',va='top')
plt.text(16.5,-15,'16',ha='center',va='top')
plt.text(17.5,-15,'17',ha='center',va='top')
plt.text(18.5,-15,'18',ha='center',va='top')
plt.text(19.5,-15,'19',ha='center',va='top')
plt.text(20.5,-15,'20',ha='center',va='top')
plt.text(21.5,-15,'21',ha='center',va='top')
plt.text(22.5,-15,'22',ha='center',va='top')
plt.text(23.5,-15,'23',ha='center',va='top')
ax1.set_xticklabels([])
plt.legend((l1[0],l2[0],l3[0],l4[0],l5[0],l6[0],l7[0]),('GFED','GFAS','QFED','MOPITT GFED','TROPOMI GFED','TROPOMI GFAS','TROPOMI QFED'),ncol=2,handletextpad=0.3,labelspacing=0.5)
plt.ylim([0,850])
plt.ylabel('CO$_2$+CO May-Sep\nfire emissions (TgC)')
plt.savefig('Figures/Byrne_etal_FigS1_revision.png', dpi=300)




#plt.xticks([])
#
#plt.text(17.*0.98,0.85*0.98,'(c)',va='top',ha='right')
#
#plt.text(2,-20/1000.,'Bottom-up',ha='center',va='top',fontsize=9)



fig = plt.figure(26, figsize=(6,2.25), dpi=300)
ax1 = fig.add_axes([0.125, 0.15, .85, 0.825])

def plot_section(GFED,GFAS,QFED,start,ispost):
    #,alpha=0.5,hatch="//"
    if ispost==0:
        l1=plt.bar(np.arange(1)+start+0.5,GFED*1e-15,color=colors[1],alpha=0.7,width=0.9)
        l2=plt.bar(np.arange(1)+start+1.5,GFAS*1e-15,color=colors[2],alpha=0.7,width=0.9)
        l3=plt.bar(np.arange(1)+start+2.5,QFED*1e-15,color=colors[3],alpha=0.7,width=0.9)
        plt.legend((l1[0],l2[0],l3[0]),('GFED','GFAS','QFED'),ncol=3,handletextpad=0.3,labelspacing=0.5,loc='upper right')
    else:
        l1=plt.bar(np.arange(1)+start+0.5,GFED*1e-15,color=colors[1],alpha=0.5,hatch="//",width=0.9,edgecolor=colors[1])
        l2=plt.bar(np.arange(1)+start+1.5,GFAS*1e-15,color=colors[2],alpha=0.5,hatch="//",width=0.9,edgecolor=colors[2])
        l3=plt.bar(np.arange(1)+start+2.5,QFED*1e-15,color=colors[3],alpha=0.5,hatch="//",width=0.9,edgecolor=colors[3]) 
        #
    prior_MaySep_vec = np.array([ GFED , GFAS , QFED ]) * (1e-15)
    plt.plot(np.arange(1)+start+3.5,np.mean(prior_MaySep_vec),'ko')
    plt.plot([np.arange(1)+start+3.5,np.arange(1)+start+3.5],[np.min(prior_MaySep_vec),np.max(prior_MaySep_vec)],'k',solid_capstyle='butt')
    plt.plot([np.arange(1)+start+3.5-0.45,np.arange(1)+start+3.5+0.45],[np.min(prior_MaySep_vec),np.min(prior_MaySep_vec)],'k',solid_capstyle='butt')
    plt.plot([np.arange(1)+start+3.5-0.45,np.arange(1)+start+3.5+0.45],[np.max(prior_MaySep_vec),np.max(prior_MaySep_vec)],'k',solid_capstyle='butt')

# ----------
plot_section(prior_GFED_MaySep_vec[20],prior_GFAS_MaySep_vec[20],prior_QFED_MaySep_vec[20],0.5,0)
plt.text(2.5,-20/1000.,'Bottom-up',ha='center',va='top',fontsize=9)
# ----------
plot_section(post_MaySep_TROPOMI_rep_GFED_3day_vec[4],post_MaySep_TROPOMI_rep_GFAS_3day_vec[4],post_MaySep_TROPOMI_rep_QFED_3day_vec[4],5.5,1)
plt.text(7.5,-20/1000.,'3 d opt\nw/ rep err',ha='center',va='top',fontsize=9)
#
GFED_max = post_MaySep_TROPOMI_rep_GFED_3day_vec[4] + np.std(post_MaySep_TROPOMI_rep_GFED_7day_MonteCarlo)
GFED_min = post_MaySep_TROPOMI_rep_GFED_3day_vec[4] - np.std(post_MaySep_TROPOMI_rep_GFED_7day_MonteCarlo)
plt.plot([5.5+0.5,5.5+0.5],[GFED_min*1e-15,GFED_max*1e-15],color=colors[1])
plt.plot([5.5+0.1,5.5+0.9],[GFED_min*1e-15,GFED_min*1e-15],color=colors[1])
plt.plot([5.5+0.1,5.5+0.9],[GFED_max*1e-15,GFED_max*1e-15],color=colors[1])
#
GFAS_max = post_MaySep_TROPOMI_rep_GFAS_3day_vec[4] + np.std(post_MaySep_TROPOMI_rep_GFAS_7day_MonteCarlo)
GFAS_min = post_MaySep_TROPOMI_rep_GFAS_3day_vec[4] - np.std(post_MaySep_TROPOMI_rep_GFAS_7day_MonteCarlo)
plt.plot([5.5+1.5,5.5+1.5],[GFAS_min*1e-15,GFAS_max*1e-15],color=colors[2])
plt.plot([5.5+1.1,5.5+1.9],[GFAS_min*1e-15,GFAS_min*1e-15],color=colors[2])
plt.plot([5.5+1.1,5.5+1.9],[GFAS_max*1e-15,GFAS_max*1e-15],color=colors[2])
#
QFED_max = post_MaySep_TROPOMI_rep_QFED_3day_vec[4] + np.std(post_MaySep_TROPOMI_rep_QFED_7day_MonteCarlo)
QFED_min = post_MaySep_TROPOMI_rep_QFED_3day_vec[4] - np.std(post_MaySep_TROPOMI_rep_QFED_7day_MonteCarlo)
plt.plot([5.5+2.5,5.5+2.5],[QFED_min*1e-15,QFED_max*1e-15],color=colors[3])
plt.plot([5.5+2.1,5.5+2.9],[QFED_min*1e-15,QFED_min*1e-15],color=colors[3])
plt.plot([5.5+2.1,5.5+2.9],[QFED_max*1e-15,QFED_max*1e-15],color=colors[3])
#
# ----------
plot_section(post_MaySep_TROPOMI_GFED_3day_vec[4],post_MaySep_TROPOMI_GFAS_3day_vec[4],post_MaySep_TROPOMI_QFED_3day_vec[4],10.5,1)
plt.text(12.5,-20/1000.,'3 d opt\nw/out rep err',ha='center',va='top',fontsize=9)
# ----------
plot_section(post_MaySep_TROPOMI_rep_GFED_7day_vec[4],post_MaySep_TROPOMI_rep_GFAS_7day_vec[4],post_MaySep_TROPOMI_rep_QFED_7day_vec[4],15.5,1)
plt.text(17.5,-20/1000.,'7 d opt\nw/ rep err',ha='center',va='top',fontsize=9)
# ----------
plot_section(post_MaySep_TROPOMI_GFED_7day_vec[4],post_MaySep_TROPOMI_GFAS_7day_vec[4],post_MaySep_TROPOMI_QFED_7day_vec[4],20.5,1)
plt.text(22.5,-20/1000.,'7 d opt\nw/out rep err',ha='center',va='top',fontsize=9)
# ----------
#
plt.ylim([0,0.95])
plt.xlim([0,25])
plt.xticks([])
plt.ylabel('CO$_2$+CO May-Sep\nfire emissions (PgC)')

plt.savefig('Figures/Byrne_etal_FigSX_revision.png', dpi=300)



#
#
# ===== Plot Figure 1 =====
#
#


nc_file ='./data_for_figures/MERRA2.20220114.I3.2x25.nc4'
f=Dataset(nc_file,mode='r')
XMid=f.variables['lon'][:]
YMid=f.variables['lat'][:]
f.close()


fig = plt.figure(1115, figsize=(5*1.2,3*1.2), dpi=300)

f=Dataset('./data_for_figures/xCO_map_2019.nc',mode='r')
Y2019=f.variables['xCO'][:]
f.close()
f=Dataset('./data_for_figures/xCO_map_2020.nc',mode='r')
Y2020=f.variables['xCO'][:]
f.close()
f=Dataset('./data_for_figures/xCO_map_2021.nc',mode='r')
Y2021=f.variables['xCO'][:]
f.close()
f=Dataset('./data_for_figures/xCO_map_2022.nc',mode='r')
Y2022=f.variables['xCO'][:]
f.close()
f=Dataset('./data_for_figures/xCO_map_2023.nc',mode='r')
Y2023=f.variables['xCO'][:]
f.close()

Ymean = ( Y2019 + Y2020 + Y2021 + Y2022 ) / 4.

cmap1 = plt.cm.CMRmap_r
bounds1 = [70,80,90,100,110,120,130,140]
norm1 = mpl.colors.BoundaryNorm(bounds1, cmap1.N, extend='both')


m = Basemap(projection='mill',llcrnrlat=20,urcrnrlat=86,
            llcrnrlon=-170,urcrnrlon=75,resolution='c')
#                                                             
X,Y = np.meshgrid(XMid[3:141]-2.5/2.,YMid[35:91]-2./2.)
xx,yy=m(X,Y)
#                                                             
# --- Plot panel A ---
ax1 = fig.add_axes([0.1125, 0.75-0.075, 0.88/2., 0.22+0.075])
m.pcolormesh(xx,yy,ma.masked_invalid(Ymean[35:91,3:141]),cmap=cmap1, norm=norm1)
m.drawcoastlines()
plt.annotate('(a) 2019-22', xy=(0.005, 0.985), xycoords='axes fraction',va='top',ha='left')
#                                                             
# --- Plot panel B ---
ax1 = fig.add_axes([0.11+0.89/2., 0.75-0.075, 0.88/2., 0.22+0.075])
tt=m.pcolormesh(xx,yy,ma.masked_invalid(Y2023[35:91,3:141]),cmap=cmap1, norm=norm1)
m.drawcoastlines()
plt.annotate('(b) 2023', xy=(0.005, 0.985), xycoords='axes fraction',va='top',ha='left')
#                                                             
ax1 = fig.add_axes([0.115-0.01/2.-0.0125,0.75-0.075,0.0125,0.22+0.075])
cbar = plt.colorbar(tt,cax=ax1, orientation='vertical',extend='both')
cbar.ax.yaxis.set_ticks_position("left")
cbar.set_label('$\mathrm{X_{CO}}$ (ppb)')
cbar.ax.yaxis.set_label_position('left')
#
# --- Plot panel C ---
ax1 = fig.add_axes([0.115, 0.175-0.075, .5, 0.8*2./3.])
plt.fill_between([-2,6.5],[0,0],[900,900],color='b',alpha=0.05,linewidth=0.0)
plot_box_whiskers(1,prior_GFED_MaySep_vec[7:20]*1e-15,colors[1],0.7,0)
plot_box_whiskers(2,prior_GFAS_MaySep_vec[7:20]*1e-15,colors[2],0.7,0)
plot_box_whiskers(3,prior_QFED_MaySep_vec[7:20]*1e-15,colors[3],0.7,0)
#
optimized_combined = np.zeros(13)
optimized_combined[0:9] = post_MaySep_MOPITT_vec[0:9]
optimized_combined[9] = (post_MaySep_MOPITT_vec[9]+post_MaySep_TROPOMI_vec[0])/2.
optimized_combined[10] = (post_MaySep_MOPITT_vec[10]+post_MaySep_TROPOMI_vec[1])/2.
optimized_combined[11] = (post_MaySep_MOPITT_vec[11]+post_MaySep_TROPOMI_vec[2])/2.
optimized_combined[12] = post_MaySep_TROPOMI_vec[3]
plot_box_whiskers(5,optimized_combined*1e-15,colors[1],0.5,1)
#
l1=plt.bar(8,prior_GFED_MaySep_vec[20]*1e-15,color=colors[1],width=0.9,alpha=0.7)
l2=plt.bar(9,prior_GFAS_MaySep_vec[20]*1e-15,color=colors[2],width=0.9,alpha=0.7)
l3=plt.bar(10,prior_QFED_MaySep_vec[20]*1e-15,color=colors[3],width=0.9,alpha=0.7)
#
all_arr = np.array([prior_GFED_MaySep_vec[20]*1e-15,prior_GFAS_MaySep_vec[20]*1e-15,prior_QFED_MaySep_vec[20]*1e-15])
plt.plot(11,np.mean(all_arr),'o',color='k')
plt.plot([11,11],[np.min(all_arr),np.max(all_arr)],color='k',solid_capstyle='butt')
plt.plot([11-0.45,11+0.45],[np.min(all_arr),np.min(all_arr)],color='k',solid_capstyle='butt')
plt.plot([11-0.45,11+0.45],[np.max(all_arr),np.max(all_arr)],color='k',solid_capstyle='butt')
#
plt.bar(13,post_MaySep_TROPOMI_vec[4]*1e-15,color=colors[1],alpha=0.5,hatch="//",edgecolor=colors[1])
plt.bar(14,post_MaySep_TROPOMI_GFAS_vec*1e-15,color=colors[2],alpha=0.5,hatch="//",edgecolor=colors[2])
plt.bar(15,post_MaySep_TROPOMI_QFED_vec*1e-15,color=colors[3],alpha=0.5,hatch="//",edgecolor=colors[3])
plt.legend((l1[0],l2[0],l3[0]),('GFED','GFAS','QFED'),ncol=1,handletextpad=0.3,labelspacing=0.5,loc='upper left')
#
all_arr = np.array([post_MaySep_TROPOMI_vec[4]*1e-15,post_MaySep_TROPOMI_GFAS_vec*1e-15,post_MaySep_TROPOMI_QFED_vec*1e-15])
plt.plot(16,np.mean(all_arr),'ko')
plt.plot([16,16],[np.min(all_arr),np.max(all_arr)],'k',solid_capstyle='butt')
plt.plot([16-0.45,16+0.45],[np.min(all_arr),np.min(all_arr)],'k',solid_capstyle='butt')
plt.plot([16-0.45,16+0.45],[np.max(all_arr),np.max(all_arr)],'k',solid_capstyle='butt')
plt.ylim([0,850/1000.])
#
plt.xlim([0.,17.])
plt.xticks([])#

plt.text(17.*0.98,0.85*0.98,'(c)',va='top',ha='right')
#
plt.text(2,-20/1000.,'Bottom-up',ha='center',va='top',fontsize=9)
plt.text(5,-20/1000.,'TD',ha='center',va='top',fontsize=9)
plt.text(9,-20/1000.,'Bottom-up',ha='center',va='top',fontsize=9)
plt.text(14,-20/1000.,'Top-down',ha='center',va='top',fontsize=9)
plt.text(2.9,-125./1000.,'2010-22',ha='center',va='center',fontsize=9)
plt.text(11.65,-125./1000.,'2023',ha='center',va='center',fontsize=9)
ax1.annotate('', xy=(0.027, -0.14), xycoords='axes fraction', xytext=(0.088, -0.14),arrowprops=dict(arrowstyle="-", color='k'))
ax1.annotate('', xy=(0.258, -0.14), xycoords='axes fraction', xytext=(0.072+0.255, -0.14),arrowprops=dict(arrowstyle="-", color='k'))
ax1.annotate('', xy=(0.448, -0.14), xycoords='axes fraction', xytext=(0.623, -0.14),arrowprops=dict(arrowstyle="-", color='k'))
ax1.annotate('', xy=(0.748, -0.14), xycoords='axes fraction', xytext=(0.938, -0.14),arrowprops=dict(arrowstyle="-", color='k'))
plt.ylabel('CO$_2$+CO May-Sep\nfire emissions (PgC)')
#
#
# --- Plot panel D ---
#
# Terretorial Fossil Fuel emissions from Global Carbon Budget 2022
China_FF = 3131.11/1000.
USA_FF = 1366.63/1000.
India_FF = 739.54/1000.
# Canada Fire emissions
Russia_FF = 479.13/1000.
Japan_FF = 291.32/1000.
Iran_FF = 204.39/1000.
Germany_FF = 184.16/1000.
Indonesia_FF = 169.02/1000.
SK_FF = 168.14/1000.
Canada_FF = 148.92/1000.
#
ax1 = fig.add_axes([0.625+0.075, 0.175-0.075, .29, 0.8*2./3.])
l1=plt.bar(1,China_FF,color='k',width=0.9,alpha=0.2)
plt.text(1,0.05,'China',ha='center',va='bottom',rotation=90,fontsize=8)
plt.bar(2,USA_FF,color='k',width=0.9,alpha=0.2)
plt.text(2,0.05,'U.S.A.',ha='center',va='bottom',rotation=90,fontsize=8)
plt.bar(3,India_FF,color='k',width=0.9,alpha=0.2)
plt.text(3,0.05,'India',ha='center',va='bottom',rotation=90,fontsize=8)
#
all_arr = np.array([post_MaySep_TROPOMI_vec[4]*1e-12,post_MaySep_TROPOMI_GFAS_vec*1e-12,post_MaySep_TROPOMI_QFED_vec*1e-12]) * 1./1000.
l2=plt.bar(4,np.mean(all_arr),color='r',width=0.9,alpha=0.2,hatch="//",edgecolor='r')
plt.plot([4,4],[np.min(all_arr),np.max(all_arr)],'r',solid_capstyle='butt',alpha=0.75)
plt.plot([4-0.45,4+0.45],[np.min(all_arr),np.min(all_arr)],'r',solid_capstyle='butt',alpha=0.75)
plt.plot([4-0.45,4+0.45],[np.max(all_arr),np.max(all_arr)],'r',solid_capstyle='butt',alpha=0.75)
plt.text(4,np.max(all_arr)+0.05,'2023 Canada fires',ha='center',va='bottom',rotation=90,fontsize=8)
#
plt.bar(5,Russia_FF,color='k',width=0.9,alpha=0.2)
plt.text(5,Russia_FF+0.05,'Russia',ha='center',va='bottom',rotation=90,fontsize=8)
plt.bar(6,Japan_FF,color='k',width=0.9,alpha=0.2)
plt.text(6,Japan_FF+0.05,'Japan',ha='center',va='bottom',rotation=90,fontsize=8)
plt.bar(7,Iran_FF,color='k',width=0.9,alpha=0.2)
plt.text(7,Iran_FF+0.05,'Iran',ha='center',va='bottom',rotation=90,fontsize=8)
plt.bar(8,Germany_FF,color='k',width=0.9,alpha=0.2)
plt.text(8,Germany_FF+0.05,'Germany',ha='center',va='bottom',rotation=90,fontsize=8)
plt.bar(9,Indonesia_FF,color='k',width=0.9,alpha=0.2)
plt.text(9,Indonesia_FF+0.05,'Indonesia',ha='center',va='bottom',rotation=90,fontsize=8)
plt.bar(10,SK_FF,color='k',width=0.9,alpha=0.2)
plt.text(10,SK_FF+0.05,'South Korea',ha='center',va='bottom',rotation=90,fontsize=8)
plt.bar(11,Canada_FF,color='k',width=0.9,alpha=0.2)
plt.text(11,Canada_FF+0.05,'Canada',ha='center',va='bottom',rotation=90,fontsize=8)
plt.ylabel('carbon emissions ($\mathrm{PgC}$)')
plt.xticks([])
#
plt.ylim([0,3.2])
plt.xlim([0.4,11.6])
#
plt.text(0.4+(11.6-0.4)*0.98,3.2*0.98,'(d)',va='top',ha='right')
#
plt.legend((l1[0],l2[0]),('Fossil','Fire'),ncol=1,handletextpad=0.3,labelspacing=0.5,loc='upper right',bbox_to_anchor=(1.02, 0.98-0.06))

plt.savefig('Figures/Byrne_etal_Fig1_revision.png', dpi=300)





