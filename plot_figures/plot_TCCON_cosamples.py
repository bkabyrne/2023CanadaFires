# --- import modules ---
from mpl_toolkits.basemap import Basemap, cm
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import glob, os 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import interpolate

        

def read_TCCON(dir_name,site):
    
    if 'HX_all' in locals():
        del Y_all
        del HX_all
        del doy_all

    days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    doy_i = 0
    for mth in range(12):
        for ddd in range(days_in_month[mth]):
            #
            nc_file = dir_name+'/OBSF/'+site+'/2023/'+str(mth+1).zfill(2)+'/'+str(ddd+1).zfill(2)+'.nc'
            #
            if os.path.exists(nc_file):
                #
                f=Dataset(nc_file,mode='r')
                lon=f.variables['longitude'][:]
                lat=f.variables['latitude'][:]
                Y=f.variables['Y'][:] * 1e9
                HX=f.variables['HX'][:] * 1e9
                f.close()
                #
                if 'HX_all' in locals():
                    Y_all = np.append(Y_all,Y)
                    HX_all = np.append(HX_all,HX)
                    doy_all = np.append(doy_all,Y*0.+doy_i)
                else:
                    Y_all = Y
                    HX_all = HX
                    doy_all = Y*0.+doy_i
                #
                doy_i += doy_i

    return Y_all, HX_all, doy_all



def plot_TCCON_obs(ax1,Y_P,HX_P,Vmin,Vmax,Vvals,tlab,ylab='None',xlab='None'):
    # ---
    plt.plot([Vmin,Vmax],[Vmin,Vmax],':',color='grey')
    plt.plot(HX_P,Y_P,'k.',alpha=0.15,markeredgewidth=0.0)
    # ---
    plt.xlim([Vmin,Vmax])
    plt.ylim([Vmin,Vmax])
    # ---
    plt.xticks(Vvals)
    plt.yticks(Vvals)
    # ---
    if ylab != 'None':
        plt.ylabel(ylab)
    else:
        ax1.set_yticklabels([])
    # ---
    if xlab != 'None':
        plt.xlabel(xlab)
    else:
        ax1.set_xticklabels([])
    # ---
    #
    plt.text(Vmax*0.98,Vmin+(Vmax-Vmin)*0.21,'obs$-$model',ha='right',va='bottom')
    plt.text(Vmax*0.98,Vmin+(Vmax-Vmin)*0.11,'mean: '+str(np.round(np.mean(Y_P-HX_P)*10.)/10.),ha='right',va='bottom')
    plt.text(Vmax*0.98,Vmin+(Vmax-Vmin)*0.01,'std: '+str(np.round(np.std(Y_P-HX_P)*10.)/10.),ha='right',va='bottom')
    plt.text(Vmin+(Vmax-Vmin)*0.01,Vmax*0.98,tlab,va='top',ha='left')


def make_full_plot(base_dir1,base_dir2,TCCONpath,QFEDprior,QFEDpost,GFASprior,GFASpost,GFEDprior,GFEDpost,fignum,figname):

    Y_Prior_QFED, HX_Prior_QFED, doy_Prior_QFED = read_TCCON(base_dir1+QFEDprior,TCCONpath)
    Y_Post_QFED, HX_Post_QFED, doy_Post_QFED = read_TCCON(base_dir2+QFEDpost,TCCONpath)
    Y_Prior_GFAS, HX_Prior_GFAS, doy_Prior_GFAS = read_TCCON(base_dir1+GFASprior,TCCONpath)
    Y_Post_GFAS, HX_Post_GFAS, doy_Post_GFAS = read_TCCON(base_dir2+GFASpost,TCCONpath)
    Y_Prior_GFED, HX_Prior_GFED, doy_Prior_GFED = read_TCCON(base_dir1+GFEDprior,TCCONpath)
    Y_Post_GFED, HX_Post_GFED, doy_Post_GFED = read_TCCON(base_dir2+GFEDpost,TCCONpath)

    x_vec = np.array([50,280])
    fig = plt.figure(fignum,figsize=(5, 6), dpi=280)
    ax1 = fig.add_axes([0.13+0./2., 0.00+2./3., 0.8/2., 0.8/3.])
    plot_TCCON_obs(ax1,Y_Prior_GFED,HX_Prior_GFED,30,325,[50,100,150,200,250,300],'(ai) GFED prior',ylab='obs (ppb)')
    plt.title('Park Falls')
    ax1 = fig.add_axes([0.13+0./2., 0.04+1./3., 0.8/2., 0.8/3.])
    plot_TCCON_obs(ax1,Y_Prior_GFAS,HX_Prior_GFAS,30,325,[50,100,150,200,250,300],'(bi) GFAS prior',ylab='obs (ppb)')
    ax1 = fig.add_axes([0.13+0./2., 0.08+0./3., 0.8/2., 0.8/3.])
    plot_TCCON_obs(ax1,Y_Prior_QFED,HX_Prior_QFED,30,325,[50,100,150,200,250,300],'(ci) QFED prior',ylab='obs (ppb)',xlab='model (ppb)')
    ax1 = fig.add_axes([0.08+1./2., 0.00+2./3., 0.8/2., 0.8/3.])
    plot_TCCON_obs(ax1,Y_Post_GFED,HX_Post_GFED,30,325,[50,100,150,200,250,300],'(aii) GFED posterior')
    plt.title('Park Falls')
    ax1 = fig.add_axes([0.08+1./2., 0.04+1./3., 0.8/2., 0.8/3.])
    plot_TCCON_obs(ax1,Y_Post_GFAS,HX_Post_GFAS,30,325,[50,100,150,200,250,300],'(bii) GFAS posterior')
    ax1 = fig.add_axes([0.08+1./2., 0.08+0./3., 0.8/2., 0.8/3.])
    plot_TCCON_obs(ax1,Y_Post_QFED,HX_Post_QFED,30,325,[50,100,150,200,250,300],'(cii) QFED posterior',xlab='model (ppb)')
    plt.savefig('Figures/'+figname)
    plt.clf()


base_dir1 = '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-2023/'
base_dir2 = '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-2023/'

make_full_plot(base_dir1,base_dir2,'TCCON_PA_GGG2020_XCO','Prior_COinv_QFED_AprSep','Run_COinv_QFED_2023',
               'Prior_COinv_GFAS_AprSep','Run_COinv_GFAS_2023','Prior_COinv_GFED_AprSep','Run_COinv_GFED_2023',1,'TCCON_XCO_PA_3day.png')
make_full_plot(base_dir1,base_dir2,'TCCON_ETL_GGG2020_XCO','Prior_COinv_QFED_AprSep','Run_COinv_QFED_2023',
               'Prior_COinv_GFAS_AprSep','Run_COinv_GFAS_2023','Prior_COinv_GFED_AprSep','Run_COinv_GFED_2023',2,'TCCON_XCO_ETL_3day.png')

make_full_plot(base_dir1,base_dir2,'TCCON_PA_GGG2020_XCO','Prior_COinv_QFED_AprSep','Run_COinv_rep_QFED_2023',
               'Prior_COinv_GFAS_AprSep','Run_COinv_rep_GFAS_2023','Prior_COinv_GFED_AprSep','Run_COinv_rep_GFED_2023',3,'TCCON_XCO_PA_rep_3day.png')
make_full_plot(base_dir1,base_dir2,'TCCON_ETL_GGG2020_XCO','Prior_COinv_QFED_AprSep','Run_COinv_rep_QFED_2023',
               'Prior_COinv_GFAS_AprSep','Run_COinv_rep_GFAS_2023','Prior_COinv_GFED_AprSep','Run_COinv_rep_GFED_2023',4,'TCCON_XCO_ETL_rep_3day.png')



base_dir2 = '/nobackup/bbyrne1/GHGF-CMS-7day-COinv-2023/'

make_full_plot(base_dir1,base_dir2,'TCCON_PA_GGG2020_XCO','Prior_COinv_QFED_AprSep','Run_COinv_QFED_2023',
               'Prior_COinv_GFAS_AprSep','Run_COinv_GFAS_2023','Prior_COinv_GFED_AprSep','Run_COinv_GFED_2023',5,'TCCON_XCO_PA_7day.png')
make_full_plot(base_dir1,base_dir2,'TCCON_ETL_GGG2020_XCO','Prior_COinv_QFED_AprSep','Run_COinv_QFED_2023',
               'Prior_COinv_GFAS_AprSep','Run_COinv_GFAS_2023','Prior_COinv_GFED_AprSep','Run_COinv_GFED_2023',6,'TCCON_XCO_ETL_7day.png')

make_full_plot(base_dir1,base_dir2,'TCCON_PA_GGG2020_XCO','Prior_COinv_QFED_AprSep','Run_COinv_rep_QFED_2023',
               'Prior_COinv_GFAS_AprSep','Run_COinv_rep_GFAS_2023','Prior_COinv_GFED_AprSep','Run_COinv_rep_GFED_2023',7,'TCCON_XCO_PA_rep_7day.png')
make_full_plot(base_dir1,base_dir2,'TCCON_ETL_GGG2020_XCO','Prior_COinv_QFED_AprSep','Run_COinv_rep_QFED_2023',
               'Prior_COinv_GFAS_AprSep','Run_COinv_rep_GFAS_2023','Prior_COinv_GFED_AprSep','Run_COinv_rep_GFED_2023',8,'TCCON_XCO_ETL_rep_7day.png')


base_dir1 = '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/'
base_dir2 = '/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/'

make_full_plot(base_dir1,base_dir2,'TCCON_PA_GGG2020_XCO','QFED_prior_3day','QFED_post_3day',
               'GFAS_prior_3day','GFAS_post_3day','GFED_prior_3day','GFED_post_3day',1,'TCCON_XCO_PA_3day_injh.png')
make_full_plot(base_dir1,base_dir2,'TCCON_ETL_GGG2020_XCO','QFED_prior_3day','QFED_post_3day',
               'GFAS_prior_3day','GFAS_post_3day','GFED_prior_3day','GFED_post_3day',1,'TCCON_XCO_ETL_3day_injh.png')

make_full_plot(base_dir1,base_dir2,'TCCON_PA_GGG2020_XCO','QFED_rep_prior_3day','QFED_rep_post_3day',
               'GFAS_rep_prior_3day','GFAS_rep_post_3day','GFED_rep_prior_3day','GFED_rep_post_3day',1,'TCCON_XCO_PA_rep_3day_injh.png')
make_full_plot(base_dir1,base_dir2,'TCCON_ETL_GGG2020_XCO','QFED_rep_prior_3day','QFED_rep_post_3day',
               'GFAS_rep_prior_3day','GFAS_rep_post_3day','GFED_rep_prior_3day','GFED_rep_post_3day',1,'TCCON_XCO_ETL_rep_3day_injh.png')

make_full_plot(base_dir1,base_dir2,'TCCON_PA_GGG2020_XCO','QFED_prior_7day','QFED_post_7day',
               'GFAS_prior_7day','GFAS_post_7day','GFED_prior_7day','GFED_post_7day',1,'TCCON_XCO_PA_7day_injh.png')
make_full_plot(base_dir1,base_dir2,'TCCON_ETL_GGG2020_XCO','QFED_prior_7day','QFED_post_7day',
               'GFAS_prior_7day','GFAS_post_7day','GFED_prior_7day','GFED_post_7day',1,'TCCON_XCO_ETL_7day_injh.png')

make_full_plot(base_dir1,base_dir2,'TCCON_PA_GGG2020_XCO','QFED_rep_prior_7day','QFED_rep_post_7day',
               'GFAS_rep_prior_7day','GFAS_rep_post_7day','GFED_rep_prior_7day','GFED_rep_post_7day',1,'TCCON_XCO_PA_rep_7day_injh.png')
make_full_plot(base_dir1,base_dir2,'TCCON_ETL_GGG2020_XCO','QFED_rep_prior_7day','QFED_rep_post_7day',
               'GFAS_rep_prior_7day','GFAS_rep_post_7day','GFED_rep_prior_7day','GFED_rep_post_7day',1,'TCCON_XCO_ETL_rep_7day_injh.png')



# =====================================

# Plot mean across experiments

def plot_mean_TCCON_surface(figure_name,title_name,TCCONpath):
    # ===========
    Y_Prior_QFED, HX_Prior_QFED, doy_Prior_QFED = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-3day-COinv-2023/Prior_COinv_QFED_AprSep',TCCONpath)
    # --
    Y_Post_QFED_3day, HX_Post_QFED_3day, doy_Post_QFED_3day = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-3day-COinv-2023/Run_COinv_QFED_2023',TCCONpath)
    Y_Post_QFED_rep_3day, HX_Post_QFED_rep_3day, doy_Post_QFED_rep_3day = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-3day-COinv-2023/Run_COinv_rep_QFED_2023',TCCONpath)
    Y_Post_QFED_7day, HX_Post_QFED_7day, doy_Post_QFED_7day = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-7day-COinv-2023/Run_COinv_QFED_2023',TCCONpath)
    Y_Post_QFED_rep_7day, HX_Post_QFED_rep_7day, doy_Post_QFED_rep_7day = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-7day-COinv-2023/Run_COinv_rep_QFED_2023',TCCONpath)
    #--
    HX_Post_QFED = ( HX_Post_QFED_3day + HX_Post_QFED_rep_3day + HX_Post_QFED_7day + HX_Post_QFED_rep_7day ) /4.
    # ===========
    Y_Prior_GFED, HX_Prior_GFED, doy_Prior_GFED = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-3day-COinv-2023/Prior_COinv_GFED_AprSep',TCCONpath)
    # --
    Y_Post_GFED_3day, HX_Post_GFED_3day, doy_Post_GFED_3day = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-3day-COinv-2023/Run_COinv_GFED_2023',TCCONpath)
    Y_Post_GFED_rep_3day, HX_Post_GFED_rep_3day, doy_Post_GFED_rep_3day = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-3day-COinv-2023/Run_COinv_rep_GFED_2023',TCCONpath)
    Y_Post_GFED_7day, HX_Post_GFED_7day, doy_Post_GFED_7day = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-7day-COinv-2023/Run_COinv_GFED_2023',TCCONpath)
    Y_Post_GFED_rep_7day, HX_Post_GFED_rep_7day, doy_Post_GFED_rep_7day = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-7day-COinv-2023/Run_COinv_rep_GFED_2023',TCCONpath)
    #--
    HX_Post_GFED = ( HX_Post_GFED_3day + HX_Post_GFED_rep_3day + HX_Post_GFED_7day + HX_Post_GFED_rep_7day ) /4.
    # ===========
    Y_Prior_GFAS, HX_Prior_GFAS, doy_Prior_GFAS = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-3day-COinv-2023/Prior_COinv_GFAS_AprSep',TCCONpath)
    # --
    Y_Post_GFAS_3day, HX_Post_GFAS_3day, doy_Post_GFAS_3day = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-3day-COinv-2023/Run_COinv_GFAS_2023',TCCONpath)
    Y_Post_GFAS_rep_3day, HX_Post_GFAS_rep_3day, doy_Post_GFAS_rep_3day = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-3day-COinv-2023/Run_COinv_rep_GFAS_2023',TCCONpath)
    Y_Post_GFAS_7day, HX_Post_GFAS_7day, doy_Post_GFAS_7day = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-7day-COinv-2023/Run_COinv_GFAS_2023',TCCONpath)
    Y_Post_GFAS_rep_7day, HX_Post_GFAS_rep_7day, doy_Post_GFAS_rep_7day = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-7day-COinv-2023/Run_COinv_rep_GFAS_2023',TCCONpath)
    #--
    HX_Post_GFAS = ( HX_Post_GFAS_3day + HX_Post_GFAS_rep_3day + HX_Post_GFAS_7day + HX_Post_GFAS_rep_7day ) /4.
    # ===========
    
    x_vec = np.array([50,280])
    fig = plt.figure(100,figsize=(5, 6), dpi=280)
    ax1 = fig.add_axes([0.13+0./2., 0.00+2./3., 0.8/2., 0.8/3.])
    plot_TCCON_obs(ax1,Y_Prior_GFED,HX_Prior_GFED,30,325,[50,100,150,200,250,300],'(ai) GFED prior',ylab='obs (ppb)')
    plt.title(title_name)
    ax1 = fig.add_axes([0.13+0./2., 0.04+1./3., 0.8/2., 0.8/3.])
    plot_TCCON_obs(ax1,Y_Prior_GFAS,HX_Prior_GFAS,30,325,[50,100,150,200,250,300],'(bi) GFAS prior',ylab='obs (ppb)')
    ax1 = fig.add_axes([0.13+0./2., 0.08+0./3., 0.8/2., 0.8/3.])
    plot_TCCON_obs(ax1,Y_Prior_QFED,HX_Prior_QFED,30,325,[50,100,150,200,250,300],'(ci) QFED prior',ylab='obs (ppb)',xlab='model (ppb)')
    ax1 = fig.add_axes([0.08+1./2., 0.00+2./3., 0.8/2., 0.8/3.])
    plot_TCCON_obs(ax1,Y_Prior_GFED,HX_Post_GFED,30,325,[50,100,150,200,250,300],'(aii) GFED posterior')
    plt.title(title_name)
    ax1 = fig.add_axes([0.08+1./2., 0.04+1./3., 0.8/2., 0.8/3.])
    plot_TCCON_obs(ax1,Y_Prior_GFAS,HX_Post_GFAS,30,325,[50,100,150,200,250,300],'(bii) GFAS posterior')
    ax1 = fig.add_axes([0.08+1./2., 0.08+0./3., 0.8/2., 0.8/3.])
    plot_TCCON_obs(ax1,Y_Prior_QFED,HX_Post_QFED,30,325,[50,100,150,200,250,300],'(cii) QFED posterior',xlab='model (ppb)')
    plt.savefig('Figures/'+figure_name)
    plt.clf()

figure_name = 'TCCON_XCO_PA_mean.png'
title_name = 'Park Falls'
TCCONpath = 'TCCON_PA_GGG2020_XCO'
plot_mean_TCCON_surface(figure_name,title_name,TCCONpath)

figure_name = 'TCCON_XCO_ETL_mean.png'
title_name = 'East Trout Lake'
TCCONpath = 'TCCON_ETL_GGG2020_XCO'
plot_mean_TCCON_surface(figure_name,title_name,TCCONpath)

    

def plot_mean_TCCON_injection(figure_name,title_name,TCCONpath):

    # ===========
    Y_Prior_QFED, HX_Prior_QFED, doy_Prior_QFED = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/QFED_prior_3day',TCCONpath)
    # --
    Y_Post_QFED_3day, HX_Post_QFED_3day, doy_Post_QFED_3day = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/QFED_post_3day',TCCONpath)
    Y_Post_QFED_rep_3day, HX_Post_QFED_rep_3day, doy_Post_QFED_rep_3day = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/QFED_rep_post_3day',TCCONpath)
    Y_Post_QFED_7day, HX_Post_QFED_7day, doy_Post_QFED_7day = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/QFED_post_7day',TCCONpath)
    Y_Post_QFED_rep_7day, HX_Post_QFED_rep_7day, doy_Post_QFED_rep_7day = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/QFED_rep_post_7day',TCCONpath)
    #--
    HX_Post_QFED = ( HX_Post_QFED_3day + HX_Post_QFED_rep_3day + HX_Post_QFED_7day + HX_Post_QFED_rep_7day ) /4.
    # ===========
    Y_Prior_GFED, HX_Prior_GFED, doy_Prior_GFED = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/GFED_prior_3day',TCCONpath)
    # --
    Y_Post_GFED_3day, HX_Post_GFED_3day, doy_Post_GFED_3day = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/GFED_post_3day',TCCONpath)
    Y_Post_GFED_rep_3day, HX_Post_GFED_rep_3day, doy_Post_GFED_rep_3day = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/GFED_rep_post_3day',TCCONpath)
    Y_Post_GFED_7day, HX_Post_GFED_7day, doy_Post_GFED_7day = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/GFED_post_7day',TCCONpath)
    Y_Post_GFED_rep_7day, HX_Post_GFED_rep_7day, doy_Post_GFED_rep_7day = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/GFED_rep_post_7day',TCCONpath)
    #--
    HX_Post_GFED = ( HX_Post_GFED_3day + HX_Post_GFED_rep_3day + HX_Post_GFED_7day + HX_Post_GFED_rep_7day ) /4.
    # ===========
    Y_Prior_GFAS, HX_Prior_GFAS, doy_Prior_GFAS = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/GFAS_prior_3day',TCCONpath)
    # --
    Y_Post_GFAS_3day, HX_Post_GFAS_3day, doy_Post_GFAS_3day = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/GFAS_post_3day',TCCONpath)
    Y_Post_GFAS_rep_3day, HX_Post_GFAS_rep_3day, doy_Post_GFAS_rep_3day = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/GFAS_rep_post_3day',TCCONpath)
    Y_Post_GFAS_7day, HX_Post_GFAS_7day, doy_Post_GFAS_7day = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/GFAS_post_7day',TCCONpath)
    Y_Post_GFAS_rep_7day, HX_Post_GFAS_rep_7day, doy_Post_GFAS_rep_7day = read_TCCON('/nobackup/bbyrne1/GHGF-CMS-3day-COinv-injh/GFAS_rep_post_7day',TCCONpath)
    #--
    HX_Post_GFAS = ( HX_Post_GFAS_3day + HX_Post_GFAS_rep_3day + HX_Post_GFAS_7day + HX_Post_GFAS_rep_7day ) /4.
    # ===========
    
    x_vec = np.array([50,280])
    fig = plt.figure(100,figsize=(5, 6), dpi=280)
    ax1 = fig.add_axes([0.13+0./2., 0.00+2./3., 0.8/2., 0.8/3.])
    plot_TCCON_obs(ax1,Y_Prior_GFED,HX_Prior_GFED,30,325,[50,100,150,200,250,300],'(ai) GFED prior',ylab='obs (ppb)')
    plt.title(title_name)
    ax1 = fig.add_axes([0.13+0./2., 0.04+1./3., 0.8/2., 0.8/3.])
    plot_TCCON_obs(ax1,Y_Prior_GFAS,HX_Prior_GFAS,30,325,[50,100,150,200,250,300],'(bi) GFAS prior',ylab='obs (ppb)')
    ax1 = fig.add_axes([0.13+0./2., 0.08+0./3., 0.8/2., 0.8/3.])
    plot_TCCON_obs(ax1,Y_Prior_QFED,HX_Prior_QFED,30,325,[50,100,150,200,250,300],'(ci) QFED prior',ylab='obs (ppb)',xlab='model (ppb)')
    ax1 = fig.add_axes([0.08+1./2., 0.00+2./3., 0.8/2., 0.8/3.])
    plot_TCCON_obs(ax1,Y_Prior_GFED,HX_Post_GFED,30,325,[50,100,150,200,250,300],'(aii) GFED posterior')
    plt.title(title_name)
    ax1 = fig.add_axes([0.08+1./2., 0.04+1./3., 0.8/2., 0.8/3.])
    plot_TCCON_obs(ax1,Y_Prior_GFAS,HX_Post_GFAS,30,325,[50,100,150,200,250,300],'(bii) GFAS posterior')
    ax1 = fig.add_axes([0.08+1./2., 0.08+0./3., 0.8/2., 0.8/3.])
    plot_TCCON_obs(ax1,Y_Prior_QFED,HX_Post_QFED,30,325,[50,100,150,200,250,300],'(cii) QFED posterior',xlab='model (ppb)')
    plt.savefig('Figures/'+figure_name)
    plt.clf()

figure_name = 'TCCON_XCO_PA_inj_mean.png'
title_name = 'Park Falls'
TCCONpath = 'TCCON_PA_GGG2020_XCO'
plot_mean_TCCON_injection(figure_name,title_name,TCCONpath)

figure_name = 'TCCON_XCO_ETL_inj_mean.png'
title_name = 'East Trout Lake'
TCCONpath = 'TCCON_ETL_GGG2020_XCO'
plot_mean_TCCON_injection(figure_name,title_name,TCCONpath)

    
