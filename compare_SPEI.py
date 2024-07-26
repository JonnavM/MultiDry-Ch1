#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:00:05 2024

@author: 6196306
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/6196306/Data/Python_scripts/")
from functions import MYD
import pandas as pd

#Import datasets
#ERA5 1980-2023
SPEI_era_cal = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/ERA5_different_calibrations/1980_2020/SPEI12_monthly_1950_2023_0_5_degree_CAL.nc").sel(time=slice("1980", "2023"))
SPEI_era_weu = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/ERA5_different_calibrations/1980_2020/SPEI12_monthly_1950_2023_0_5_degree_WEU.nc").sel(time=slice("1980", "2023"))
SPEI_era_ind = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/ERA5_different_calibrations/1980_2020/SPEI12_monthly_1950_2023_0_5_degree_IND.nc").sel(time=slice("1980", "2023"))
SPEI_era_arg = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/ERA5_different_calibrations/1980_2020/SPEI12_monthly_1950_2023_0_5_degree_ARG.nc").sel(time=slice("1980", "2023"))
SPEI_era_sa = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/ERA5_different_calibrations/1980_2020/SPEI12_monthly_1950_2023_0_5_degree_SA.nc").sel(time=slice("1980", "2023"))
SPEI_era_aus = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/ERA5_different_calibrations/1980_2020/SPEI12_monthly_1950_2023_0_5_degree_AUS.nc").sel(time=slice("1980", "2023"))
SPEI_era_bah = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/ERA5_different_calibrations/1980_2020/SPEI12_monthly_1950_2023_0_5_degree_BAH.nc").sel(time=slice("1980", "2023"))

#MERRA2 1980-2023
SPEI_merra_cal = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/MERRA2/SPEI12_monthly_1980_2023_0_5_degree_CAL.nc")
SPEI_merra_weu = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/MERRA2/SPEI12_monthly_1980_2023_0_5_degree_WEU.nc")
SPEI_merra_ind = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/MERRA2/SPEI12_monthly_1980_2023_0_5_degree_IND.nc")
SPEI_merra_arg = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/MERRA2/SPEI12_monthly_1980_2023_0_5_degree_ARG.nc")
SPEI_merra_sa = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/MERRA2/SPEI12_monthly_1980_2023_0_5_degree_SA.nc")
SPEI_merra_aus = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/MERRA2/SPEI12_monthly_1980_2023_0_5_degree_AUS.nc")
SPEI_merra_bah = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/MERRA2/SPEI12_monthly_1980_2023_0_5_degree_BAH.nc")

#JRA-3Q 1980-2023
SPEI_jra_cal = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/JRA3Q/1980_2020_calibration/SPEI12_monthly_1980_2023_0_5_degree_CAL.nc").sel(time=slice("1980", "2023"))
SPEI_jra_weu = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/JRA3Q/1980_2020_calibration/SPEI12_monthly_1980_2023_0_5_degree_WEU.nc").sel(time=slice("1980", "2023"))
SPEI_jra_ind = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/JRA3Q/1980_2020_calibration/SPEI12_monthly_1980_2023_0_5_degree_IND.nc").sel(time=slice("1980", "2023"))
SPEI_jra_arg = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/JRA3Q/1980_2020_calibration/SPEI12_monthly_1980_2023_0_5_degree_ARG.nc").sel(time=slice("1980", "2023"))
SPEI_jra_sa = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/JRA3Q/1980_2020_calibration/SPEI12_monthly_1980_2023_0_5_degree_SA.nc").sel(time=slice("1980", "2023"))
SPEI_jra_aus = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/JRA3Q/1980_2020_calibration/SPEI12_monthly_1980_2023_0_5_degree_AUS.nc").sel(time=slice("1980", "2023"))
SPEI_jra_bah = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/JRA3Q/1980_2020_calibration/SPEI12_monthly_1980_2023_0_5_degree_BAH.nc").sel(time=slice("1980", "2023"))
 
# Put in lists
SPEI_era = [SPEI_era_cal, SPEI_era_weu, SPEI_era_ind, SPEI_era_arg, SPEI_era_sa, SPEI_era_aus, SPEI_era_bah]
SPEI_merra = [SPEI_merra_cal, SPEI_merra_weu, SPEI_merra_ind, SPEI_merra_arg, SPEI_merra_sa, SPEI_merra_aus, SPEI_merra_bah]
SPEI_jra = [SPEI_jra_cal, SPEI_jra_weu, SPEI_jra_ind, SPEI_jra_arg, SPEI_jra_sa, SPEI_jra_aus, SPEI_jra_bah]
region = ["CAL", "WEU", "IND", "ARG" , "SA", "AUS", "BAH"]

#Calculate MYDs
MYDs_era = [MYD(SPEI_era_cal, "CAL"), MYD(SPEI_era_weu, "WEU"), MYD(SPEI_era_ind, "IND"), MYD(SPEI_era_arg, "ARG"), MYD(SPEI_era_sa, "SA"), MYD(SPEI_era_aus, "AUS"), MYD(SPEI_era_bah, "BAH")]
MYDs_merra = [MYD(SPEI_merra_cal, "CAL"), MYD(SPEI_merra_weu, "WEU"), MYD(SPEI_merra_ind, "IND"), MYD(SPEI_merra_arg, "ARG"), MYD(SPEI_merra_sa, "SA"), MYD(SPEI_merra_aus, "AUS"), MYD(SPEI_merra_bah, "BAH")]
MYDs_jra = [MYD(SPEI_jra_cal, "CAL"), MYD(SPEI_jra_weu, "WEU"), MYD(SPEI_jra_ind, "IND"), MYD(SPEI_jra_arg, "ARG"), MYD(SPEI_jra_sa, "SA"), MYD(SPEI_jra_aus, "AUS"), MYD(SPEI_jra_bah, "BAH")]

#%% Plot SPEI for ERA, MERRA, and JRA
for i in range(len(SPEI_era)):
    fig, ax = plt.subplots(3, sharex=True, sharey=True, figsize=(12,8), tight_layout=True)
    #Convert to Pandas
    spei_era_pd = SPEI_era[i].to_pandas()
    spei_merra_pd = SPEI_merra[i].to_pandas()
    spei_jra_pd = SPEI_jra[i].to_pandas()
    spei_df = pd.DataFrame({
    'ERA5': spei_era_pd,
    'MERRA-2': spei_merra_pd,
    'JRA-3Q': spei_jra_pd
})
    spei_mean = spei_df.mean(axis=1)
    spei_min = spei_df.min(axis=1)
    spei_max = spei_df.max(axis=1)

    for axes in ax:
        axes.fill_between(spei_mean[11:].index, spei_max[11:], spei_min[11:], alpha=0.3, color="grey")
        axes.plot(spei_mean[11:].index, spei_mean[11:], color="black", linewidth=0.8, alpha=0.6)
    ax[0].plot(spei_era_pd, color="black", linewidth=1, zorder=2)
    ax[1].plot(spei_merra_pd, color="black", linewidth=1, zorder=2)
    ax[2].plot(spei_jra_pd, color="black", linewidth=1, zorder=2)

    #Add horizontal lines for the MYDs
    start_date_era = MYDs_era[i][0]
    end_date_era = MYDs_era[i][1]
    start_date_merra = MYDs_merra[i][0]
    end_date_merra = MYDs_merra[i][1]
    start_date_jra = MYDs_jra[i][0]
    end_date_jra = MYDs_jra[i][1]
    for start, end in zip(start_date_era, end_date_era):
        spei_subset_era = SPEI_era[i].sel(time=slice(start, end))
        if (spei_subset_era <= -1).any():
            ax[0].fill_between(spei_subset_era.time, spei_subset_era, y2=-1, color='red', zorder=1, alpha=0.5)
            ax[0].plot(spei_subset_era.time, np.full(len(spei_subset_era.time), -1), color="black", zorder=1)
    for start, end in zip(start_date_merra, end_date_merra):
        spei_subset_merra = SPEI_merra[i].sel(time=slice(start, end))
        if (spei_subset_merra <= -1).any():
            ax[1].fill_between(spei_subset_merra.time, spei_subset_merra, y2=-1, color='red', zorder=1, alpha=0.5)
            ax[1].plot(spei_subset_merra.time, np.full(len(spei_subset_merra.time), -1), color="black", zorder=1)
    for start, end in zip(start_date_jra, end_date_jra):
        spei_subset_jra = SPEI_jra[i].sel(time=slice(start, end))
        if (spei_subset_jra <= -1).any():
            ax[2].fill_between(spei_subset_jra.time, spei_subset_jra, y2=-1, color='red', zorder=1, alpha=0.5)
            ax[2].plot(spei_subset_jra.time, np.full(len(spei_subset_jra.time), -1), color="black", zorder=1)
    
    #Add titles
    ax[0].set_title(str(region[i]))
        
    fig.supylabel("SPEI-12 [-]", fontsize=14)
    ax[0].set_ylabel("ERA5", fontsize=14)
    ax[1].set_ylabel("MERRA2", fontsize=14)
    ax[2].set_ylabel("JRA-3Q", fontsize=14)
    
    for ax in ax[0:4]:
        ax.axhline(y=0, color="black", linestyle="dashed", linewidth=0.9)
        ax.axhline(y=-1, color="red", linestyle="dashed", linewidth=0.8, alpha=0.5)
        ax.axhline(y=1, color="blue", linestyle="dashed", linewidth=0.8, alpha=0.5)
        ax.tick_params(labelsize=14)
        
    fig.savefig("/home/6196306/Data/Figures/1950-2023/SPEI-12_ERA_MERRA_JRA_1980-2020_"+str(region[i])+"_v5.jpg", dpi=1200)
    fig.savefig("/home/6196306/Data/Figures/1950-2023/SPEI-12_ERA_MERRA_JRA_1980-2020_"+str(region[i])+"_v5.pdf")