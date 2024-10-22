#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 13:15:32 2023

@author: 6196306

"""
#Import packages
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import spei as si  # si for standardized index
import sys
sys.path.append("/home/6196306/Data/Python_scripts/")
from functions import MYD

# Load in data
SPEI_ind = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_IND.nc").__xarray_dataarray_variable__
SPEI_cal = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_CAL.nc").__xarray_dataarray_variable__
SPEI_aus = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_AUS.nc").__xarray_dataarray_variable__
SPEI_weu = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_WEU.nc").__xarray_dataarray_variable__
SPEI_sa = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_SA.nc").__xarray_dataarray_variable__
SPEI_arg = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_ARG.nc").__xarray_dataarray_variable__
SPEI_BRAH = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_BAH.nc").__xarray_dataarray_variable__
SPEI_IND_mean = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_IND_ens_mean.nc").__xarray_dataarray_variable__

#Caluclate MYDs
SPEI = [SPEI_cal, SPEI_weu, SPEI_ind, SPEI_arg, SPEI_sa, SPEI_aus]
region = ["CAL", "WEU", "IND", "ARG" , "SA", "AUS"]
MYDs = [MYD(SPEI_cal, "CAL"), MYD(SPEI_weu, "WEU"), MYD(SPEI_ind, "IND"), MYD(SPEI_arg, "ARG"), MYD(SPEI_sa, "SA"), MYD(SPEI_aus, "AUS")]

#%% Plot SPEI-12
fig, axes = plt.subplots(6,1, sharex=True, figsize=(12,12), tight_layout=True)
for i, ax in enumerate(axes):
    spei_pd = SPEI[i].to_pandas()
    si.plot.si(spei_pd[11:], ax=ax)
    ax.axhline(y=-1, color="red", linestyle="dashed", linewidth=0.8, alpha=0.5)
    ax.axhline(y=1, color="blue", linestyle="dashed", linewidth=0.8, alpha=0.5)
    start_date = MYDs[i][0]
    end_date = MYDs[i][1]
    for start, end in zip(start_date, end_date):
        spei_subset = SPEI[i].sel(time=slice(start, end))
        if (spei_subset <= -1).any():
            ax.fill_between(spei_subset.time, spei_subset, y2=-1, color='red', alpha=0.3, zorder=2)
            ax.plot(spei_subset.time, np.full(len(spei_subset.time), -1), color="black", zorder=3)
            ax.fill_between(spei_subset.time, y1=3, y2=-3, color="gold", alpha=0.5, zorder=1)
    ax.tick_params(labelsize=14)
    ax.set_title(str(region[i]))
fig.supylabel("SPEI-12 [-]", fontsize=14)
fig.savefig("/home/6196306/Data/Figures/1950-2023/SPEI-12_all_regions_v5.jpg", dpi=1200)
fig.savefig("/home/6196306/Data/Figures/1950-2023/SPEI-12_all_regions_v5.pdf")
