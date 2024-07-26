#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:05:53 2024

@author: 6196306
SPEI3 and SPEI6
"""
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import sys
sys.path.append("/home/6196306/Data/Python_scripts/")
from functions import MYD

#%% Load in data
#SPEI12
SPEI12_ind = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_IND.nc").__xarray_dataarray_variable__
SPEI12_cal = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_CAL.nc").__xarray_dataarray_variable__
SPEI12_aus = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_AUS.nc").__xarray_dataarray_variable__
SPEI12_weu = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_WEU.nc").__xarray_dataarray_variable__
SPEI12_sa = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_SA.nc").__xarray_dataarray_variable__
SPEI12_arg = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_ARG.nc").__xarray_dataarray_variable__

#SPEI6
SPEI6_ind = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI06_monthly_1950_2023_0_5_degree_IND.nc").__xarray_dataarray_variable__
SPEI6_cal = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI06_monthly_1950_2023_0_5_degree_CAL.nc").__xarray_dataarray_variable__
SPEI6_aus = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI06_monthly_1950_2023_0_5_degree_AUS.nc").__xarray_dataarray_variable__
SPEI6_weu = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI06_monthly_1950_2023_0_5_degree_WEU.nc").__xarray_dataarray_variable__
SPEI6_sa = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI06_monthly_1950_2023_0_5_degree_SA.nc").__xarray_dataarray_variable__
SPEI6_arg = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI06_monthly_1950_2023_0_5_degree_ARG.nc").__xarray_dataarray_variable__

#SPEI3
SPEI3_ind = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI03_monthly_1950_2023_0_5_degree_IND.nc").__xarray_dataarray_variable__
SPEI3_cal = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI03_monthly_1950_2023_0_5_degree_CAL.nc").__xarray_dataarray_variable__
SPEI3_aus = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI03_monthly_1950_2023_0_5_degree_AUS.nc").__xarray_dataarray_variable__
SPEI3_weu = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI03_monthly_1950_2023_0_5_degree_WEU.nc").__xarray_dataarray_variable__
SPEI3_sa = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI03_monthly_1950_2023_0_5_degree_SA.nc").__xarray_dataarray_variable__
SPEI3_arg = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI03_monthly_1950_2023_0_5_degree_ARG.nc").__xarray_dataarray_variable__

#SPEI1
SPEI1_ind = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/regions/SPEI01_monthly_1950_2023_0_5_degree_IND.nc").__xarray_dataarray_variable__
SPEI1_cal = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/regions/SPEI01_monthly_1950_2023_0_5_degree_CAL.nc").__xarray_dataarray_variable__
SPEI1_aus = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/regions/SPEI01_monthly_1950_2023_0_5_degree_AUS.nc").__xarray_dataarray_variable__
SPEI1_weu = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/regions/SPEI01_monthly_1950_2023_0_5_degree_WEU.nc").__xarray_dataarray_variable__
SPEI1_sa = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/regions/SPEI01_monthly_1950_2023_0_5_degree_SA.nc").__xarray_dataarray_variable__
SPEI1_arg = xr.open_dataset("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/regions/SPEI01_monthly_1950_2023_0_5_degree_ARG.nc").__xarray_dataarray_variable__

#Put in one dataarray
SPEI_cal = xr.concat([SPEI1_cal, SPEI3_cal, SPEI6_cal, SPEI12_cal], dim="SPEI")
SPEI_weu = xr.concat([SPEI1_weu, SPEI3_weu, SPEI6_weu, SPEI12_weu], dim="SPEI")
SPEI_ind = xr.concat([SPEI1_ind, SPEI3_ind, SPEI6_ind, SPEI12_ind], dim="SPEI")
SPEI_arg = xr.concat([SPEI1_arg, SPEI3_arg, SPEI6_arg, SPEI12_arg], dim="SPEI")
SPEI_sa = xr.concat([SPEI1_sa, SPEI3_sa, SPEI6_sa, SPEI12_sa], dim="SPEI")
SPEI_aus = xr.concat([SPEI1_aus, SPEI3_aus, SPEI6_aus, SPEI12_aus], dim="SPEI")

#Calculate MYDs
MYD_cal = MYD(SPEI12_cal, "CAL")
MYD_weu = MYD(SPEI12_weu, "WEU")
MYD_ind = MYD(SPEI12_ind, "IND")
MYD_arg = MYD(SPEI12_arg, "ARG")
MYD_sa = MYD(SPEI12_sa, "SA")
MYD_aus = MYD(SPEI12_aus, "AUS")

#Combine everything in one list
SPEI = [SPEI_cal, SPEI_weu, SPEI_ind, SPEI_arg, SPEI_sa, SPEI_aus]
afk = ["CAL", "WEU", "IND", "ARG", "SA", "AUS"]
region = ["California", "Western Europe", "India", "Argentina", "South Africa", "Australia"]
MYD_reg = [MYD_cal, MYD_weu, MYD_ind, MYD_arg, MYD_sa, MYD_aus]

#%% Selection of four MYD
fig, ax = plt.subplots(2,2, figsize=(12,4), sharey=True, gridspec_kw={"hspace":0.7, "wspace":0.1})
#India, 1979-1980
im=SPEI[2].where(SPEI[2]<=-1).sel(time=slice(MYD_reg[2][0][0].values-pd.DateOffset(months=12), MYD_reg[2][1][0].values+pd.DateOffset(months=12))).plot(ax=ax[0,0], cmap="Reds_r", vmin=-3, vmax=-1, add_colorbar=False)
ax[0,0].axvline(x=MYD_reg[2][0][0].values, color="black")
ax[0,0].axvline(x=MYD_reg[2][1][0].values, color="black")
ax[0,0].set_title("India")
ax[0,0].set_title("a)", loc="left")
#Western Europe, 1975-1976
SPEI[1].where(SPEI[1]<=-1).sel(time=slice(MYD_reg[1][0][1].values-pd.DateOffset(months=12), MYD_reg[1][1][1].values+pd.DateOffset(months=12))).plot(ax=ax[0,1], cmap="Reds_r", vmin=-3, vmax=-1, add_colorbar=False)
ax[0,1].axvline(x=MYD_reg[1][0][1].values, color="black")
ax[0,1].axvline(x=MYD_reg[1][1][1].values, color="black")
ax[0,1].set_title("Western Europe")
ax[0,1].set_title("b)", loc="left")
#Australia, 2002-2003 -> deze misschien nog vervangen door een langere MYD?
SPEI[3].where(SPEI[3]<=-1).sel(time=slice(MYD_reg[3][0][1].values-pd.DateOffset(months=12), MYD_reg[3][1][1].values+pd.DateOffset(months=12))).plot(ax=ax[1,0], cmap="Reds_r", vmin=-3, vmax=-1, add_colorbar=False)
ax[1,0].axvline(x=MYD_reg[3][0][1].values, color="black")
ax[1,0].axvline(x=MYD_reg[3][1][1].values, color="black")
ax[1,0].set_title("Argentina")
ax[1,0].set_title("c)", loc="left")
#California, 2013-2016
SPEI[0].where(SPEI[0]<=-1).sel(time=slice(MYD_reg[0][0][0].values-pd.DateOffset(months=12), MYD_reg[0][1][0].values+pd.DateOffset(months=12))).plot(ax=ax[1,1], cmap="Reds_r", vmin=-3, vmax=-1, add_colorbar=False)
ax[1,1].axvline(x=MYD_reg[0][0][0].values, color="black")
ax[1,1].axvline(x=MYD_reg[0][1][0].values, color="black")
ax[1,1].set_title("California")
ax[1,1].set_title("d)", loc="left")

for ax in ax.flatten():
    ax.set_yticks([0, 1, 2, 3], [1, 3, 6, 12])
    ax.set_ylabel(" ")
    ax.set_xlabel(" ")

fig.supylabel("SPEI-timescale", x=0.07)
cbar_ax = fig.add_axes([0.93, 0.12, 0.02, 0.76])  #left, bottom, width, height
cbar = plt.colorbar(im, cax=cbar_ax, label="SPEI [-]")
fig.savefig("/home/6196306/Data/Figures/1950-2023/SPEI-1-3-6-12_four_examples_v2.pdf", bbox_inches="tight")
fig.savefig("/home/6196306/Data/Figures/1950-2023/SPEI-1-3-6-12_four_examples_v2.jpg", bbox_inches="tight", dpi=1200)
