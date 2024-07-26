#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:38:05 2024

@author: 6196306
Calculate linear regressions
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress
import sys
sys.path.append("/home/6196306/Data/Python_scripts/")
from functions import MYD, ND, mask_MYD, mask_ND
from scipy.stats import gaussian_kde
import matplotlib.gridspec as gridspec

#Data
SPEI_AUS = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_AUS.nc").sel(time=slice("1951", "2023"))
SPEI_WEU = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_WEU.nc").sel(time=slice("1951", "2023"))
SPEI_CAL = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_CAL.nc").sel(time=slice("1951", "2023"))
SPEI_IND = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_IND.nc").sel(time=slice("1951", "2023"))
SPEI_SA = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_SA.nc").sel(time=slice("1951", "2023"))
SPEI_ARG = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_ARG.nc").sel(time=slice("1951", "2023"))

#Complete grid
SPEI_grid = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree.nc")#.sel(time=slice("1956", "2022"))

# Masks
mask_AUS = xr.open_dataarray("/scratch/6196306/masks/mask_AUS.nc")
mask_WEU = xr.open_dataarray("/scratch/6196306/masks/mask_WEU.nc")
mask_IND = xr.open_dataarray("/scratch/6196306/masks/mask_IND.nc")
mask_SA = xr.open_dataarray("/scratch/6196306/masks/mask_SA.nc") 
mask_CAL = xr.open_dataarray("/scratch/6196306/masks/mask_CAL.nc")
mask_ARG = xr.open_dataarray("/scratch/6196306/masks/mask_ARG.nc")

#Australia
MYD_AUS = MYD(SPEI_AUS, "AUS")
ND_AUS = ND(SPEI_AUS, "AUS")
mask_MYD_AUS = mask_MYD(SPEI_AUS, "AUS")
mask_ND_AUS = mask_ND(SPEI_AUS, "AUS")

#South Africa
MYD_SA = MYD(SPEI_SA, "SA")
ND_SA = ND(SPEI_SA, "SA")
mask_MYD_SA = mask_MYD(SPEI_SA, "SA")
mask_ND_SA = mask_ND(SPEI_SA, "SA")

#California
MYD_CAL = MYD(SPEI_CAL, "CAL")
ND_CAL = ND(SPEI_CAL, "CAL")
mask_MYD_CAL = mask_MYD(SPEI_CAL, "CAL")
mask_ND_CAL = mask_ND(SPEI_CAL, "CAL")
      
#Western Europe
MYD_WEU = MYD(SPEI_WEU, "WEU")
ND_WEU = ND(SPEI_WEU, "WEU")
mask_MYD_WEU = mask_MYD(SPEI_WEU, "WEU")
mask_ND_WEU = mask_ND(SPEI_WEU, "WEU")

#Middle Argentina
MYD_ARG = MYD(SPEI_ARG, "ARG")
ND_ARG = ND(SPEI_ARG, "ARG")
mask_MYD_ARG = mask_MYD(SPEI_ARG, "ARG")
mask_ND_ARG = mask_ND(SPEI_ARG, "ARG")

#India
MYD_IND = MYD(SPEI_IND, "IND")
ND_IND = ND(SPEI_IND, "IND")
mask_MYD_IND = mask_MYD(SPEI_IND, "IND")
mask_ND_IND = mask_ND(SPEI_IND, "IND")

#Lats and lons per region
lat_WEU = slice(45, 55) 
lon_WEU = slice(-1, 13)
lat_IND = slice(22, 32)
lon_IND = slice(72, 90)
lat_AUS = slice(-40, -20)
lon_AUS = slice(135, 155)
lat_SA = slice(-33, -21)
lon_SA = slice(15, 31)
lat_ARG = slice(-45, -25)
lon_ARG = slice(-75, -55)
lat_CAL = slice(32, 41.5)
lon_CAL = slice(-124, -115)

#%% Download other variables
pr = xr.open_dataarray("/scratch/6196306/ERA5/total_precipitation/era5_total_precipitation_1950-2023_monthly_0_5.nc")*1000#.sel(time=slice("1950", "2022"), expver=1)*1000
pet = xr.open_dataset("/scratch/6196306/PET/PenmanMonteith/pm_fao56_1950-2023_monthly_0_5_v3.nc").PM_FAO_56
pet["time"]=pr["time"]

#Rolling mean
def rolling12(data):
    return data.rolling(time=12).mean("time").sel(time=slice("1951", "2023"))

#Anomaly
def std_anom(data):
    climatology_mean = data.groupby('time.month').mean('time')
    climatology_std = data.groupby('time.month').std('time')
    climatology_std = climatology_std.where(climatology_std != 0, float('nan'))
    stand_anomalies = xr.apply_ufunc(lambda x, m, s: (x - m) / s, data.groupby('time.month'), climatology_mean, climatology_std,dask = 'allowed', vectorize = True)
    print ('anomalies done')
    return stand_anomalies

SPEI_reg = [SPEI_CAL, SPEI_WEU, SPEI_IND, SPEI_ARG, SPEI_SA, SPEI_AUS]
mask_reg = [mask_CAL, mask_WEU, mask_IND, mask_ARG, mask_SA, mask_AUS] 
mask_MYD_reg = [mask_MYD_CAL, mask_MYD_WEU, mask_MYD_IND, mask_MYD_ARG, mask_MYD_SA, mask_MYD_AUS]
MYD_reg = [MYD_CAL, MYD_WEU, MYD_IND, MYD_ARG, MYD_SA, MYD_AUS]
ND_reg = [ND_CAL, ND_WEU, ND_IND, ND_ARG, ND_SA, ND_AUS]
title_reg = ["California", "Western Europe", "India", "Argentina", "South Africa", "Australia"]
afk = ["CAL", "WEU", "IND", "ARG", "SA", "AUS"]

#Change units to mm/month
days_in_month = pr.time.dt.days_in_month
pr_month = pr*days_in_month
pet_month = pet*days_in_month
pr_month.attrs['units'] = 'mm/month'
pet_month.attrs['units'] = 'mm/month'

#%% Composite plot of PET and PR for NDs and MYDs
fig = plt.figure(layout='constrained', figsize=(12, 8))
subfigs = fig.subfigures(2,3, wspace=0)
abc = ["a)", "b)", "c)", "d)", "e)", "f)"]

for i in range(len(MYD_reg)):
    pr_12_rol = rolling12(pr_month.where(mask_reg[i]==1).mean(dim=("lat", "lon"))) #try anomalies
    pet_12_rol = rolling12(pet_month.where(mask_reg[i]==1).mean(dim=("lat", "lon")))
    
    # Standardize data
    pr_12 = (pr_12_rol - pr_12_rol.mean()) 
    pet_12 = (pet_12_rol - pet_12_rol.mean())
    
    #Select non-droughts, droughts, and MYDs
    pr_drought = pr_12.where((mask_MYD_reg[i]==False) & (SPEI_reg[i]<=-1))
    pr_MYD = pr_12.where(mask_MYD_reg[i]==True)
    
    SPEI_drought = SPEI_reg[i].where((mask_MYD_reg[i]==False) & (SPEI_reg[i]<=-1))
    SPEI_MYD = SPEI_reg[i].where(mask_MYD_reg[i]==True)
    
    pet_drought = pet_12.where((mask_MYD_reg[i]==False) & (SPEI_reg[i]<=-1))
    pet_MYD = pet_12.where(mask_MYD_reg[i]==True)
    
    #Make list
    length = []
    length_ND = []
    for j, start in enumerate(MYD_reg[i][0]):
        end = MYD_reg[i][1][j]
        myd_in_slice = (SPEI_MYD.time >= start.values) & (SPEI_MYD.time <= end.values)
        if myd_in_slice.any():
            length.append(myd_in_slice.sum())
            
    for j, start in enumerate(ND_reg[i][0]):
        end = ND_reg[i][1][j]
        nd_in_slice = (SPEI_drought.time >= start.values) & (SPEI_drought.time <= end.values)
        if nd_in_slice.any():
            length_ND.append(nd_in_slice.sum())
    
    #Create empty arrays for both ND and MYDs
    pet_MYD_matrix = np.full([len(MYD_reg[i][0]), np.max(length)], np.nan)
    pr_MYD_matrix = np.full([len(MYD_reg[i][0]), np.max(length)], np.nan)
    pet_ND_matrix = np.full([len(ND_reg[i][0]), np.max(length_ND)], np.nan)
    pr_ND_matrix = np.full([len(ND_reg[i][0]), np.max(length_ND)], np.nan)
    
    for j, start in enumerate(MYD_reg[i][0]):
        end = MYD_reg[i][1][j]
        myd_in_slice = (SPEI_MYD.time >= start.values) & (SPEI_MYD.time <= end.values)
        if myd_in_slice.any():
            # Get the indices where myd_in_slice is True
            indices = np.where(myd_in_slice)[0]
            # Assign values to the corresponding positions in the matrix
            pet_MYD_matrix[j][:len(indices)] = pet_MYD[myd_in_slice][:]
            pr_MYD_matrix[j][:len(indices)] = pr_MYD[myd_in_slice][:]
            
    for j, start in enumerate(ND_reg[i][0]):
        end = ND_reg[i][1][j]
        print(end)
        nd_in_slice = (SPEI_drought.time >= start.values) & (SPEI_drought.time <= end.values)
        if nd_in_slice.any():
            # Get the indices where myd_in_slice is True
            indices = np.where(nd_in_slice)[0]
            # Assign values to the corresponding positions in the matrix
            pet_ND_matrix[j][:len(indices)] = pet_drought[nd_in_slice][:]
            pr_ND_matrix[j][:len(indices)] = pr_drought[nd_in_slice][:]
            
    #Take the mean per time step
    mean_pet_MYD = np.nanmean(pet_MYD_matrix, axis=0)
    mean_pr_MYD = np.nanmean(pr_MYD_matrix, axis=0)
    mean_pet_ND = np.nanmean(pet_ND_matrix, axis=0)
    mean_pr_ND = np.nanmean(pr_ND_matrix, axis=0)
    #Same for standard devations
    std_pet_MYD = np.nanstd(pet_MYD_matrix, axis=0)
    std_pr_MYD = np.nanstd(pr_MYD_matrix, axis=0)
    std_pet_ND = np.nanstd(pet_ND_matrix, axis=0)
    std_pr_ND = np.nanstd(pr_ND_matrix, axis=0)
    
    # Plot PR and PET separate from each other
    x = np.arange(0, len(mean_pet_MYD), 1)
    x2 = np.arange(0, len(mean_pet_ND), 1)
    x_shorter = np.arange(0, np.sort(length)[-2])
    x2_shorter = np.arange(0, np.sort(length_ND)[-2])
    
    ax2 = subfigs.flatten()[i].subplots(2, 1, sharex=True)
    ax2[0].set_title(title_reg[i])
    ax2[0].set_title(abc[i], loc="left")
    ax2[0].set_ylim(-30,4)
    ax2[0].axhline(y=0, color="black", linewidth=0.75, linestyle="dashed")
    ax2[0].set_xlim(0, 32)
    ax2[0].plot(x_shorter, mean_pr_MYD[:np.sort(length)[-2]], label="MYD", color="red")
    ax2[0].fill_between(x_shorter, mean_pr_MYD[:np.sort(length)[-2]]-std_pr_MYD[:np.sort(length)[-2]], mean_pr_MYD[:np.sort(length)[-2]]+std_pr_MYD[:np.sort(length)[-2]], alpha=0.2, color="red")
    for b in range(len(pr_MYD_matrix)):
        ax2[0].plot(x, pr_MYD_matrix[b], color="lightcoral", linestyle=":")
    if (i==0) or (i==3):
        ax2[0].set_ylabel(r"$\Delta$ PR [mm month$^{-1}$]")
    else:
        ax2[0].set_ylabel(" ")
    ax2[0].plot(x2_shorter, mean_pr_ND[:np.sort(length_ND)[-2]], label="ND", color="purple")
    ax2[0].fill_between(x2_shorter, mean_pr_ND[:np.sort(length_ND)[-2]]-std_pr_ND[:np.sort(length_ND)[-2]], mean_pr_ND[:np.sort(length_ND)[-2]]+std_pr_ND[:np.sort(length_ND)[-2]], alpha=0.2, color="purple")
    for b in range(len(pr_ND_matrix)):
        ax2[0].plot(x2, pr_ND_matrix[b], color="purple", linestyle=":", alpha=0.5)
    
    ax2[1].plot(x_shorter, mean_pet_MYD[:np.sort(length)[-2]], label="MYD", color="red")
    ax2[1].set_ylim(-4, 18)
    ax2[1].axhline(y=0, color="black", linewidth=0.75, linestyle="dashed")
    ax2[1].set_xlim(0, 32)
    ax2[1].fill_between(x_shorter, mean_pet_MYD[:np.sort(length)[-2]]-std_pet_MYD[:np.sort(length)[-2]], mean_pet_MYD[:np.sort(length)[-2]]+std_pet_MYD[:np.sort(length)[-2]], alpha=0.2, color="red")
    for b in range(len(pet_MYD_matrix)):
        ax2[1].plot(x, pet_MYD_matrix[b], color="lightcoral", linestyle=":")
    if (i==0) or (i==3):
        ax2[1].set_ylabel(r"$\Delta$ PET [mm month$^{-1}$]")
    else:
        ax2[1].set_ylabel(" ")
    ax2[1].plot(x2_shorter, mean_pet_ND[:np.sort(length_ND)[-2]], label="ND", color="purple")
    ax2[1].fill_between(x2_shorter, mean_pet_ND[:np.sort(length_ND)[-2]]-std_pet_ND[:np.sort(length_ND)[-2]], mean_pet_ND[:np.sort(length_ND)[-2]]+std_pet_ND[:np.sort(length_ND)[-2]], alpha=0.2, color="purple")
    ax2[1].set_xlabel("Months in drought")
    if i==0:
        ax2[0].legend(loc="lower right")
    for b in range(len(pet_ND_matrix)):
        ax2[1].plot(x2, pet_ND_matrix[b], color="purple", alpha=0.5, linestyle=":")

fig.savefig("/home/6196306/Data/Figures/1950-2023/pet_pr_all_droughts&MYDs_dashed_anom_v5.jpg", dpi=1200)
fig.savefig("/home/6196306/Data/Figures/1950-2023/pet_pr_all_droughts&MYDs_dashed_anom_v5.pdf")

#%% Make boxplots with the lengths of MYDs and NDs
len_MYD = [MYD_AUS[2], MYD_SA[2], MYD_ARG[2], MYD_IND[2], MYD_WEU[2], MYD_CAL[2]]
len_ND = [ND_AUS[2], ND_SA[2], ND_ARG[2], ND_IND[2], ND_WEU[2], ND_CAL[2]]
labels_MYD = ["AUS", "SA", "ARG", "IND", "WEU", "CAL"]
labels_ND = [""]*len(labels_MYD)
# Colors for each region
colors = ["#332288", "#ddcc77", "#117733", "#1f77b4", "#aa4499", "#661100"] 

# Plot boxplot
fig, ax = plt.subplots()
ax.axvline(x=11.5, color="darkred", linestyle="dashed", linewidth=0.6)
meanpointprops = dict(marker='x', markeredgecolor='black')
                    
# Plotting MYD data
bplot1 = ax.boxplot(len_MYD, vert=False, labels=labels_MYD, patch_artist=True, medianprops=dict(color='black'), meanprops=meanpointprops, meanline=False, showmeans=True)
for patch, color in zip(bplot1['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(1)
    patch.set_edgecolor('grey')
    
# Add text annotation for MYD_reg[2]
for i, median in enumerate(bplot1['medians']):
    ax.text(median.get_xdata()[0]-1.2, i+1.37, f'n={len(len_MYD[i])}', verticalalignment='center', fontsize=9, color='black')#, fontweight='bold')

# Plotting ND data
bplot2 = ax.boxplot(len_ND, vert=False, labels=labels_ND, patch_artist=True, medianprops=dict(color='black'), meanprops=meanpointprops, meanline=False, showmeans=True)
for patch, color in zip(bplot2['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.2)
    patch.set_edgecolor("grey")
    
# Add text annotation for ND_reg[2]
for i, median in enumerate(bplot2['medians']):
    ax.text(median.get_xdata()[0]-1.2, i+1.37, f'n={len(len_ND[i])}', verticalalignment='center', fontsize=9, color='black')#, fontweight='bold')


# Custom legend
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], marker="|", color='black', markersize=10, linestyle='None'),
                Line2D([0], [0], marker='x', color='black', markersize=6, linestyle='None')]
ax.legend(custom_lines, ['Median', 'mean'], bbox_to_anchor=(1, 0.75))
ax.set_xlabel("Months in drought")
fig.savefig("/home/6196306/Data/Figures/1950-2023/boxplot_duration_droughts&MYDs_v3.jpg", dpi=1200)
fig.savefig("/home/6196306/Data/Figures/1950-2023/boxplot_duration_droughts&MYDs_v3.pdf")

        
#%% Combinde probability of droughts with linear regression of droughts
fig = plt.figure(figsize=(18, 12))

gs = gridspec.GridSpec(4, 4, figure=fig, wspace=0.1, hspace=-0.25)

main_axes = []
small_axes = []

for i in range(2):
    for j in range(3):
        main_ax = fig.add_subplot(gs[i*2, j])
        main_axes.append(main_ax)
        
        # Create small plot
        small_ax = fig.add_subplot(gs[i*2+1, j])#, sharex=main_ax)
        small_axes.append(small_ax)
        
        # Remove space between main and small plot by adjusting their positions
        pos_main = main_ax.get_position()
        pos_small = small_ax.get_position()
        
        # Set new position for the small plot
        small_ax.set_position([pos_main.x0, pos_main.y0 - pos_main.height/4 - 0.01, 
                               pos_main.width, pos_main.height/4])

        # Adjust main plot to have 2/3 of the height
        main_ax.set_position([pos_main.x0, pos_main.y0, 
                              pos_main.width, pos_main.height])
sel = [0, 1, 5]
for i in sel:
    pr_12 = rolling12(pr_month.where(mask_reg[i]==1).mean(dim=("lat", "lon")))
    pet_12 = rolling12(pet_month.where(mask_reg[i]==1).mean(dim=("lat", "lon")))
    
    #Select non-droughts, droughts, and MYDs
    pr_normal = pr_12.where((mask_MYD_reg[i]==False) & (SPEI_reg[i]>-1))
    pr_drought = pr_12.where((mask_MYD_reg[i]==False) & (SPEI_reg[i]<=-1))
    pr_MYD = pr_12.where(mask_MYD_reg[i]==True)
    
    SPEI_normal = SPEI_reg[i].where((mask_MYD_reg[i]==False) & (SPEI_reg[i]>-1))
    SPEI_drought = SPEI_reg[i].where((mask_MYD_reg[i]==False) & (SPEI_reg[i]<=-1))
    SPEI_MYD = SPEI_reg[i].where(mask_MYD_reg[i]==True)
    
    pet_normal = pet_12.where((mask_MYD_reg[i]==False) & (SPEI_reg[i]>-1))
    pet_drought = pet_12.where((mask_MYD_reg[i]==False) & (SPEI_reg[i]<=-1))
    pet_MYD = pet_12.where(mask_MYD_reg[i]==True)
    
    #Calculate linear regressions
    #For precipitation
    pr_coef_all = np.polyfit(pr_12.dropna(dim="time"), SPEI_reg[i].dropna(dim="time"), deg=1)
    pr_all_values = np.linspace(pr_12.min(), pr_12.max(), 100)
    
    pr_coef_normal = np.polyfit(pr_normal.dropna(dim="time"), SPEI_normal.dropna(dim="time"), deg=1)
    pr_normal_values = np.linspace(pr_normal.min(), pr_normal.max(), 100)
    
    pr_coef_drought = np.polyfit(pr_drought.dropna(dim="time"), SPEI_drought.dropna(dim="time"), deg=1)
    pr_drought_values = np.linspace(pr_drought.min(), pr_drought.max(), 100)
    
    pr_coef_MYD = np.polyfit(pr_MYD.dropna(dim="time"), SPEI_MYD.dropna(dim="time"), deg=1)
    pr_MYD_values = np.linspace(pr_MYD.min(), pr_MYD.max(), 100)
    
    #For pet
    pet_coef_all = np.polyfit(pet_12.dropna(dim="time"), SPEI_reg[i].dropna(dim="time"), deg=1)
    pet_all_values = np.linspace(pet_12.min(), pet_12.max(), 100)
    
    pet_coef_normal = np.polyfit(pet_normal.dropna(dim="time"), SPEI_normal.dropna(dim="time"), deg=1)
    pet_normal_values = np.linspace(pet_normal.min(), pet_normal.max(), 100)
    
    pet_coef_drought = np.polyfit(pet_drought.dropna(dim="time"), SPEI_drought.dropna(dim="time"), deg=1)
    pet_drought_values = np.linspace(pet_drought.min(), pet_drought.max(), 100)
    
    pet_coef_MYD = np.polyfit(pet_MYD.dropna(dim="time"), SPEI_MYD.dropna(dim="time"), deg=1)
    pet_MYD_values = np.linspace(pet_MYD.min(), pet_MYD.max(), 100)
    
    #Calculate r² for all
    r2_pr_all = linregress(pr_12, SPEI_reg[i])[2]**2
    r2_pr_MYD = linregress(pr_MYD.dropna(dim="time"), SPEI_MYD.dropna("time"))[2]**2
    r2_pr_ND = linregress(pr_drought.dropna(dim="time"), SPEI_drought.dropna("time"))[2]**2
    r2_PET_all = linregress(pet_12, SPEI_reg[i])[2]**2
    r2_PET_MYD = linregress(pet_MYD.dropna(dim="time"), SPEI_MYD.dropna("time"))[2]**2
    r2_PET_ND = linregress(pet_drought.dropna(dim="time"), SPEI_drought.dropna("time"))[2]**2

    #Calculate R² and RMSE
    slope_all, intercept_all, r_value_all, p_value_all, std_error_all = linregress(pr_12, SPEI_reg[i])
    slope_ND, intercept_ND, r_value_ND, p_value_ND, std_error_ND = linregress(pr_drought.dropna(dim="time"), SPEI_drought.dropna("time"))
    slope_MYD, intercept_MYD, r_value_MYD, p_value_MYD, std_error_MYD = linregress(pr_MYD.dropna(dim="time"), SPEI_MYD.dropna("time"))
    
    rmse_all = np.sqrt(mean_squared_error(SPEI_reg[i], intercept_all+slope_all*pr_12))
    rmse_ND = np.sqrt(mean_squared_error(SPEI_drought.dropna(dim="time"), intercept_ND+slope_ND*pr_drought.dropna(dim="time")))
    rmse_MYD = np.sqrt(mean_squared_error(SPEI_MYD.dropna(dim="time"), intercept_MYD+slope_MYD*pr_MYD.dropna(dim="time")))
    
    #Combine Australia, India and Western Europe
    if afk[i]=="WEU":
        j = 1
        vmin_pr = 1.1
        vmax_pr = 3.6
        vmin_pet = 1.2
        vmax_pet = 2.4
        title = "Western Europe"
    elif afk[i]=="CAL":
        j = 2
        vmin_pr = 0.3
        vmax_pr = 2.8
        vmin_pet = 2.7
        vmax_pet = 3.9
        title = "California"
    elif afk[i]=="AUS":
        j = 0
        vmin_pr = 0.3
        vmax_pr = 2.8
        vmin_pet = 3.4
        vmax_pet = 4.6
        title = "Australia"
    else:
        j=np.nan
    if j==1 or j==2 or j==0:
        # Combine the data points
        combined_pr = np.concatenate((pr_drought, pr_MYD))
        combined_SPEI = np.concatenate((SPEI_drought, SPEI_MYD))
        colors = np.array(['orange'] * len(pr_drought) + ['red'] * len(pr_MYD))
        
        # Create an array of indices and shuffle them
        indices = np.arange(len(combined_pr))
        np.random.shuffle(indices)
        
        # Apply the shuffled indices to the combined arrays
        shuffled_pr = combined_pr[indices]
        shuffled_SPEI = combined_SPEI[indices]
        shuffled_colors = colors[indices]
        
        main_axes[j].set_title(title)
        main_axes[j].scatter(pr_normal, SPEI_normal, label=fr"All: $\alpha$={pr_coef_all[0]:.2f}, r²={r2_pr_all:.2f}")
        main_axes[j].plot(pr_all_values, pr_coef_all[0]*pr_all_values+pr_coef_all[1], color="black")
        
        main_axes[j].scatter(shuffled_pr, shuffled_SPEI, color=shuffled_colors)

        main_axes[j].plot(pr_drought_values, pr_coef_drought[0]*pr_drought_values+pr_coef_drought[1], color="black")
        main_axes[j].plot(pr_MYD_values, pr_coef_MYD[0]*pr_MYD_values+pr_coef_MYD[1], color="black")
        # Create custom legend entries
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', markersize=9, label=fr"All: $\alpha$={pr_coef_all[0]:.2f}, r²={r2_pr_all:.2f}"),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=9, label=fr"ND: $\alpha$={pr_coef_drought[0]:.2f}, r²={r2_pr_ND:.2f}"),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=9, label=fr"MYD: $\alpha$={pr_coef_MYD[0]:.2f}, r²={r2_pr_MYD:.2f}")
        ]

        # Add the legend to the plot
        main_axes[j].legend(handles=handles, loc="upper left", handletextpad=0.1, labelspacing=0.1, borderpad=0.1)
        small_axes[j].set_xlabel(r"PR [mm month$^{-1}$]")
        main_axes[0].set_ylabel("SPEI-12 [-]")
        main_axes[3].set_ylabel("SPEI-12 [-]")
        main_axes[j].set_ylim(-3.3, 3.3)
        main_axes[j+3].set_ylim(-3.3, 3.3)
        main_axes[j].set_xlim(vmin_pr*30.5, vmax_pr*30.5)
        if j!=0:
            main_axes[j].set_yticklabels([])
            main_axes[j+3].set_yticklabels([])
            small_axes[j].set_yticklabels([])
            small_axes[j+3].set_yticklabels([])
        
        main_axes[j+3].scatter(pet_normal, SPEI_normal, label=fr"All: $\alpha$={pet_coef_all[0]:.2f}, r²={r2_PET_all:.2f}")
        main_axes[j+3].plot(pet_all_values, pet_coef_all[0]*pet_all_values+pet_coef_all[1], color="black")
        
        combined_pet = np.concatenate((pet_drought, pet_MYD))
        colors_pet = np.array(['orange'] * len(pet_drought) + ['red'] * len(pet_MYD))
        
        # Create an array of indices and shuffle them
        indices_pet = np.arange(len(combined_pet))
        np.random.shuffle(indices_pet)
        
        # Apply the shuffled indices to the combined arrays
        shuffled_pet = combined_pet[indices_pet]
        shuffled_SPEI_pet = combined_SPEI[indices_pet]
        shuffled_colors_pet = colors[indices_pet]
        
        main_axes[j+3].scatter(shuffled_pet, shuffled_SPEI_pet, color=shuffled_colors_pet)
        
        main_axes[j+3].plot(pet_drought_values, pet_coef_drought[0]*pet_drought_values+pet_coef_drought[1], color="black")
        main_axes[j+3].plot(pet_MYD_values, pet_coef_MYD[0]*pet_MYD_values+pet_coef_MYD[1], color="black")
        small_axes[j+3].set_xlabel(r"PET [mm month$^{-1}$]")
        
        # Create custom legend entries
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', markersize=9, label=fr"All: $\alpha$={pet_coef_all[0]:.2f}, r²={r2_PET_all:.2f}"),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=9, label=fr"ND: $\alpha$={pet_coef_drought[0]:.2f}, r²={r2_PET_ND:.2f}"),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=9, label=fr"MYD: $\alpha$={pet_coef_MYD[0]:.2f}, r²={r2_PET_MYD:.2f}")
        ]

        # Add the legend to the plot
        main_axes[j+3].legend(handles=handles, loc="upper right", handletextpad=0.1, labelspacing=0.1, borderpad=0.1)
        #ax1[1,j].legend(loc="upper right")
        main_axes[j+3].set_xlim(vmin_pet*30.5, vmax_pet*30.5)
        main_axes[j].set_title(abc[j], loc="left")
        #Part where the probabilities are plotted
        # Remove NaN values
        pr_normal = pr_normal[~np.isnan(pr_normal)]
        pr_drought = pr_drought[~np.isnan(pr_drought)]
        pr_MYD = pr_MYD[~np.isnan(pr_MYD)]
        
        pet_normal = pet_normal[~np.isnan(pet_normal)]
        pet_drought = pet_drought[~np.isnan(pet_drought)]
        pet_MYD = pet_MYD[~np.isnan(pet_MYD)]
        
        # Calculate PDFs
        pr_pdf_normal = gaussian_kde(pr_normal)
        pr_pdf_drought = gaussian_kde(pr_drought)
        pr_pdf_MYD = gaussian_kde(pr_MYD)
        
        pet_pdf_normal = gaussian_kde(pet_normal)
        pet_pdf_drought = gaussian_kde(pet_drought)
        pet_pdf_MYD = gaussian_kde(pet_MYD)
        
        # Define a range of PET values for the x-axis
        x_pr = np.linspace(min(pr_normal.min(), pr_drought.min(), pr_MYD.min()), 
                        max(pr_normal.max(), pr_drought.max(), pr_MYD.max()), 1000)
        
        x_pet = np.linspace(min(pet_normal.min(), pet_drought.min(), pet_MYD.min()), 
                        max(pet_normal.max(), pet_drought.max(), pet_MYD.max()), 1000)
        
        # Evaluate PDFs
        pr_pdf_normal_values = pr_pdf_normal(x_pr)
        pr_pdf_drought_values = pr_pdf_drought(x_pr)
        pr_pdf_MYD_values = pr_pdf_MYD(x_pr)
        
        pet_pdf_normal_values = pet_pdf_normal(x_pet)
        pet_pdf_drought_values = pet_pdf_drought(x_pet)
        pet_pdf_MYD_values = pet_pdf_MYD(x_pet)
        
        # Normalize the PDFs so they stack to 1
        pr_total_pdf_values = pr_pdf_normal_values + pr_pdf_drought_values + pr_pdf_MYD_values
        pr_pdf_normal_values /= pr_total_pdf_values
        pr_pdf_drought_values /= pr_total_pdf_values
        pr_pdf_MYD_values /= pr_total_pdf_values
        
        pet_total_pdf_values = pet_pdf_normal_values + pet_pdf_drought_values + pet_pdf_MYD_values
        pet_pdf_normal_values /= pet_total_pdf_values
        pet_pdf_drought_values /= pet_total_pdf_values
        pet_pdf_MYD_values /= pet_total_pdf_values
        
        # Plot the stacked probability plot
        small_axes[j].fill_between(x_pr, 0, pr_pdf_normal_values, label='Not dry', color='tab:blue')
        small_axes[j].fill_between(x_pr, pr_pdf_normal_values, pr_pdf_normal_values + pr_pdf_drought_values, label='ND', color='orange')
        small_axes[j].fill_between(x_pr, pr_pdf_normal_values + pr_pdf_drought_values, pr_pdf_normal_values + pr_pdf_drought_values + pr_pdf_MYD_values, label='MYD', color='red')#, alpha=0.5)
        
        small_axes[j+3].fill_between(x_pet, 0, pet_pdf_normal_values, label='Not dry', color='tab:blue')#, alpha=0.5)
        small_axes[j+3].fill_between(x_pet, pet_pdf_normal_values, pet_pdf_normal_values + pet_pdf_drought_values, label='ND', color='orange')#, alpha=0.5)
        small_axes[j+3].fill_between(x_pet, pet_pdf_normal_values + pet_pdf_drought_values, pet_pdf_normal_values + pet_pdf_drought_values + pet_pdf_MYD_values, label='MYD', color='red')#, alpha=0.5)
        
        #Draw line at P(Not dry)=0 and at P(MYD)=1
        if (pr_pdf_normal_values<=0.01).any()==True:
            small_axes[j].axvline(max(x_pr[pr_pdf_normal_values<=0.005]), color="black", linestyle="dashed")
        if (pr_pdf_MYD_values>=0.99).any()==True:
            small_axes[j].axvline(max(x_pr[pr_pdf_MYD_values>=0.995]), color="black")
            
        if (pet_pdf_normal_values<=0.01).any()==True:
            small_axes[j+3].axvline(min(x_pet[pet_pdf_normal_values<=0.005]), color="black", linestyle="dashed")
        if (pet_pdf_MYD_values>=0.99).any()==True:
            small_axes[j+3].axvline(min(x_pet[pet_pdf_MYD_values>=0.995]), color="black")
            
        small_axes[j].set_ylim(0, 1)
        small_axes[j+3].set_ylim(0, 1)
        small_axes[j].set_xlim(vmin_pr*30.5, vmax_pr*30.5)
        small_axes[j+3].set_xlim(vmin_pet*30.5, vmax_pet*30.5)
        small_axes[0].set_ylabel("Probability")
        small_axes[3].set_ylabel("Probability")
        
        main_axes[j].set_xticklabels([])
        main_axes[j+3].set_xticklabels([])
        main_axes[j].tick_params(left=True, bottom=True)
        main_axes[j+3].tick_params(left=True, bottom=True)

fig.savefig("/home/6196306/Data/Figures/1950-2023/lin_regress+prob_AUS_WEU_CAL_v2.jpg", dpi=1200, bbox_inches="tight")
fig.savefig("/home/6196306/Data/Figures/1950-2023/lin_regress+prob_AUS_WEU_CAL_v2.pdf", bbox_inches="tight")
