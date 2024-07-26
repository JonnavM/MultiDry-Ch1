#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:20:27 2024

@author: 6196306
Climatological plots for precipitation and PET
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import sys
sys.path.append("/home/6196306/Data/Python_scripts/")
from functions import MYD, ND, mask_MYD, mask_ND
import calendar
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors

#Load in SPEI
SPEI_AUS = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_AUS.nc")#.sel(time=slice("1956", "2022"))
SPEI_WEU = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_WEU.nc")#.sel(time=slice("1956", "2022"))
SPEI_CAL = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_CAL.nc")#.sel(time=slice("1956", "2022"))
SPEI_IND = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_IND.nc")#.sel(time=slice("1956", "2022"))
SPEI_SA = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_SA.nc")#.sel(time=slice("1956", "2022"))
SPEI_ARG = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_ARG.nc")#.sel(time=slice("1956", "2022"))

#Load in masks
mask_AUS = xr.open_dataarray("/scratch/6196306/masks/mask_AUS.nc")
mask_WEU = xr.open_dataarray("/scratch/6196306/masks/mask_WEU.nc")
mask_IND = xr.open_dataarray("/scratch/6196306/masks/mask_IND.nc")
mask_SA = xr.open_dataarray("/scratch/6196306/masks/mask_SA.nc") 
mask_CAL = xr.open_dataarray("/scratch/6196306/masks/mask_CAL.nc")
mask_ARG = xr.open_dataarray("/scratch/6196306/masks/mask_ARG.nc")
mask_PAK = xr.open_dataarray("/scratch/6196306/masks/mask_PAK.nc")

#Load in PET and pr
pr = xr.open_dataarray("/scratch/6196306/ERA5/total_precipitation/era5_total_precipitation_1950-2023_monthly_0_5.nc")*1000 #.sel(time=slice("1950", "2022"), expver=1)*1000 #Units: mm/day
pet = xr.open_dataset("/scratch/6196306/PET/PenmanMonteith/pm_fao56_1950-2023_monthly_0_5_v3.nc").PM_FAO_56 #Units: mm/day
pet["time"]=pr["time"]
#Change units to mm/month
days_in_month = pr.time.dt.days_in_month
pr_month = pr*days_in_month
pet_month = pet*days_in_month
pr_month.attrs['units'] = 'mm/month'
pet_month.attrs['units'] = 'mm/month'

#Calculate MYDs and NDs
#Australia
MYD_AUS = MYD(SPEI_AUS, "AUS")
mask_MYD_AUS = mask_MYD(SPEI_AUS, "AUS")

#South Africa
MYD_SA = MYD(SPEI_SA, "SA")
ND_SA = ND(SPEI_SA, "SA")
mask_MYD_SA = mask_MYD(SPEI_SA, "SA")
mask_ND_SA = mask_ND(SPEI_SA, "SA")

#California
MYD_CAL = MYD(SPEI_CAL, "CAL")
mask_MYD_CAL = mask_MYD(SPEI_CAL, "CAL")
      
#Western Europe
MYD_WEU = MYD(SPEI_WEU, "WEU")
mask_MYD_WEU = mask_MYD(SPEI_WEU, "WEU")

#Middle Argentina
MYD_ARG = MYD(SPEI_ARG, "ARG")
mask_MYD_ARG = mask_MYD(SPEI_ARG, "ARG")

#India
MYD_IND = MYD(SPEI_IND, "IND")
mask_MYD_IND = mask_MYD(SPEI_IND, "IND")

#Cut-outs per region
lat_WEU = slice(42, 58) 
lon_WEU = slice(-5, 17)
lat_IND = slice(16, 34)
lon_IND = slice(68, 94)
lat_AUS = slice(-40, -20)
lon_AUS = slice(135, 155)
lat_SA = slice(-33, -21)
lon_SA = slice(15, 31)
lat_ARG = slice(-45, -25)
lon_ARG = slice(-75, -55)
lat_CAL = slice(30, 44)
lon_CAL = slice(-126, -111)

lat_reg = [lat_CAL, lat_WEU, lat_IND, lat_ARG, lat_SA, lat_AUS]
lon_reg = [lon_CAL, lon_WEU, lon_IND, lon_ARG, lon_SA, lon_AUS]
mask_reg = [mask_CAL, mask_WEU, mask_IND, mask_ARG, mask_SA, mask_AUS]
mask_MYD_reg = [mask_MYD_CAL, mask_MYD_WEU, mask_MYD_IND, mask_MYD_ARG, mask_MYD_SA, mask_MYD_AUS]
reg = ["CAL", "WEU", "IND", "ARG", "SA", "AUS"]
month_names = [calendar.month_abbr[m] for m in range(1,13)]

#%% Definitions 
SPEI_grid = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree.nc")
# Land cover, needed to mask ice, snow, and scarce vegetation
lc = xr.open_dataset("/scratch/ruiij001/Data/Landcover/C3S-LC-L4-LCCS-Map-P1Y-2019-v2.1.1_regrid_0_1_degrees.nc").lccs_class

def MYD_gridcell(spei_grid):
    # Create a mask where SPEI_AUS_grid is less than or equal to -1
    mask = spei_grid <= -1
    # Apply a rolling window of size 12 along the time dimension and check where the sum is 12
    rolling_sum = mask.rolling(time=12).sum()
    # Create a new mask where rolling_sum is greater than or equal to 12
    final_mask = rolling_sum >= 12
    # Convert the DataArray to a numpy array
    data_array = final_mask
    data_array_np = data_array.values
    # Iterate over each latitude and longitude
    for lat in range(data_array_np.shape[1]):
        for lon in range(data_array_np.shape[2]):
            # Iterate over each time step
            for t in range(data_array_np.shape[0]):
                # Check if the current value is True
                if data_array_np[t, lat, lon]:
                    # Set the 12 values before the True value to True
                    data_array_np[max(0, t-11):t, lat, lon] = True
    # Convert the modified numpy array back to a DataArray
    modified_data_array = xr.DataArray(data_array_np, coords=data_array.coords, dims=data_array.dims)
    # Print the modified DataArray
    print(modified_data_array.sel(lon = 145, lat = -36)[-100:])
    # Convert the DataArray to a numpy array
    data_array_np = modified_data_array.values
    # Create a new array with 1 for True values and nan for False values
    new_mask_array = np.where(data_array_np, 1, np.nan)
    # Convert the new array back to a DataArray
    new_mask_data_array = xr.DataArray(new_mask_array, coords=data_array.coords, dims=data_array.dims)
    return new_mask_data_array

SPEI_MYD_grid = MYD_gridcell(SPEI_grid).where(lc != 220).where(lc != 210).where(lc != 200).where(lc != 150) #150=sparce vegetation
SPEI_ND_grid = SPEI_grid.where((SPEI_MYD_grid!=1)&(SPEI_grid<=-1), np.nan).where(lc != 220).where(lc != 210).where(lc != 200).where(lc != 150)
SPEI_ND_grid = SPEI_ND_grid.where(SPEI_ND_grid.isnull(), 1)

def number_of_droughts(SPEI_MYD_grid):
    da = SPEI_MYD_grid.where(SPEI_MYD_grid.notnull(), 0)
    drought_starts = da.diff("time")==1
    total_droughts = np.sum(drought_starts, axis=0)
    # Create a new xarray DataArray with the total number of droughts
    total_droughts_da = xr.DataArray(total_droughts, coords={'lat': da['lat'], 'lon': da['lon']}, dims=('lat', 'lon'))
    return total_droughts_da

n_droughts = number_of_droughts(SPEI_MYD_grid)
n_droughts = n_droughts.where(n_droughts != 0, float('nan'))


#%% Plot map of the world
#Make own colormap
colors = [(0.13, 0.8, 0), (0.40, 0.9, 0.3), (0.67, 1, 0.6), (1, 0.65, 0.3), (1, 0.5, 0.3), (1, 0.3, 0.77), (0.8, 0.15, 0.69), (0.6, 0, 0.6)] #This one is used for v2
# Create a colormap object
cmap_name = 'Custom_Red'
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=len(colors))
cmap2 = plt.get_cmap("PRGn_r", 6) 

cmap_original = cmap
colors_original = [cmap_original(i) for i in np.linspace(0, 1, len(colors))]

# Define custom colors for specific ranges of values
custom_colors = [
    colors_original[0],
    colors_original[1],  # Color for values 1 and 2
    colors_original[2],  # Color for value 3
    colors_original[3],
    colors_original[4],# Color for values 4 and 5
    colors_original[5],  # Color for value 6
    colors_original[6],
    colors_original[7]# Color for all values higher
]

# Create a new colormap with the custom colors
custom_cmap = ListedColormap(custom_colors)

#Plot
fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=11)), figsize=(12,12))
n_droughts.plot(transform=ccrs.PlateCarree(), cmap=custom_cmap, vmin=0.5, vmax=8.5, cbar_kwargs=dict(orientation="horizontal", pad=0.05, aspect=40, location="bottom", label="Number of MYDs between 1950-2023", ticks=[1,2,3,4,5,6,7,8]))
ax.coastlines()
ax.set_extent([-180, 180, -63, 90], crs=ccrs.PlateCarree()) #lonW, lonE, latS, latN
ax.contour(mask_ARG.lon, mask_ARG.lat, np.isnan(mask_ARG), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.contour(mask_AUS.lon, mask_AUS.lat, np.isnan(mask_AUS), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.contour(mask_WEU.lon, mask_WEU.lat, np.isnan(mask_WEU), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.contour(mask_CAL.lon, mask_CAL.lat, np.isnan(mask_CAL), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.contour(mask_SA.lon, mask_SA.lat, np.isnan(mask_SA), colors='black', linewidths=2, transform=ccrs.PlateCarree())
ax.contour(mask_IND.lon, mask_IND.lat, np.isnan(mask_IND), colors='black', linewidths=2, transform=ccrs.PlateCarree())

ax.axis("off")

lat_reg = [lat_CAL, lat_WEU, lat_IND, lat_ARG, lat_SA, lat_AUS]
lon_reg = [lon_CAL, lon_WEU, lon_IND, lon_ARG, lon_SA, lon_AUS]
mask_reg = [mask_CAL, mask_WEU, mask_IND, mask_ARG, mask_SA, mask_AUS]
mask_MYD_reg = [mask_MYD_CAL, mask_MYD_WEU, mask_MYD_IND, mask_MYD_ARG, mask_MYD_SA, mask_MYD_AUS]
reg = ["CAL", "WEU", "IND", "ARG", "SA", "AUS"]
cb_color = ["#661100", "#aa4499", "#1f77b4", "#117733", "#ddcc77", "#332288"]
month_letter = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

for i, region_name in enumerate(reg):
    if region_name == "IND": #IND, AUS, WEU, SA, SSA, CAL
        reg_lat = slice(22, 32)
        reg_lon = slice(72, 90)
        left = 0.82 #Fraction of figure
        bottom = 0.43 #Fraction of figure
        inset_center_x = 132 #End of arrow
        inset_center_y = 25 #Start of arrow
        box_center_lon = reg_lon.stop - 13
        box_center_lat = (reg_lat.start + reg_lat.stop) / 2
    elif region_name == "AUS":
        reg_lat = slice(-40, -22)
        reg_lon = slice(135, 155)
        left = 0.63
        bottom = 0.28
        inset_center_x = 97
        inset_center_y = -40
        box_center_lon = reg_lon.start - 6
        box_center_lat = (reg_lat.start + reg_lat.stop) / 2 - 2
    elif region_name == "WEU":
        reg_lat = slice(45, 55)
        reg_lon = slice(-1, 13)
        left = 0.37
        bottom = 0.445
        inset_center_x = -30
        inset_center_y = 47
        box_center_lon = reg_lon.start - 9
        box_center_lat = (reg_lat.start + reg_lat.stop) / 2
    elif region_name == "SA":
        reg_lat = slice(-33, -21)
        reg_lon = slice(15, 31)
        left = 0.42
        bottom = 0.28
        inset_center_x = -5
        inset_center_y = -28
        box_center_lon = reg_lon.start - 9
        box_center_lat = (reg_lat.start + reg_lat.stop) / 2
    elif region_name == "ARG":
        reg_lat = slice(-43, -26)
        reg_lon = slice(-80, -55)
        left = 0.20
        bottom = 0.28
        inset_center_x = -105
        inset_center_y = -35
        box_center_lon = reg_lon.start - 1
        box_center_lat = (reg_lat.start + reg_lat.stop) / 2
    elif region_name == "CAL":
        reg_lat = slice(32, 41.5)
        reg_lon = slice(-124, -115)
        left = 0.11
        bottom = 0.43
        inset_center_x = -149
        inset_center_y = 36
        box_center_lon = reg_lon.start - 9
        box_center_lat = (reg_lat.start + reg_lat.stop) / 2
    fig.patch.set_facecolor('white')
    left, bottom, width, height = [left, bottom, 0.08, 0.08]
    ax_inset = fig.add_axes([left, bottom, width, height])
    ax_inset.set_facecolor('white')
    
    pr_reg = pr_month.sel(lat=lat_reg[i], lon=lon_reg[i]).where(mask_reg[i]==1).mean(dim=("lat", "lon")).groupby("time.month").mean("time")
    pr_reg_std = pr_month.sel(lat=lat_reg[i], lon=lon_reg[i]).where(mask_reg[i]==1).mean(dim=("lat", "lon")).groupby("time.month").std("time")
    pet_reg = pet_month.sel(lat=lat_reg[i], lon=lon_reg[i]).where(mask_reg[i]==1).mean(dim=("lat", "lon")).groupby("time.month").mean("time")
    pet_reg_std = pet_month.sel(lat=lat_reg[i], lon=lon_reg[i]).where(mask_reg[i]==1).mean(dim=("lat", "lon")).groupby("time.month").std("time")

    ax_inset.bar(pr_reg.month, pr_reg, color="tab:blue", yerr=pr_reg_std, label="Pr")
    ax_inset.set_xticks(pr_reg.month, month_letter, fontsize=8)

    ax_inset.set_ylim(0,14*30.5)
    ax_inset.tick_params(axis='y', labelsize=8)
    pet_reg.plot(ax=ax_inset, color="red", label="PET")
    ax_inset.fill_between(x=pet_reg.month, y1=pet_reg-pet_reg_std, y2=pet_reg+pet_reg_std, color="red", alpha=0.2)
    ax_inset.set_title(reg[i], color=cb_color[i], fontweight="bold")
    
    # Calculate total annual precipitation and PET
    total_pr_ann = pr_reg.sum().item()
    total_pet_ann = pet_reg.sum().item()

    # Add text with total annual values in the upper right corner
    if region_name == "ARG":
        ax_inset.set_ylabel("[mm/month]", fontsize=8)
        ax_inset.text(12, 12+350, f"PET:{total_pet_ann:.0f} mm/yr", color="red", fontsize=8, horizontalalignment="right")
        ax_inset.text(12, 10.5+300, f"PR:{total_pr_ann:.0f} mm/yr", color="tab:blue", fontsize=8, horizontalalignment="right")
    else:
        ax_inset.set_ylabel(" ")
        ax_inset.text(12, 10.5+300, f'{total_pr_ann:.0f}', color='tab:blue', fontsize=8, ha='right')
        ax_inset.text(12, 12+350, f'{total_pet_ann:.0f}', color='red', fontsize=8, ha='right')

    ax_inset.set_xlabel(" ")
    # Draw an arrow pointing towards the inset plot
    ax.annotate("", xy=(inset_center_x, inset_center_y), xytext=(box_center_lon, box_center_lat),
                    arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=2), fontsize=12)
    
fig.savefig("/home/6196306/Data/Figures/1950-2023/world_map_MYDS+climate_1950-2023_v6.pdf", bbox_inches="tight")
fig.savefig("/home/6196306/Data/Figures/1950-2023/world_map_MYDS+climate_1950-2023_v6.jpg", bbox_inches="tight", dpi=1200)

#%% Add statistics on duration, intensity and fraction of MYDs
# Define discrete intervals for colorbars
duration_bounds = [0, 20, 40, 60, 80, 100, 120, 140]
intensity_bounds = [-2.4, -2.2, -2.0, -1.8, -1.6, -1.4, -1.2, -1.0]
ratio_bounds = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

# Create discrete colormap norms
duration_norm = mcolors.BoundaryNorm(duration_bounds, ncolors=256, clip=True)
intensity_norm = mcolors.BoundaryNorm(intensity_bounds, ncolors=256, clip=True)
ratio_norm = mcolors.BoundaryNorm(ratio_bounds, ncolors=256, clip=True)

fig, ax = plt.subplots(2,2, subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=11)), figsize=(12,12), layout="tight")
#Number of MYDs
n_droughts.plot(ax=ax[0,0], transform=ccrs.PlateCarree(), cmap=custom_cmap, vmin=0.5, vmax=8.5, cbar_kwargs=dict(orientation="horizontal", pad=0.05, label="Number of MYDs", ticks=[1,2,3,4,5,6,7,8]))
#Duration
duration = SPEI_MYD_grid.count(dim="time")
duration.where(duration!=0, np.nan).plot(ax=ax[0,1], transform=ccrs.PlateCarree(), vmax=140, cmap="Blues", norm=duration_norm, cbar_kwargs=dict(orientation="horizontal", pad=0.05, label="Months in MYDs"))
#Intensity
SPEI_grid.where(SPEI_MYD_grid==1).mean("time").plot(ax=ax[1,0], transform=ccrs.PlateCarree(), vmin=-2.4, vmax=-1, cmap="Reds_r", norm=intensity_norm, cbar_kwargs=dict(orientation="horizontal", pad=0.05, label="MYD intensity"))
#Ratio of months
(SPEI_MYD_grid.count(dim="time")/SPEI_ND_grid.count(dim="time")).plot(ax=ax[1,1], transform=ccrs.PlateCarree(), cmap="PRGn", norm=ratio_norm, vmax=2, cbar_kwargs=dict(orientation="horizontal", pad=0.05, label="Ratio months in MYDs/NDs"))
for ax in ax.flatten():
    ax.coastlines()
    ax.set_extent([-180, 180, -63, 90], crs=ccrs.PlateCarree()) #lonW, lonE, latS, latN
    ax.contour(mask_ARG.lon, mask_ARG.lat, np.isnan(mask_ARG), colors='black', linewidths=2, transform=ccrs.PlateCarree())
    ax.contour(mask_AUS.lon, mask_AUS.lat, np.isnan(mask_AUS), colors='black', linewidths=2, transform=ccrs.PlateCarree())
    ax.contour(mask_WEU.lon, mask_WEU.lat, np.isnan(mask_WEU), colors='black', linewidths=2, transform=ccrs.PlateCarree())
    ax.contour(mask_CAL.lon, mask_CAL.lat, np.isnan(mask_CAL), colors='black', linewidths=2, transform=ccrs.PlateCarree())
    ax.contour(mask_SA.lon, mask_SA.lat, np.isnan(mask_SA), colors='black', linewidths=2, transform=ccrs.PlateCarree())
    ax.contour(mask_IND.lon, mask_IND.lat, np.isnan(mask_IND), colors='black', linewidths=2, transform=ccrs.PlateCarree())

    ax.axis("off")
fig.savefig("/home/6196306/Data/Figures/1950-2023/world_map_allstats_MYDs_v2.pdf", bbox_inches="tight")
fig.savefig("/home/6196306/Data/Figures/1950-2023/world_map_allstats_MYDs_v2.jpg", bbox_inches="tight", dpi=1200)   