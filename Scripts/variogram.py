#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:19:36 2023

@author: 6196306
Variogram 
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from skgstat import Variogram
import cartopy.crs as ccrs
from scipy.stats import pearsonr
from scipy.ndimage.measurements import center_of_mass, label
from matplotlib.colors import ListedColormap, BoundaryNorm

# Get the original "Reds" colormap
cmap = plt.get_cmap('Reds')
# Sample colors from the "Reds" colormap at specified intervals
reds_colors = [cmap(0.2), cmap(0.4), cmap(0.6), cmap(0.8), cmap(1.0)]   # Dark red for the highest values
# Insert white color for the 0-0.2 range
colors = [(1, 1, 1, 1), reds_colors[0], reds_colors[1], reds_colors[2], reds_colors[3]]
# Create a ListedColormap with these colors
Reds_w = ListedColormap(colors, name='WhiteRedsQuantized')
# Define boundaries for the intervals
bounds = [0, 0.2, 0.4, 0.6, 0.8, 1]
# Create a BoundaryNorm to map values to the colormap intervals
norm = BoundaryNorm(bounds, Reds_w.N)

# Import all datasets
dir = "/scratch/6196306/"
z500 = xr.open_dataset("/home/6196306/Data/ERA5/Z500/era5_geopotential_2020_daily_0_5.nc").z/9.81
z500_monthly = (xr.open_dataset(dir+"/ERA5/Z500/regrid/era5_geopotential_1950-2023_monthly_0_5.nc").z/9.81).sel(expver=1).groupby("time.month").mean("time")
rb5 = xr.open_dataset(dir+"majorCatchments_0_5.nc").Band1
rb_or = xr.open_dataset(dir+"majorCatchments.nc").Band1
rb = rb_or
rb["lat"]=rb5["lat"]
rb["lon"]=rb5["lon"]
res = 0.5

z500_anom = xr.full_like(z500, fill_value=np.nan)

for i in range(12):
    print(i)
    z500_anom_i = z500.where(z500["time.month"]==i+1)-z500_monthly.sel(month=i+1).values

    z500_anom.loc[dict(time=z500["time.month"] == i+1)] = z500_anom_i.loc[dict(time=z500["time.month"] == i+1)]

# All latitudes and longitudes and masks
lat_AUS = slice(-50, -10)
lon_AUS = slice(120, 165)

lat_WEU = slice(20, 70)
lon_WEU = slice(-30,30)

lat_SA = slice(-45,-12)
lon_SA = slice(0, 50)

lat_CAL = slice(15, 62)
lon_CAL = slice(-140, -100)

lat_ARG = slice(-60, -16)
lon_ARG = slice(-88, -38)

lat_IND=slice(10,47)
lon_IND=slice(60,110)

mask_AUS = xr.open_dataarray("/scratch/6196306/masks/mask_AUS.nc")
mask_WEU = xr.open_dataarray("/scratch/6196306/masks/mask_WEU.nc")
mask_IND = xr.open_dataarray("/scratch/6196306/masks/mask_IND.nc")
mask_SA = xr.open_dataarray("/scratch/6196306/masks/mask_SA.nc") 
mask_ARG = xr.open_dataarray("/scratch/6196306/masks/mask_ARG.nc")
mask_CAL = xr.open_dataarray("/scratch/6196306/masks/mask_CAL.nc")

#Put in lists
mask_reg = [mask_CAL, mask_WEU, mask_IND, mask_ARG, mask_SA, mask_AUS]
region = ["California", "Western Europe", "India", "Argentina", "South Africa", "Australia"]
lat_reg = [lat_CAL, lat_WEU, lat_IND, lat_ARG, lat_SA, lat_AUS]
lon_reg = [lon_CAL, lon_WEU, lon_IND, lon_ARG, lon_SA, lon_AUS]
#%% Make one version with all figures in one plot
mean_r_2020 = []
std_r_2020 = []

for j in range(len(mask_reg)):
    var_ts = z500_anom.sel(lat=lat_reg[j], lon=lon_reg[j])
    time_2020 = var_ts.time
    lon, lat = np.meshgrid(var_ts.lon, var_ts.lat)
    lon_flat, lat_flat = lon.flatten(), lat.flatten()
    r_2020 = []

    for i, t in enumerate(time_2020[::5]):
        z500_day = var_ts.sel(time=t)
        # Flatten the lat and lon coordinates
        values_day = z500_day.values.flatten()
        variogram_day = Variogram(np.column_stack([lon_flat, lat_flat]), values_day, model="spherical", fit_method="lm")
        r_day = variogram_day.describe().get("effective_range")
        if r_day > np.sqrt(len(lon)**2+len(lat)**2):
            r_day = np.nan
        r_2020.append(r_day)

    mean_r_2020.append(np.nanmean(r_2020))
    std_r_2020.append(np.nanstd(r_2020))

#Plot
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=11)}, figsize=(12, 6))
for j in range(len(mask_reg)):
    
    # For the map
    z500_corr = z500_anom.sel(lat=lat_reg[j], lon=lon_reg[j])

    # Select the central point
    masked_mask = np.ma.masked_invalid(mask_reg[j])
    labeled_mask, num_features = label(~masked_mask.mask)
    center_coords = center_of_mass(~masked_mask.mask, labeled_mask, index=1)  # Use index=1 for the first connected region
    central_lat_idx, central_lon_idx = center_coords
    central_lat = mask_reg[j].lat[int(central_lat_idx)].values
    central_lon = mask_reg[j].lon[int(central_lon_idx)].values

    # Extract the time series for the central point
    central_values = z500_corr.sel(lat=central_lat, lon=central_lon, method="nearest")

    # Flatten the lat and lon coordinates
    lon, lat = np.meshgrid(z500_corr.lon, z500_corr.lat)
    lon_flat, lat_flat = lon.flatten(), lat.flatten()

    #Draw a circle around the central point with radius effecitive range
    theta = np.linspace(0, 2*np.pi, 100)
    circle_lat = central_lat + mean_r_2020[j]*res * np.sin(theta)
    circle_lon = central_lon + mean_r_2020[j]*res * np.cos(theta)
    circle_lat_min = central_lat + (mean_r_2020[j]-std_r_2020[j])*res *np.sin(theta)
    circle_lat_max = central_lat + (mean_r_2020[j]+std_r_2020[j])*res *np.sin(theta)
    circle_lon_min = central_lon + (mean_r_2020[j]-std_r_2020[j])*res *np.cos(theta)
    circle_lon_max = central_lon + (mean_r_2020[j]+std_r_2020[j])*res *np.cos(theta)

    # List to store correlation coefficients for each grid cell
    correlations = []

    # Iterate over each grid cell
    for i in range(len(lon_flat)):
        # Extract the time series for the current grid cell
        cell_values = z500_corr.isel(lat=i // len(z500_corr.lon), lon=i % len(z500_corr.lon)).values
        # Calculate the correlation coefficient
        correlation_coefficient, _ = pearsonr(central_values, cell_values)
        # Append to the list
        correlations.append(correlation_coefficient)
    # Reshape the correlations array to match the grid
    correlations = np.array(correlations).reshape(lon.shape)

    # Plot the correlation coefficients using Cartopy
    c = ax.pcolormesh(lon, lat, correlations, cmap=Reds_w, norm=norm, shading='auto', transform=ccrs.PlateCarree())
    # Add contour lines
    contour_levels = [0.2, 0.4, 0.6, 0.8]

    if j==0:
        ax.plot(circle_lon, circle_lat, color='blue', transform=ccrs.PlateCarree(), label=r"$\mu$")
        ax.plot(circle_lon_min, circle_lat_min, color='blue', transform=ccrs.PlateCarree(), label=r"$\mu-\sigma$", alpha=0.5)
    else:
        ax.plot(circle_lon, circle_lat, color='blue', transform=ccrs.PlateCarree())
        ax.plot(circle_lon_min, circle_lat_min, color='blue', transform=ccrs.PlateCarree(), alpha=0.5)
    ax.contour(mask_reg[j].lon, mask_reg[j].lat, np.isnan(mask_reg[j]), colors='black', linewidths=1.2, transform=ccrs.PlateCarree())
ax.coastlines(linewidth=0.5)
ax.set_extent([-180, 180, -63, 90], crs=ccrs.PlateCarree()) #lonW, lonE, latS, latN
ax.axis("off")
   
fig.legend(bbox_to_anchor=(0.21, 0.35))
cbar = plt.colorbar(c, ax=ax, orientation='vertical', label='Correlation Coefficient', fraction=0.022)
fig.savefig("/home/6196306/Data/Figures/1950-2023/corr_var_all_v2.jpg", bbox_inches="tight", dpi=1200)
fig.savefig("/home/6196306/Data/Figures/1950-2023/corr_var_all_v2.pdf", bbox_inches="tight")