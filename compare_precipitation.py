#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 09:32:18 2024

@author: 6196306
Test precipitation
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

#Masks
mask_AUS = xr.open_dataarray("/scratch/6196306/masks/mask_AUS.nc")
mask_WEU = xr.open_dataarray("/scratch/6196306/masks/mask_WEU.nc")
mask_IND = xr.open_dataarray("/scratch/6196306/masks/mask_IND.nc")
mask_SA = xr.open_dataarray("/scratch/6196306/masks/mask_SA.nc") 
mask_SSA = xr.open_dataarray("/scratch/6196306/masks/mask_SSA.nc")
mask_CAL = xr.open_dataarray("/scratch/6196306/masks/mask_CAL.nc")
mask_ARG = xr.open_dataarray("/scratch/6196306/masks/mask_ARG.nc")

#Combine in lists
afk = ["CAL", "WEU", "IND", "ARG", "SA", "AUS"]
mask = [mask_CAL, mask_WEU, mask_IND, mask_ARG, mask_SA, mask_AUS]
region = ["California", "Western Europe", "India", "Argentina", "South Africa", "Australia"]

# Load in different precipitation datasets and transform all to mm/day
p_era = xr.open_dataarray("/scratch/6196306/ERA5/total_precipitation/era5_total_precipitation_1950-2023_monthly_0_5.nc")*1000
p_gpcp = xr.open_dataset("/scratch/6196306/GPCP/gpcp_v02r03_monthly_d1979-2023_2_5.nc").precip
p_cru = (xr.open_dataset("/scratch/ruiij001/Data/CRU data/cru_ts4.03.1951.2018.pre.dat_0_5.nc").pre)*0.03285
p_cru = p_cru.resample(time="1MS").mean("time")
p_chirps = xr.open_dataset("/scratch/6196306/CHIRPS/chirps-v2.0.monthly_0_5.nc").precip*0.03285
p_obs = xr.open_dataset("/scratch/6196306/E-OBS/rr_ens_mean_0.25deg_reg_v29.0e_0_5.nc").rr
p_obs = p_obs.resample(time="1MS").mean("time")
p_imerg = xr.open_mfdataset("/scratch/6196306/IMERG/IMERG_2*.HDF5.nc4").precipitation.transpose('time', 'lat', 'lon')*24
p_imerg = p_imerg.resample(time="1MS").mean("time")

#Make new colormap
cmap_rdylygn = plt.colormaps['RdYlGn']
cmap_piyg = plt.colormaps['PiYG']

# Define a function to create the new colormap
def combine_colormaps(cmap1, cmap2, mid_point=0.3):
    new_colors = []
    n_colors = 256
    for i in range(n_colors):
        position = i / (n_colors - 1)
        if position <= mid_point:
            color = cmap2(position)
        else:
            color = cmap1(position)
        new_colors.append(color)
    new_cmap = LinearSegmentedColormap.from_list('CustomCmap', new_colors, N=n_colors)
    return new_cmap

# Combine the colormaps
custom_cmap = combine_colormaps(cmap_rdylygn, cmap_piyg, mid_point=0.5)

#%% Compare mean precipitation per region
fig, axes = plt.subplots(2,3, sharex=True, sharey=True, layout="tight", figsize=(12,8))
for i, ax in enumerate(axes.flatten()):
    p_era.where(mask[i]==1).mean(dim=("lat", "lon")).rolling(time=12).mean("time").plot(ax=ax, label="ERA5", color="black", zorder=10, linewidth=0.8)
    p_gpcp.where(mask[i]==1).mean(dim=("lat", "lon")).rolling(time=12).mean("time").plot(ax=ax, label="GPCP", color="tab:blue")
    p_cru.where(mask[i]==1).mean(dim=("lat", "lon")).rolling(time=12).mean("time").plot(ax=ax, label="CRU", color="tab:orange")
    p_imerg.where(mask[i]==1).mean(dim=("lat", "lon")).rolling(time=12).mean("time").plot(ax=ax, label="IMERG", color="tab:purple")
    if i!=1:
        p_chirps.where(mask[i]==1).mean(dim=("lat", "lon")).rolling(time=12).mean("time").plot(ax=ax, label="CHIRPS", color="tab:olive")
    else:
        p_obs.where(mask[i]==1).mean(dim=("lat", "lon")).rolling(time=12).mean("time").plot(ax=ax, label="E-OBS", color="tab:green")
        ax.legend()
    ax.set_title(region[i])        
    ax.set_ylabel(" ")
    ax.set_xlabel(" ")
axes[0,0].legend()
fig.supylabel("Precipitation [mm/day]")
fig.savefig("/home/6196306/Data/Figures/1950-2023/dataset_precipitation_comparison_v2.pdf", bbox_inches="tight")
fig.savefig("/home/6196306/Data/Figures/1950-2023/dataset_precipitation_comparison_v2.jpg", bbox_inches="tight", dpi=1200)

#%% Calculate MBE and PCC

# Define a function to calculate MBE and PCC
def calculate_metrics(ref, dataset, masker):
    ref_mean = ref.where(masker == 1).mean(dim=("lat", "lon"), skipna=True).rolling(time=12, min_periods=1).mean("time")
    dataset_mean = dataset.where(masker == 1).mean(dim=("lat", "lon"), skipna=True).rolling(time=12, min_periods=1).mean("time").compute()
    # Align time coordinates
    ref_mean, dataset_mean = xr.align(ref_mean, dataset_mean, join='inner')
    # Compute the valid mask to handle NaNs
    valid_mask = np.isfinite(ref_mean) & np.isfinite(dataset_mean)
    # Convert to numpy arrays and drop NaNs
    ref_mean = ref_mean.where(valid_mask, drop=True)
    dataset_mean = dataset_mean.where(valid_mask, drop=True)
    if len(ref_mean) == 0 or len(dataset_mean) == 0:
        return np.nan, np.nan
    mbe = (dataset_mean - ref_mean).mean().item()
    pcc = xr.corr(ref_mean, dataset_mean, dim="time").item()
    return mbe, pcc

# Datasets dictionary with masks and regions
datasets = {
    "GPCP": p_gpcp,
    "CRU": p_cru,
    "IMERG": p_imerg,
    "CHIRPS": p_chirps,
    "E-OBS": p_obs}

metrics = {}
for i, region_name in enumerate(region):
    metrics[region_name] = {}
    for name, dataset in datasets.items():
        # Use E-OBS for region 1 and CHIRPS otherwise
        if i == 1 and name == "CHIRPS":
            continue
        if i != 1 and name == "E-OBS":
            continue
        mbe, pcc = calculate_metrics(p_era, dataset, mask[i])
        metrics[region_name][name] = {"MBE": mbe, "PCC": pcc}

# Plot the metrics
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

mbe_values = np.full((len(region), len(datasets)), np.nan)
pcc_values = np.full((len(region), len(datasets)), np.nan)

for i, region_name in enumerate(region):
    for j, dataset_name in enumerate(datasets.keys()):
        if dataset_name in metrics[region_name]:
            mbe_values[i, j] = metrics[region_name][dataset_name]["MBE"]
            pcc_values[i, j] = metrics[region_name][dataset_name]["PCC"]

im1 = axes[0].imshow(mbe_values, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
axes[0].set_xticks(np.arange(len(datasets)))
axes[0].set_yticks(np.arange(len(region)))
axes[0].set_xticklabels(datasets.keys(), rotation=45)
axes[0].set_yticklabels(region)
plt.colorbar(im1, ax=axes[0], label="Mean Bias Error (MBE)")

# Add MBE values to the plot
for i in range(len(region)):
    for j in range(len(datasets)):
        if not np.isnan(mbe_values[i, j]):
            axes[0].text(j, i, f"{mbe_values[i, j]:.2f}", ha='center', va='center', color='black')

im2 = axes[1].imshow(pcc_values, cmap=custom_cmap, aspect="auto", vmin=0, vmax=1)
axes[1].set_xticks(np.arange(len(datasets)))
axes[1].set_yticks(np.arange(len(region)))
axes[1].set_xticklabels(datasets.keys(), rotation=45)
axes[1].set_yticklabels(region)
plt.colorbar(im2, ax=axes[1], label=r"Correlation coefficient $r$")
# Add PCC values to the plot
for i in range(len(region)):
    for j in range(len(datasets)):
        if not np.isnan(pcc_values[i, j]):
            axes[1].text(j, i, f"{pcc_values[i, j]:.2f}", ha='center', va='center', color='black')

plt.tight_layout()
fig.savefig("/home/6196306/Data/Figures/1950-2023/dataset_precipitation_comparison_MBE&PCC_v4.pdf", bbox_inches="tight")
fig.savefig("/home/6196306/Data/Figures/1950-2023/dataset_precipitation_comparison_MBE&PCC_v4.jpg", bbox_inches="tight", dpi=1200)