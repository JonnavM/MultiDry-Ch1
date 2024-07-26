#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:57:28 2024

@author: 6196306
e-folding timescales based on auto-correlation
e-folding nog veranderen e-macht fitted
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

#Data
SPEI12_AUS = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_AUS.nc").sel(time=slice("1951", "2023"))
SPEI12_ARG = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_ARG.nc").sel(time=slice("1951", "2023"))
SPEI12_WEU = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_WEU.nc").sel(time=slice("1951", "2023"))
SPEI12_CAL = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_CAL.nc").sel(time=slice("1951", "2023"))
SPEI12_IND = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_IND.nc").sel(time=slice("1951", "2023"))
SPEI12_SA = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree_SA.nc").sel(time=slice("1951", "2023"))

SPEI6_AUS = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI06_monthly_1950_2023_0_5_degree_AUS.nc").sel(time=slice("1951", "2023"))
SPEI6_ARG = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI06_monthly_1950_2023_0_5_degree_ARG.nc").sel(time=slice("1951", "2023"))
SPEI6_WEU = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI06_monthly_1950_2023_0_5_degree_WEU.nc").sel(time=slice("1951", "2023"))
SPEI6_CAL = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI06_monthly_1950_2023_0_5_degree_CAL.nc").sel(time=slice("1951", "2023"))
SPEI6_IND = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI06_monthly_1950_2023_0_5_degree_IND.nc").sel(time=slice("1951", "2023"))
SPEI6_SA = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI06_monthly_1950_2023_0_5_degree_SA.nc").sel(time=slice("1951", "2023"))

#Complete grid
SPEI12_grid = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI12_monthly_1950_2023_0_5_degree.nc").sel(time=slice("1951", "2023"))

# Masks
mask_AUS = xr.open_dataarray("/scratch/6196306/masks/mask_AUS.nc")
mask_WEU = xr.open_dataarray("/scratch/6196306/masks/mask_WEU.nc")
mask_IND = xr.open_dataarray("/scratch/6196306/masks/mask_IND.nc")
mask_SA = xr.open_dataarray("/scratch/6196306/masks/mask_SA.nc") 
mask_ARG = xr.open_dataarray("/scratch/6196306/masks/mask_ARG.nc")
mask_CAL = xr.open_dataarray("/scratch/6196306/masks/mask_CAL.nc")

#Separate variables
pr = xr.open_dataarray("/scratch/6196306/ERA5/total_precipitation/era5_total_precipitation_1950-2023_monthly_0_5.nc").rolling(time=12).mean("time").sel(time=slice("1951", "2023"))*1000
pet = xr.open_dataset("/scratch/6196306/PET/PenmanMonteith/pm_fao56_1950-2023_monthly_0_5_v3.nc").PM_FAO_56.rolling(time=12).mean("time").sel(time=slice("1951", "2023"))
t2m = xr.open_dataset("/scratch/6196306/ERA5/t2m/era5_2m_temperature_1950-2023_monthly_0_5.nc").t2m.rolling(time=12).mean("time").sel(time=slice("1951", "2023"))
d2m = xr.open_dataset("/scratch/6196306/ERA5/d2m/era5_2m_dewpoint_temperature_1950-2023_monthly_0_5.nc").d2m.rolling(time=12).mean("time").sel(time=slice("1951", "2023"))
uwind = xr.open_dataset("/scratch/6196306/ERA5/uwind/era5_10m_u_component_of_wind_1950-2023_monthly_0_5.nc").u10
vwind = xr.open_dataset("/scratch/6196306/ERA5/vwind/era5_10m_v_component_of_wind_1950-2023_monthly_0_5.nc").v10
wind = np.sqrt(uwind**2+vwind**2).rolling(time=12).mean("time").sel(time=slice("1951", "2023"))
surf_pres = xr.open_dataset("/scratch/6196306/ERA5/surfpres/era5_surface_pressure_1950-2023_monthly_0_5.nc").sp.rolling(time=12).mean("time").sel(time=slice("1951", "2023"))
sol_rad = xr.open_dataset("/scratch/6196306/ERA5/solar_rad/era5_surface_net_solar_radiation_1950-2023_monthly_0_5.nc").ssr.rolling(time=12).mean("time").sel(time=slice("1951", "2023"))
therm_rad = xr.open_dataset("/scratch/6196306/ERA5/therm_rad/era5_surface_net_thermal_radiation_1950-2023_monthly_0_5.nc").str.rolling(time=12).mean("time").sel(time=slice("1951", "2023"))

max_lag = 24  # Choose the maximum lag you want to consider

SPEI12 = [SPEI12_CAL, SPEI12_WEU, SPEI12_IND, SPEI12_ARG, SPEI12_SA, SPEI12_AUS]
title = ["California", "Western Europe", "India", "Argentina", "South Africa", "Australia"]
reg = ["CAL", "WEU", "IND", "ARG", "SA", "AUS"]

#%% Functions
def auto_lag(x, max_lag):
    "Returns the autocorrelation of a variable x for all time differences (lags) within max_lag"
    correlations = []
    lags = np.arange(0, max_lag + 1)

    for lag in lags:
        shifted_x = np.roll(x, shift=lag)
        correlation = np.corrcoef(x, shifted_x)[0, 1]
        correlations.append(correlation)

    return lags, correlations

def bootstrap_fit_data(data, max_lag, step, num_bootstraps=1000):
    bootstrap_results = []

    for _ in range(num_bootstraps):
        # Resample with replacement
        resampled_noise = np.random.choice(data, size=len(data), replace=True)
        # Take 12 month running mean
        da_bootstrap = xr.DataArray(data=resampled_noise, dims=["time"], coords={"time": data.time}).rolling(time=step).mean("time").sel(time=slice("1951", "2023"))
        # Take autocorrelation
        lags_bootstrap, corr_bootstrap = auto_lag(da_bootstrap, max_lag)
        bootstrap_results.append(corr_bootstrap)

    # Calculate confidence intervals from bootstrap results
    conf_interval_bootstrapped_95 = np.percentile(bootstrap_results, [2.5, 97.5], axis=0)
    conf_interval_bootstrapped_90 = np.percentile(bootstrap_results, [5, 95], axis=0)
    median = np.median(bootstrap_results, axis=0)

    return conf_interval_bootstrapped_95, conf_interval_bootstrapped_90, median

def sel_variable(var):
    var_ind = var.sel(lat=slice(22, 32), lon=slice(72, 90)).where(mask_IND==1).mean(dim=("lat", "lon"))
    var_aus = var.sel(lat=slice(-40, -20), lon=slice(135, 155)).where(mask_AUS==1).mean(dim=("lat", "lon"))
    var_weu = var.sel(lat=slice(45, 55), lon=slice(-1, 13)).where(mask_WEU==1).mean(dim=("lat", "lon"))
    var_sa = var.sel(lat=slice(-33, -21), lon=slice(15, 31)).where(mask_SA==1).mean(dim=("lat", "lon"))
    var_arg = var.sel(lat=slice(-45, -25), lon=slice(-75, -55)).where(mask_ARG==1).mean(dim=("lat", "lon"))
    var_cal = var.sel(lat=slice(32, 41.5), lon=slice(-124, -115)).where(mask_CAL==1).mean(dim=("lat", "lon"))
    return var_ind, var_aus, var_weu, var_sa, var_arg, var_cal

#%% Check all trends:
var_index = [sel_variable(t2m), sel_variable(d2m), sel_variable(wind), sel_variable(surf_pres), sel_variable(sol_rad), sel_variable(therm_rad)]
var = ["t2m", "d2m", "wind", "slp", "ssr", "str"]

for i in range(6):
    time = var_index[i][0].time
    time_len = np.arange(0, len(time), 1)
    
    fig, ax = plt.subplots(3, 2, sharex=True, tight_layout=True, figsize=(10,7))
    #Australia
    a_aus, b_aus = np.polyfit(x=time_len, y=var_index[i][1].values, deg=1)
    corr_aus = np.corrcoef(time_len, var_index[i][1])[0,1]
    ax[0,0].fill_between(time, var_index[i][1].mean()-var_index[i][1].std(), var_index[i][1].mean()+var_index[i][1].std(), alpha=0.1, color="red")
    var_index[i][1].plot(ax=ax[0,0], label="1 year running mean")
    ax[0,0].axhline(y=var_index[i][1].mean(), color="red", label=r"$\mu \pm \sigma$")
    ax[0,0].plot(time, a_aus*time_len+b_aus, color="black", linestyle="dashed", label="trend")
    ax[0,0].set_title(f"AUS, r={corr_aus:.2f}")
    #West Europe
    a_weu, b_weu = np.polyfit(x=time_len, y=var_index[i][2].values, deg=1)
    corr_weu = np.corrcoef(time_len, var_index[i][2])[0,1]
    ax[0,1].fill_between(time, var_index[i][2].mean()-var_index[i][2].std(), var_index[i][2].mean()+var_index[i][2].std(), alpha=0.1, color="red")
    var_index[i][2].plot(ax=ax[0,1])
    ax[0,1].axhline(y=var_index[i][2].mean(), color="red")
    ax[0,1].plot(time, a_weu*time_len+b_weu, color="black", linestyle="dashed")
    ax[0,1].set_title(f"WEU, r={corr_weu:.2f}")
    ax[0,1].set_ylabel(" ")
    #India
    a_ind, b_ind = np.polyfit(x=time_len, y=var_index[i][0].values, deg=1)
    corr_ind = np.corrcoef(time_len, var_index[i][0])[0,1]
    ax[1,0].fill_between(time, var_index[i][0].mean()-var_index[i][0].std(), var_index[i][0].mean()+var_index[i][0].std(), alpha=0.1, color="red")
    var_index[i][0].plot(ax=ax[1,0])
    ax[1,0].axhline(y=var_index[i][0].mean(), color="red")
    ax[1,0].plot(time, a_ind*time_len+b_ind, color="black", linestyle="dashed")
    ax[1,0].set_title(f"IND, r={corr_ind:.2f}")
    #South Africa
    a_sa, b_sa = np.polyfit(x=time_len, y=var_index[i][3].values, deg=1)
    corr_sa = np.corrcoef(time_len, var_index[i][3])[0,1]
    ax[1,1].fill_between(time, var_index[i][3].mean()-var_index[i][3].std(), var_index[i][3].mean()+var_index[i][3].std(), alpha=0.1, color="red")
    var_index[i][3].plot(ax=ax[1,1])
    ax[1,1].axhline(y=var_index[i][3].mean(), color="red")
    ax[1,1].plot(time, a_sa*time_len+b_sa, color="black", linestyle="dashed")
    ax[1,1].set_title(f"SA, r={corr_sa:.2f}")
    ax[1,1].set_ylabel(" ")
    #Argentina
    a_ssa, b_ssa = np.polyfit(x=time_len, y=var_index[i][4].values, deg=1)
    corr_ssa = np.corrcoef(time_len, var_index[i][4])[0,1]
    ax[2,0].fill_between(time, var_index[i][4].mean()-var_index[i][4].std(), var_index[i][4].mean()+var_index[i][4].std(), alpha=0.1, color="red")
    var_index[i][4].plot(ax=ax[2,0])
    ax[2,0].axhline(y=var_index[i][4].mean(), color="red")
    ax[2,0].plot(time, a_ssa*time_len+b_ssa, color="black", linestyle="dashed")
    ax[2,0].set_title(f"ARG, r={corr_ssa:.2f}")
    #California
    a_cal, b_cal = np.polyfit(x=time_len, y=var_index[i][5].values, deg=1)
    corr_cal = np.corrcoef(time_len, var_index[i][5])[0,1]
    ax[2,1].fill_between(time, var_index[i][5].mean()-var_index[i][5].std(), var_index[i][5].mean()+var_index[i][5].std(), alpha=0.1, color="red")
    var_index[i][5].plot(ax=ax[2,1])
    ax[2,1].axhline(y=var_index[i][5].mean(), color="red")
    ax[2,1].plot(time, a_cal*time_len+b_cal, color="black", linestyle="dashed")
    ax[2,1].set_title(f"CAL, r={corr_cal:.2f}")
    ax[2,1].set_ylabel(" ")
    fig.legend(loc = 7, bbox_to_anchor=(1.2, 0.9))
    fig.suptitle(var[i])
    fig.savefig("/home/6196306/Data/Figures/1950-2023/trend_"+str(var[i])+"_running12month_v3.jpg", dpi=1200, bbox_inches="tight")
    fig.savefig("/home/6196306/Data/Figures/1950-2023/trend_"+str(var[i])+"_running12month_v3.pdf", bbox_inches="tight")


#%% Do lagged autocorrelations for PET and precipitation
# e-folding time scale after decreasing with 1/e
#Add white noise
#Create imaginary SPEI-12 time series as an example for white noise
test_time = xr.open_dataarray("/scratch/ruiij001/Data/SPEI/0_5_degrees_apr_2024/SPEI06_monthly_1950_2023_0_5_degree.nc")
np.random.seed(42)
white_noise = np.clip(np.random.normal(loc=0, scale=1, size=len(test_time)), -3, 3)

#Take 12 month running mean
da_white = xr.DataArray(data=white_noise, dims=["time"], coords={"time":test_time.time})#.rolling(time=12).mean("time").sel(time=slice("1956", "2022"))

#Take autocorrelation
lags_white, corr_white = auto_lag(da_white, max_lag)
conf_interval_bs_dd_95 = bootstrap_fit_data(da_white, max_lag, step=12)[0]#white_noise, max_lag)[0]
conf_interval_bs_dd_90 = bootstrap_fit_data(da_white, max_lag, step=12)[1]
median_bs_dd = bootstrap_fit_data(da_white, max_lag, step=12)[2]

#Also for all regions
lags_aus, corr_aus = auto_lag(SPEI12_AUS.values, max_lag)
lags_weu, corr_weu = auto_lag(SPEI12_WEU.values, max_lag)
lags_ind, corr_ind = auto_lag(SPEI12_IND.values, max_lag)
lags_sa, corr_sa = auto_lag(SPEI12_SA.values, max_lag)
lags_arg, corr_arg = auto_lag(SPEI12_ARG.values, max_lag)
lags_cal, corr_cal = auto_lag(SPEI12_CAL.values, max_lag)

lags_pr_aus, corr_pr_aus = auto_lag(pr.where(mask_AUS==1).mean(dim=("lat", "lon")), max_lag)
lags_pr_weu, corr_pr_weu = auto_lag(pr.where(mask_WEU==1).mean(dim=("lat", "lon")), max_lag)
lags_pr_ind, corr_pr_ind = auto_lag(pr.where(mask_IND==1).mean(dim=("lat", "lon")), max_lag)
lags_pr_sa, corr_pr_sa = auto_lag(pr.where(mask_SA==1).mean(dim=("lat", "lon")), max_lag)
lags_pr_arg, corr_pr_arg = auto_lag(pr.where(mask_ARG==1).mean(dim=("lat", "lon")), max_lag)
lags_pr_cal, corr_pr_cal = auto_lag(pr.where(mask_CAL==1).mean(dim=("lat", "lon")), max_lag)

lags_pet_aus, corr_pet_aus = auto_lag(pet.where(mask_AUS==1).mean(dim=("lat", "lon")), max_lag)
lags_pet_weu, corr_pet_weu = auto_lag(pet.where(mask_WEU==1).mean(dim=("lat", "lon")), max_lag)
lags_pet_ind, corr_pet_ind = auto_lag(pet.where(mask_IND==1).mean(dim=("lat", "lon")), max_lag)
lags_pet_sa, corr_pet_sa = auto_lag(pet.where(mask_SA==1).mean(dim=("lat", "lon")), max_lag)
lags_pet_arg, corr_pet_arg = auto_lag(pet.where(mask_ARG==1).mean(dim=("lat", "lon")), max_lag)
lags_pet_cal, corr_pet_cal = auto_lag(pet.where(mask_CAL==1).mean(dim=("lat", "lon")), max_lag)

#Calculate bootstraps for precipitation and PET (no white noise, just reshuffling of data)
pr_orig = xr.open_dataarray("/scratch/6196306/ERA5/total_precipitation/era5_total_precipitation_1950-2023_monthly_0_5.nc").sel(time=slice("1950", "2023"))
pet_orig = xr.open_dataset("/scratch/6196306/PET/PenmanMonteith/pm_fao56_1950-2023_monthly_0_5_v3.nc").PM_FAO_56.sel(time=slice("1950", "2023"))
ci_pr_95 = bootstrap_fit_data(pr_orig.where(mask_ARG==1).mean(dim=("lat", "lon")), max_lag, step=12)[0]
ci_pr_90 = bootstrap_fit_data(pr_orig.where(mask_ARG==1).mean(dim=("lat", "lon")), max_lag, step=12)[1]
median_pr = bootstrap_fit_data(pr_orig.where(mask_ARG==1).mean(dim=("lat", "lon")), max_lag, step=12)[2]

ci_pet_95 = bootstrap_fit_data(pet_orig.where(mask_ARG==1).mean(dim=("lat", "lon")), max_lag, step=12)[0]
ci_pet_90 = bootstrap_fit_data(pet_orig.where(mask_ARG==1).mean(dim=("lat", "lon")), max_lag, step=12)[1]
median_pet = bootstrap_fit_data(pet_orig.where(mask_ARG==1).mean(dim=("lat", "lon")), max_lag, step=12)[2]

#%% Combine lagged auto-correlation for SPEI-12, Precipitation, and PET
cb_color = ["#661100", "#aa4499", "#1f77b4", "#117733", "#ddcc77", "#332288"]

lags_spei = [lags_cal, lags_weu, lags_ind, lags_arg, lags_sa, lags_aus]
lags_pr = [lags_pr_cal, lags_pr_weu, lags_pr_ind, lags_pr_arg, lags_pr_sa, lags_pr_aus]
lags_pet = [lags_pet_cal, lags_pet_weu, lags_pet_ind, lags_pet_arg, lags_pet_sa, lags_pet_aus]
corr_spei = [corr_cal, corr_weu, corr_ind, corr_arg, corr_sa, corr_aus]
corr_pr = [corr_pr_cal, corr_pr_weu, corr_pr_ind, corr_pr_arg, corr_pr_sa, corr_pr_aus]
corr_pet = [corr_pet_cal, corr_pet_weu, corr_pet_ind, corr_pet_arg, corr_pet_sa, corr_pet_aus]

fig, ax = plt.subplot_mosaic("ABC", figsize=(12,5), layout="tight", sharey=True)
plt.rcParams.update({
    'font.size': 14,            # Controls default text sizes
    'axes.labelsize': 14,       # Font size of x and y labels
    'axes.titlesize': 14,       # Font size of title
    'xtick.labelsize': 14,      # Font size of x-axis tick labels
    'ytick.labelsize': 14,      # Font size of y-axis tick labels
    'legend.fontsize': 14,      # Font size of legend
    'figure.titlesize': 14,     # Font size of figure title
})

#SPEI-12
ax["A"].plot(lags_white, median_bs_dd, label="White noise median", color="black", linestyle="-.")
ax["A"].fill_between(lags_white, conf_interval_bs_dd_95[0], conf_interval_bs_dd_95[1], color='black', alpha=0.2, label='95% CI')
ax["A"].fill_between(lags_white, conf_interval_bs_dd_90[0], conf_interval_bs_dd_90[1], color='black', alpha=0.25, label='90% CI')
for i in range(len(reg)):
    ax["A"].plot(lags_spei[i], corr_spei[i], label=reg[i], color=cb_color[i])
ax["A"].set_title('SPEI-12')
ax["A"].set_title("a)", loc="left")
ax["A"].set_ylabel('Correlation')
ax["A"].set_xlim(0, max_lag)
ax["A"].set_xticks([0,4,8,12,16,20,24])
ax["A"].grid(True)
#Precipitation
ax["B"].plot(lags_white, median_pr, label="White noise median", color="black", linestyle="-.")
ax["B"].fill_between(lags_white, ci_pr_95[0], ci_pr_95[1], color='black', alpha=0.2, label='95% CI')
ax["B"].fill_between(lags_white, ci_pr_90[0], ci_pr_90[1], color='black', alpha=0.25, label='90% CI')
for i in range(len(reg)):
    ax["B"].plot(lags_pr[i], corr_pr[i], label=reg[i], color=cb_color[i])
ax["B"].set_title("Precipitation")
ax["B"].set_title("b)", loc="left")
ax["B"].set_xlabel('Lag [months]')
ax["B"].set_xlim(0, max_lag)
ax["B"].set_xticks([0,4,8,12,16,20,24])
ax["B"].grid(True)
#PET
ax["C"].plot(lags_white, median_pet, label="White noise median", color="black", linestyle="-.")
ax["C"].fill_between(lags_white, ci_pet_95[0], ci_pet_95[1], color='black', alpha=0.2, label='95% CI')
ax["C"].fill_between(lags_white, ci_pet_90[0], ci_pet_90[1], color='black', alpha=0.25, label='90% CI')
for i in range(len(reg)):
    ax["C"].plot(lags_pet[i], corr_pet[i], label=reg[i], color=cb_color[i])
ax["C"].set_title("PET")
ax["C"].set_title("c)", loc="left")
ax["C"].grid(True)
ax["C"].set_xlim(0, max_lag)
ax["C"].set_xticks([0,4,8,12,16,20,24])
ax["C"].tick_params(labelleft=False)

handles, labels = ax["A"].get_legend_handles_labels()
fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.64), ncol=1)

plt.subplots_adjust(wspace=0.05)  # Adjust the value as needed

fig.savefig("/home/6196306/Data/Figures/1950-2023/lag_autocorr+e-fold_SPEI12_pet_pr_v3.jpg", dpi=1200, bbox_inches="tight")
fig.savefig("/home/6196306/Data/Figures/1950-2023/lag_autocorr+e-fold_SPEI12_pet_pr_v3.pdf", bbox_inches="tight")
