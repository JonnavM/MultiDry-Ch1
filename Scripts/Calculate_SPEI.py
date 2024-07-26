#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 08:48:38 2024

@author: 6196306
"""

import xclim
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pint
from xclim import indices
from xclim.core import units
from xclim.indices import standardized_precipitation_evapotranspiration_index
import pandas as pd
import spei as si  # si for standardized index

xr.set_options(keep_attrs=True)

#%% Define SPEI
def SPEI_region(region_name, prec, pet, spei_period, offset, cal_start, cal_end, dir):
    if region_name == "IND": #IND, AUS, WEU, SA, ARG, CAL
        reg_lat = slice(10,47)
        reg_lon = slice(60,110)
        region_mask = xr.open_dataarray("/scratch/6196306/masks/mask_IND.nc").sel(lat=reg_lat, lon=reg_lon)
    elif region_name == "PAK":
        reg_lat = slice(20,40)
        reg_lon = slice(62,85)
        region_mask = xr.open_dataarray("/scratch/6196306/masks/mask_PAK.nc").sel(lat=reg_lat, lon=reg_lon)
    if region_name == "BAH": 
        reg_lat = slice(19,37) 
        reg_lon = slice(81,99)
        region_mask = xr.open_dataarray("/scratch/ruiij001/Data/Masks/mask_Brahmaputra_0_5_degrees.nc").sel(lat=reg_lat, lon=reg_lon)
    elif region_name == "AUS":
        reg_lat = slice(-50, -10)
        reg_lon = slice(120, 165)
        region_mask = xr.open_dataarray("/scratch/6196306/masks/mask_AUS.nc").sel(lat=reg_lat, lon=reg_lon)
    elif region_name == "WEU":
        reg_lat = slice(20, 70)
        reg_lon = slice(-30,30)
        region_mask = xr.open_dataarray("/scratch/6196306/masks/mask_WEU.nc").sel(lat=reg_lat, lon=reg_lon)
    elif region_name == "SA":
        reg_lat = slice(-45,-12)
        reg_lon = slice(2, 50)
        region_mask = xr.open_dataarray("/scratch/6196306/masks/mask_SA.nc").sel(lat=reg_lat, lon=reg_lon)
    elif region_name == "ARG":
        reg_lat = slice(-45, -25)
        reg_lon = slice(-75, -55)
        region_mask = xr.open_dataarray("/scratch/6196306/masks/mask_ARG.nc").sel(lat=reg_lat, lon=reg_lon)
    elif region_name == "CAL":
        reg_lat = slice(15, 60)
        reg_lon = slice(-138, -100)
        region_mask = xr.open_dataarray("/scratch/6196306/masks/mask_CAL.nc").sel(lat=reg_lat, lon=reg_lon)
        
    prec_region = prec.where(region_mask==1)
    pet_region = pet.where(region_mask==1)
    
    prec_region_mean = prec_region.mean(dim = ["lon","lat"])
    pet_region_mean = pet_region.mean(dim = ["lon","lat"])
    
    #pe_region_mean = prec_region_mean.sel(time = slice("1955-01-01","2022-12-31")).assign_attrs(units='mm/d') - pet_region_mean.sel(time = slice("1955-01-01","2022-12-31")).assign_attrs(units='mm/d')
    pe_region_mean = prec_region_mean.assign_attrs(units='mm/d') - pet_region_mean.assign_attrs(units='mm/d')
    print("calculating spei")
    SPEI = standardized_precipitation_evapotranspiration_index(pe_region_mean, window = spei_period, dist = "fisk",freq= "MS", offset=offset,  cal_start = cal_start, cal_end = cal_end)

    del SPEI.attrs['freq']
    del SPEI.attrs['time_indexer']
    del SPEI.attrs['units']
    del SPEI.attrs['offset']
    print("saving")
    SPEI.to_netcdf(path = "/scratch/ruiij001/Data/SPEI/" + dir)
    print("done")
    
    SPEI12 = xr.open_dataset("/scratch/ruiij001/Data/SPEI/" + dir).__xarray_dataarray_variable__
    df_spei = SPEI12.to_pandas()
    
    f, ax = plt.subplots(1, 1, figsize=(16, 9), sharex=False)
    si.plot.si(df_spei[11:], ax=ax)
    [ax.set_ylabel(n, fontsize=14) for i, n in enumerate(["SPEI"])]
    
    df_pre = prec_region_mean.to_pandas()
    df_pet = pet_region_mean.to_pandas()
    df_pe = pe_region_mean.to_pandas()
    
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    df_pre[:-2].plot(ax=ax[0], legend=True, grid=True, label = "Total Precipitation (mm)").legend(loc='upper left')
    df_pet.plot(ax=ax[1], color="C1", legend=True, grid=True, label = "Potential Evapotranspiration (mm/day)")
    df_pe.plot(ax=ax[2], color="k", legend=True, grid=True, label = "PREC - PET")
    
#%% Load in data
# landmask, since we don't need data over the oceans
landmask = xr.open_dataarray("/scratch/6196306/ERA5/land-sea-mask_0_5.nc").mean("time")
# Precipitation, is in m, but we need it in mm.
total_prec_mm = (xr.open_dataset("/scratch/6196306/ERA5/total_precipitation/era5_total_precipitation_1950-2023_monthly_0_5.nc").tp*1000).where(landmask>=0.5)#.resample(time="MS").mean()
# PET, resample to monthly values
pet = xr.open_mfdataset("/scratch/6196306/PET/PenmanMonteith/pm_fao56_*_daily_0_5_v3.nc").PM_FAO_56.where(landmask>=0.5).resample(time="1MS").mean()

#%% Calculate SPEI
# Set the offset, specify start- and end of the calibration period. Mean is zero between these dates
prec = total_prec_mm
pet = pet
spei_period = 12
offset = '20 mm/d'
cal_start = "1950-01-01"
cal_end = "2020-12-31"

region_name = ["CAL", "WEU", "IND", "ARG", "SA", "AUS"]
dir = ["SPEI12_monthly_1950_2023_0_5_degree_CAL.nc", "SPEI12_monthly_1950_2023_0_5_degree_WEU.nc",
       "SPEI12_monthly_1950_2023_0_5_degree_IND.nc", "SPEI12_monthly_1950_2023_0_5_degree_ARG.nc", 
       "SPEI12_monthly_1950_2023_0_5_degree_SA.nc", "SPEI12_monthly_1950_2023_0_5_degree_AUS.nc"]

for i in range(len(region_name)):
    SPEI_region(region_name[i], prec, pet, spei_period, offset, cal_start, cal_end, dir[i])
