#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:31:36 2023

@author: 6196306
PET Penman-Monteith
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyet as pyet
import xarray as xr
#%%
#Specify year and resolution
years = np.arange(2009, 2023, 1)
for year in years:
    print(year)
    sel_time = slice(str(year)+"-01-01", str(year)+"-12-31")
       
    #Load in datasets    
    dir = "/scratch/6196306/JRA3Q/"
    tmax = xr.open_dataset(dir+"jra3q.anl_surf.0_0_0.tmp2m-hgt-an-gauss."+str(year)+".daymax_0_5.nc")["tmp2m-hgt-an-gauss"].sel(time=sel_time)-273.15 # Daily maximum temperature [°C]
    tmax['time'] = tmax.indexes['time'].normalize()
    tmin = xr.open_dataset(dir+"jra3q.anl_surf.0_0_0.tmp2m-hgt-an-gauss."+str(year)+".daymin_0_5.nc")["tmp2m-hgt-an-gauss"].sel(time=sel_time)-273.15 # Daily minimum temperature [°C]
    tmin['time'] = tmin.indexes['time'].normalize()
    tmean = xr.open_dataset(dir+"jra3q.anl_surf.0_0_0.tmp2m-hgt-an-gauss."+str(year)+".daymean_0_5.nc")["tmp2m-hgt-an-gauss"].sel(time=sel_time)-273.15 # Daily mean temperature [°C]
    tmean['time'] = tmean.indexes['time'].normalize()
    print("Temperature loaded in")
    
    rh_mean = xr.open_dataset(dir+"jra3q.anl_surf.0_1_1.rh2m-hgt-an-gauss."+str(year)+".daymean_0_5.nc")["rh2m-hgt-an-gauss"].sel(time=sel_time) # Daily mean relative humidity [%]
    rh_mean['time'] = rh_mean.indexes['time'].normalize()
    rh_max = xr.open_dataset(dir+"jra3q.anl_surf.0_1_1.rh2m-hgt-an-gauss."+str(year)+".daymax_0_5.nc")["rh2m-hgt-an-gauss"].sel(time=sel_time) # Daily max relative humidity [%]
    rh_max['time'] = rh_max.indexes['time'].normalize()
    rh_min = xr.open_dataset(dir+"jra3q.anl_surf.0_1_1.rh2m-hgt-an-gauss."+str(year)+".daymin_0_5.nc")["rh2m-hgt-an-gauss"].sel(time=sel_time) # Daily min relative humidity [%]
    rh_min['time'] = rh_min.indexes['time'].normalize()
    print("Relative humidity calculated")

    u10 = xr.open_dataset(dir+"jra3q.anl_surf.0_2_2.ugrd10m-hgt-an-gauss."+str(year)+".daymean_0_5.nc")["ugrd10m-hgt-an-gauss"].sel(time=sel_time) #u wind at 10 m [m/s]
    u10['time'] = u10.indexes['time'].normalize()
    v10 = xr.open_dataset(dir+"jra3q.anl_surf.0_2_3.vgrd10m-hgt-an-gauss."+str(year)+".daymean_0_5.nc")["vgrd10m-hgt-an-gauss"].sel(time=sel_time) #v wind at 10 m [m/s]
    v10['time'] = v10.indexes['time'].normalize()
    uz = np.sqrt(u10**2+v10**2)  # Wind speed at 10 m [m/s]
    z = 10  # Height of wind measurement [m]
    wind_fao56 = uz * 4.87 / np.log(67.8*z-5.42)  # wind speed at 2 m after Allen et al., 1998
    print("Wind loaded in and calculated for 2m instead of 10m")
    
    p = xr.open_dataset(dir+"jra3q.anl_surf.0_3_0.pres-sfc-an-gauss."+str(year)+".daymean_0_5.nc")["pres-sfc-an-gauss"].sel(time=sel_time)*1e-3 #Surface pressure [kPa]
    p['time'] = p.indexes['time'].normalize()
    print("Surface pressure loaded in")

    drs = xr.open_dataset(dir+"jra3q.fcst_phy2m.0_4_7.dswrf1have-sfc-fc-gauss."+str(year)+".daysum_0_5.nc")["dswrf1have-sfc-fc-gauss"].sel(time=sel_time)*3600*1e-6 # Compute solar radiation [MJ/m2day]
    drs['time'] = drs.indexes['time'].normalize()
    drt = xr.open_dataset(dir+"jra3q.fcst_phy2m.0_5_3.dlwrf1have-sfc-fc-gauss."+str(year)+".daysum_0_5.nc")["dlwrf1have-sfc-fc-gauss"].sel(time=sel_time)*3600*1e-6 #thermal radiation [MJ/m2day]
    drt['time'] = drt.indexes['time'].normalize()
    
    urs = xr.open_dataset(dir+"jra3q.fcst_phy2m.0_4_8.uswrf1have-sfc-fc-gauss."+str(year)+".daysum_0_5.nc")["uswrf1have-sfc-fc-gauss"].sel(time=sel_time)*3600*1e-6 # Compute solar radiation [MJ/m2day]
    urs['time'] = urs.indexes['time'].normalize()
    urt = xr.open_dataset(dir+"jra3q.fcst_phy2m.0_5_4.ulwrf1have-sfc-fc-gauss."+str(year)+".daysum_0_5.nc")["ulwrf1have-sfc-fc-gauss"].sel(time=sel_time)*3600*1e-6 #thermal radiation [MJ/m2day]
    urt['time'] = urt.indexes['time'].normalize()
    
    nrs = drs - urs
    nrt = drt - urt
    rn = nrs + nrt #Turn into + if rt<0! Net radiation
    time = tmean.time
    lat = tmean.lat
    elevation = 2
    print("Radiation loaded in and calculated")
    
    pm_fao56 = pyet.pm_fao56(tmean, wind=wind_fao56, rs=nrs, rn=rn, pressure=p, elevation=elevation, lat=lat, tmax=tmax, tmin=tmin, rh=rh_mean, rhmax=rh_max, rhmin=rh_min)
    print("Penman-Monteith calculated")
    
    pm_fao56.to_netcdf("/scratch/6196306/PET/PenmanMonteith/pm_fao56_"+str(year)+"_daily_JRA3Q.nc")
    print("Penman-Monteith saved to netCDF")
