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
years = np.arange(1950, 2023, 1)
for year in years:
    print(year)
    res = "0_5"
    sel_time = slice(str(year)+"-01-01", str(year)+"-12-31")
       
    #Load in datasets    
    dir = "/scratch/6196306/ERA5/"
    tmax = xr.open_dataset(dir+"t2m/era5_2m_max_temperature_"+str(year)+"_daily_"+str(res)+".nc").t2m.sel(time=sel_time)-273.15 # Daily maximum temperature [°C]
    tmax['time'] = tmax.indexes['time'].normalize()
    tmin = xr.open_dataset(dir+"t2m/era5_2m_min_temperature_"+str(year)+"_daily_"+str(res)+".nc").t2m.sel(time=sel_time)-273.15 # Daily minimum temperature [°C]
    tmin['time'] = tmin.indexes['time'].normalize()
    tmean = xr.open_dataset(dir+"t2m/era5_2m_temperature_"+str(year)+"_daily_"+str(res)+".nc").t2m.sel(time=sel_time)-273.15 # Daily mean temperature [°C]
    tmean['time'] = tmean.indexes['time'].normalize()
    t_dew_mean = xr.open_dataset(dir+"d2m/era5_2m_dewpoint_temperature_"+str(year)+"_daily_"+str(res)+".nc").d2m.sel(time=sel_time)-273.15 # Daily dew temperature [°C]
    t_dew_mean['time'] = t_dew_mean.indexes['time'].normalize()
    t_dew_max = xr.open_dataset(dir+"d2m/era5_2m_max_dewpoint_temperature_"+str(year)+"_daily_"+str(res)+".nc").d2m.sel(time=sel_time)-273.15 #Daily max dew temperature [°C]
    t_dew_max['time'] = t_dew_max.indexes['time'].normalize()
    t_dew_min = xr.open_dataset(dir+"d2m/era5_2m_min_dewpoint_temperature_"+str(year)+"_daily_"+str(res)+".nc").d2m.sel(time=sel_time)-273.15 #Daily min dew temperature [°C]
    t_dew_min['time'] = t_dew_min.indexes['time'].normalize()
    print("Temperature and dewpoint temperature loaded in")
    
    #Calculate vapour pressure
    e_a_mean = 0.6108*np.exp((17.27*t_dew_mean)/(t_dew_mean+237.3))
    e_a_max = 0.6108*np.exp((17.27*t_dew_max)/(t_dew_max+237.3)) 
    e_a_min = 0.6108*np.exp((17.27*t_dew_min)/(t_dew_min+237.3)) 
    e_T_mean = 0.6108*np.exp((17.27*tmean)/(tmean+237.3))
    e_T_max = 0.6108*np.exp((17.27*tmax)/(tmax+237.3))
    e_T_min = 0.6108*np.exp((17.27*tmin)/(tmin+237.3))
    print("e_a and e_T calculated")
    
    rh_mean = 100*e_a_mean/e_T_mean # Daily mean relative humidity [%]
    rh_max = 100*e_a_mean/e_T_min
    rh_min = 100*e_a_mean/e_T_max
    print("Relative humidity calculated")

    u10 = xr.open_dataset(dir+"uwind/era5_10m_u_component_of_wind_"+str(year)+"_daily_"+str(res)+".nc").u10.sel(time=sel_time) #u wind at 10 m [m/s]
    u10['time'] = u10.indexes['time'].normalize()
    v10 = xr.open_dataset(dir+"vwind/era5_10m_v_component_of_wind_"+str(year)+"_daily_"+str(res)+".nc").v10.sel(time=sel_time) #v wind at 10 m [m/s]
    v10['time'] = v10.indexes['time'].normalize()
    uz = np.sqrt(u10**2+v10**2)  # Wind speed at 10 m [m/s]
    z = 10  # Height of wind measurement [m]
    wind_fao56 = uz * 4.87 / np.log(67.8*z-5.42)  # wind speed at 2 m after Allen et al., 1998
    print("Wind loaded in and calculated for 2m instead of 10m")
    
    p = xr.open_dataset(dir+"surfpres/era5_surface_pressure_"+str(year)+"_daily_"+str(res)+".nc").sp.sel(time=sel_time)*1e-3 #Surface pressure [kPa]
    p['time'] = p.indexes['time'].normalize()
    print("Surface pressure loaded in")
    
    rs = xr.open_dataset(dir+"solar_rad/era5_surface_net_solar_radiation_"+str(year)+"_daily_"+str(res)+".nc").ssr.sel(time=sel_time)*1e-6 # Compute solar radiation [MJ/m2day]
    rs['time'] = rs.indexes['time'].normalize()
    rt = xr.open_dataset(dir+"therm_rad/era5_surface_net_thermal_radiation_"+str(year)+"_daily_"+str(res)+".nc").str.sel(time=sel_time)*1e-6 #thermal radiation [MJ/m2day]
    rt['time'] = rt.indexes['time'].normalize()
    rn = rs + rt #Turn into + if rt<0! Net radiation
    time = tmean.time
    lat = tmean.lat
    elevation = 2
    print("Radiation loaded in and calculated")
    
    pm_fao56 = pyet.pm_fao56(tmean, wind=wind_fao56, rs=rs, rn=rn, pressure=p, elevation=elevation, lat=lat, tmax=tmax, tmin=tmin, rh=rh_mean, rhmax=rh_max, rhmin=rh_min)
    print("Penman-Monteith calculated")
    
    pm_fao56.to_netcdf("/scratch/6196306/PET/PenmanMonteith/pm_fao56_"+str(year)+"_daily_"+str(res)+"_v3.nc")
    print("Penman-Monteith saved to netCDF")
