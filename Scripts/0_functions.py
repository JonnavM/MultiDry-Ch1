#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Mar 12 13:05:00 2024

@author: 6196306
Drought functions 
"""

import pandas as pd
import xarray as xr
import numpy as np

#MYD periods masks
def MYD(spei, region_name):
    """
    Function to calculate the start, end and length of multi-year droughts. 
    Input: timeseries of SPEI, and a string with the region name. 
    Output: start_date, end_date, length
    """
    print(region_name)
    i=0
    start_date = []
    end_date = []
    length = []
    print("Check for droughts within spei<=-1")
    for t in spei.time:
      if spei.sel(time=t).mean()<=-1:
        i=i+1
        if i==1:
            start_drought = t.values
            start = t
        elif t==spei.time[-1] and i>=12:
            start_date.append(start)
            end_date.append(t)
            length.append(i)
            print("From ", start_drought, " until ", t.values, ". Duration is ", i, " months.")
      else:
        if i>=12:
            start_date.append(start)
            end = t.values-pd.DateOffset(months=1)
            end_date.append(spei.time.sel(time=end))
            length.append(i)
            #end_date.append(t)
            print("From ", start_drought, " until ", end, ". Duration is ", i, " months.")
        i=0
        continue
    return start_date, end_date, length

def ND(spei, region_name):
    """
    Function to calculate the start, end and length of shorter (<12 months) droughts. 
    Input: timeseries of SPEI, and a string with the region name. 
    Output: start_date, end_date, length
    """
    print(region_name)
    i=0
    start_date = []
    end_date = []
    length = []
    print("Check for droughts within spei<=-1")
    for t in spei.time:
      if spei.sel(time=t).mean()<=-1:
        i=i+1
        if i==1:
            start_drought = t.values
            start = t
        elif t==spei.time[-1] and i>=0 and i <12:
            start_date.append(start)
            end_date.append(t)
            length.append(i)
            print("From ", start_drought, " until ", t.values, ". Duration is ", i, " months.")
      else:
        if (i>0) & (i<12):
            start_date.append(start)
            end = t.values-pd.DateOffset(months=1)
            end_date.append(spei.time.sel(time=end))
            length.append(i)
            #end_date.append(t)
            print("From ", start_drought, " until ", end, ". Duration is ", i, " months.")
        i=0
        continue
    return start_date, end_date, length

def mask_MYD(SPEI, region_name):
    """
    Function to rephrase MYD to a timeseries with True for the timesteps where a MYD occurs.
    Input: timeseries of SPEI, string with region name
    Output: timeseries with True and False
    """
    MYD_reg = MYD(SPEI, region_name)
    mask_MYD_reg = xr.DataArray(False, dims=("time",), coords={"time": SPEI.time})
    for start, end, length in zip(*MYD_reg):
        # Set True values for the specified time slices
        mask_MYD_reg = mask_MYD_reg | ((SPEI.time >= np.datetime64(start.values)) & (SPEI.time <= np.datetime64(end.values)))
    return mask_MYD_reg

def mask_ND(SPEI, region_name):
    """
    Function to rephrase MYD to a timeseries with True for the timesteps where a normal drought occurs.
    Input: timeseries of SPEI, string with region name
    Output: timeseries with True and False
    """
    ND_reg = ND(SPEI, region_name)
    mask_ND_reg = xr.DataArray(False, dims=("time",), coords={"time": SPEI.time})
    for start, end, length in zip(*ND_reg):
        # Set True values for the specified time slices
        mask_ND_reg = mask_ND_reg | ((SPEI.time >= np.datetime64(start.values)) & (SPEI.time <= np.datetime64(end.values)))
    return mask_ND_reg
