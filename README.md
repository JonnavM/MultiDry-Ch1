This repository provides the necessary scripts to reproduce the results and figures from paper "drivers_characteristics_MYDs". 
It contains the following folders and content:
a. Scripts:
   0. Scripts with functions
   0_functions.py: contains the functions to calculate the multi-year droughts (MYD), the normal droughts (ND), and creates masks for these droughts (mask_MYD and mask_ND). These functions are loaded in in the other scripts where necessary.
   1. Scripts to calculate PET and SPEI
   1_PenmanMonteith.py: Calculates PET with use of ERA5 data
   1_PenmanMonteith_JRA.py: Calculates PET with use of JRA-3Q data
   1_PenmanMonteith_MERRA.py: Calculates PET with use of MERRA-2 data
   1_Calculate_SPEI.py: Calculates SPEI with use of PET and PR
   2. Scripts to validate data and results
   2_compare_SPEI.py: Compares SPEI-12 between ERA5, MERRA-2, and JRA-3Q. Results in Figures S5-10.
   2_compare_precipitation.py: Compares monthly precipitation between ERA5, CHIRPS, E-OBS, GPCP, CRU, and IMERG. Results in Figures S3 and S4.
   2_variogram.py: Checks if size of focus region is appropriate. Results in Figure S2.
   3. Scripts for figures
   3_Regression.py: Figures 3 (boxplot), 5 (PR and PET anomalies), 6 (linear regressions)
   3_SPEI12-06-03.py: Plots SPEI12, SPEI06, SPEI03, and SPEI01 to show development of MYDs. Results in Figure 4.
   3_SPEI_figures.py: Makes figures of SPEI-12 per region. Results in Figure 2.
   3_autocorrelation_trends.py: Makes plots for the lagged auto-correlation of SPEI-12, precipitation, and PET. Also plots the trends of all variables included in PET. Results in Figures 7 (lagged auto-correlations), and S11-16 (trends)
   3_worldmaps_climate.py: Makes maps of climatology of PR, PET, number of MYDs, duration of MYDs, intensity of MYDs, fraction of months in MYDs compared to NDs. Results in Figures 1 and S1.
b. Masks: contains masks for central Argentina (ARG), Southeast Australia (AUS), California (CAL), India (IND), South Africa (SA), and Western Europe (WEU). All masks are based on (combinations of) river basins.
