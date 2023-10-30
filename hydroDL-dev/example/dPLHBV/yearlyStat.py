import sys
sys.path.append('../../')
import os
# os.environ['OSR_USE_TRADITIONAL_GDAL_AXIS_MAPPING'] = 'YES'

# from hydroDL.post import plot, stat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx


# from hydroDL.post import plot, stat

lossfactor = 23
smoothfactor = 0
# lossfactor =2
pred = np.load('/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss23smooth0/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/pred50.npy', allow_pickle=True)
pred_lstm = np.load("/data/kas7897/dPLHBVrelease/output/rnnStreamflow/CAMELSDemo-daymet/TestRun/epochs300_batch100_rho365_hiddensize256_Tstart19801001_Tend19951001/pred.npy", allow_pickle=True)
pred_dpl = np.load('/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/daymet/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/pred50.npy', allow_pickle=True)
pred_kz = pd.read_csv('/data/kas7897/multiple_forcing-kratzert/kratzert_10/qsim_avg.csv').to_numpy()
# pred = np.load("/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/maurer_extendedwithloss0smooth0/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/pred50.npy", allow_pickle=True)
# pred_2 = np.load("/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all1PET_extended_withloss200smooth0/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/pred50.npy", allow_pickle=True)
# obs = np.load('/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/maurer_extendedwithloss0smooth0/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/obs.npy', allow_pickle=True)
obs = np.load('/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss200smooth0/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/obs.npy', allow_pickle=True)
crd = pd.read_csv("/data/kas7897/dPLHBVrelease/hydroDL-dev/gages_list.csv")
eva0 = np.load(f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/'
               f'allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/Eva50.npy', allow_pickle=True)
daymet_eva = np.load(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/daymet/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/Eva50.npy', allow_pickle=True)
# stats_kratzert = stat.statError(pred_kratzert.to_numpy(), obs.squeeze())
# gages_select = ['02177000', '14301000', '12145500']
# gages_select = ['02177000', '01667500', '12145500']
# gages_select = ['02177000', '09065500', '12145500']
# gages_select = ['02177000', '09510200', '12145500']
# gages_select = ['01485500', '09447800', '12145500']
# gages_select = ['06879650', '07145700', '08050800']
# gages_select = ['09447800', '08050800', '07335700']
# gages_select = ['09447800', '08050800', '14400000']
# gages_select = ['09484600', '02055100', '08050800'] #ind: 510; 106, 445
# gages_select = ['09447800', '02055100', '08050800']
# gages_select = ['01583500', '02055100', '08050800']
# gages_select = [crd['gage'][240], crd['gage'][587], crd['gage'][469]]
gages_select = [crd['gage'][240], crd['gage'][587], crd['gage'][477]]

start_date = '1995-10-01'
end_date = '2005-09-30'
num_days = 60
# num_days = 3652
date = pd.date_range(start=start_date, end=end_date)
serial = ['A', 'B', 'C']
# date_ind = date.get_loc('1997-05-01')
date_ind = date.get_loc('2001-06-01')


# wghts_sdate = '1980-10-01'
# lossfactor=23
# smoothfactor=0
# w0 = pd.read_csv(
#     f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts1.csv',
#     header=None)
# wghts_days = len(w0)
# date_range_wghts = pd.date_range(start=wghts_sdate, periods=wghts_days)
# ind_wght =date_range_wghts.get_loc('1996-01-01')
#
# w0['dates'] = date_range_wghts
# w0['Day'] = w0['dates'].dt.day
# w0['Month'] = w0['dates'].dt.month
# w0['Year'] = w0['dates'].dt.year
# w0.drop(columns=['dates'], inplace=True)
#
#
# w1 = pd.read_csv(
#     f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts2.csv',
#     header=None)
# w1['dates'] = date_range_wghts
# w1['Day'] = w1['dates'].dt.day
# w1['Month'] = w1['dates'].dt.month
# w1['Year'] = w1['dates'].dt.year
# w1.drop(columns=['dates'], inplace=True)
#
#
# w2 = pd.read_csv(
#     f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts3.csv',
#     header=None)
#
# w2['dates'] = date_range_wghts
# w2['Day'] = w2['dates'].dt.day
# w2['Month'] = w2['dates'].dt.month
# w2['Year'] = w2['dates'].dt.year
# w2.drop(columns=['dates'], inplace=True)


fig, axs = plt.subplots(4, 1, figsize=(10, 24), gridspec_kw={'height_ratios': [1, 1, 1, 2]})
fig.subplots_adjust(hspace=0.5)
ind = crd[crd['gage']==int(gages_select[0])].index
huc = crd.loc[crd['gage'] == int(gages_select[0]), 'huc'].values[0]
if len(str(huc)) == 1:
    huc = '0' + str(huc)
if len(str(gages_select[0])) == 7:
    gages_select[0] = '0' + str(gages_select[0])
# prcp_daymet_dir = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/daymet/{huc}/{gages_select[0]}_lump_cida_forcing_leap.txt"
# prcp_maurer_dir = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/maurer_extended/{huc}/{gages_select[0]}_lump_maurer_forcing_leap.txt"
# prcp_nldas_dir = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/nldas_extended/{huc}/{gages_select[0]}_lump_nldas_forcing_leap.txt"
# prcp_daymet = pd.read_csv(prcp_daymet_dir, sep=r'\s+', header=None, skiprows=4)
# prcp_maurer = pd.read_csv(prcp_maurer_dir, sep=r'\s+', header=None, skiprows=4)
# prcp_nldas = pd.read_csv(prcp_nldas_dir, sep=r'\s+', header=None, skiprows=4)
# prcp_daymet.rename(columns={0:'Year', 1:'Month', 2:'Day', 5:'precip_daymet'}, inplace=True)
# prcp_maurer.rename(columns={0:'Year', 1:'Month', 2:'Day', 5:'precip_maurer'}, inplace=True)
# prcp_nldas.rename(columns={0:'Year', 1:'Month', 2:'Day', 5:'precip_nldas'}, inplace=True)
# prcp_daymet.drop(columns=[3, 4, 6, 7, 8, 9, 10], inplace=True)
# prcp_maurer.drop(columns=[3, 4, 6, 7, 8, 9, 10], inplace=True)
# prcp_nldas.drop(columns=[3, 4, 6, 7, 8, 9, 10], inplace=True)
# w0_station = w0[[ind[0], 'Year', 'Month', 'Day']]
# w1_station = w1[[ind[0], 'Year', 'Month', 'Day']]
# w2_station = w2[[ind[0], 'Year', 'Month', 'Day']]
# prcp_daymet = pd.merge(prcp_daymet, w0_station, on=['Year', 'Month', 'Day'], how='inner')
# prcp_maurer = pd.merge(prcp_maurer, w1_station, on=['Year', 'Month', 'Day'], how='inner')
# prcp_nldas = pd.merge(prcp_nldas, w2_station, on=['Year', 'Month', 'Day'], how='inner')
# prcp_daymet = prcp_daymet['precip_daymet'][ind_wght:ind_wght+num_days]
# w0_select = w0[ind[0]][ind_wght:ind_wght+num_days]
# prcp_maurer = prcp_maurer['precip_maurer'][ind_wght:ind_wght+num_days]
# w1_select = w1[ind[0]][ind_wght:ind_wght+num_days]
# prcp_nldas = prcp_nldas['precip_nldas'][ind_wght:ind_wght+num_days]
# w2_select = w2[ind[0]][ind_wght:ind_wght+num_days]
# fused = (w0_select*prcp_daymet) + (w1_select*prcp_maurer) + (w2_select*prcp_nldas)

pred_gage = pred[ind[0], :, 0]
pred_kz_gage = pred_kz[ind[0]]
pred_lstm_gage = pred_lstm[0,ind[0], :, 0]
pred_dpl_gage = pred_dpl[ind[0], :, 0]
obs_gage = obs[ind[0], :, 0]
pred_gage_select = pred_gage[date_ind:date_ind+num_days]
pred_kz_select = pred_kz_gage[date_ind:date_ind+num_days]
pred_lstm_gage_select = pred_lstm_gage[date_ind:date_ind+num_days]
pred_dpl_gage_select = pred_dpl_gage[date_ind:date_ind+num_days]
obs_gage_select = obs_gage[date_ind:date_ind+num_days]

ax1 = axs[0]
# ax2 = ax1.twinx()
# ax1.set_xlabel('Date', fontsize=10)
# ax1.set_ylabel('Predicted Steamflow (mm/day)', fontsize=10, color='red')
ax1.tick_params(axis="y")

ax1.plot(date[date_ind:date_ind+num_days], pred_gage_select, label = f'Multiple Precipitation dPLHBV', linestyle='dashed', color = 'red')
ax1.plot(date[date_ind:date_ind+num_days], pred_dpl_gage_select, label = f'dPLHBV (Daymet Forced)', linestyle='dashed', color = 'blue')
ax1.plot(date[date_ind:date_ind+num_days], pred_kz_select, label = f'Kratzert et al. Multiforcing LSTM', linestyle='dashed', color = 'green')
ax1.plot(date[date_ind:date_ind+num_days], obs_gage_select, label = f'Observed', linestyle='solid', color = 'black')
ax1.tick_params(axis='x', rotation =45)

# ax1.plot(date[date_ind:date_ind+num_days], prcp_daymet, label = f'Daymet(std = {np.std(prcp_daymet):.3f}, mean={np.mean(prcp_daymet):.3f}', linestyle='dashed', color = 'red')
# ax1.plot(date[date_ind:date_ind+num_days], prcp_nldas, label = f'NLDAS(std = {np.std(prcp_nldas):.3f}, mean={np.mean(prcp_nldas):.3f}', linestyle='dashed', color = 'blue')
# ax1.plot(date[date_ind:date_ind+num_days], prcp_maurer, label = f'Maurer(std = {np.std(prcp_maurer):.3f}, mean={np.mean(prcp_maurer):.3f}', linestyle='dashed', color = 'yellow')
# ax1.plot(date[date_ind:date_ind+num_days], fused, label = f'Fused(std = {np.std(fused):.3f}, mean={np.mean(fused):.3f}', color='green')
# ax2.plot(date[date_ind:date_ind+num_days], pred_gage_select, label = 'Predicted Streamflow (Fused)', color='magenta')
# ax2.tick_params(axis="y", labelcolor='magenta')

axs[0].set_title(f"(A){gages_select[0]}", fontsize=15)
axs[0].legend(loc='upper left', fontsize='small')
# axs[0].legend(loc='upper left', fontsize='small')



ind = crd[crd['gage']==int(gages_select[1])].index
huc = crd.loc[crd['gage'] == int(gages_select[1]), 'huc'].values[0]
if len(str(huc)) == 1:
    huc = '0' + str(huc)
if len(str(gages_select[1])) == 7:
    gages_select[1] = '0' + str(gages_select[1])
# prcp_daymet_dir = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/daymet/{huc}/{gages_select[1]}_lump_cida_forcing_leap.txt"
# prcp_maurer_dir = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/maurer_extended/{huc}/{gages_select[1]}_lump_maurer_forcing_leap.txt"
# prcp_nldas_dir = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/nldas_extended/{huc}/{gages_select[1]}_lump_nldas_forcing_leap.txt"
# prcp_daymet = pd.read_csv(prcp_daymet_dir, sep=r'\s+', header=None, skiprows=4)
# prcp_maurer = pd.read_csv(prcp_maurer_dir, sep=r'\s+', header=None, skiprows=4)
# prcp_nldas = pd.read_csv(prcp_nldas_dir, sep=r'\s+', header=None, skiprows=4)
# prcp_daymet.rename(columns={0:'Year', 1:'Month', 2:'Day', 5:'precip_daymet'}, inplace=True)
# prcp_maurer.rename(columns={0:'Year', 1:'Month', 2:'Day', 5:'precip_maurer'}, inplace=True)
# prcp_nldas.rename(columns={0:'Year', 1:'Month', 2:'Day', 5:'precip_nldas'}, inplace=True)
# prcp_daymet.drop(columns=[3, 4, 6, 7, 8, 9, 10], inplace=True)
# prcp_maurer.drop(columns=[3, 4, 6, 7, 8, 9, 10], inplace=True)
# prcp_nldas.drop(columns=[3, 4, 6, 7, 8, 9, 10], inplace=True)
# w0_station = w0[[ind[0], 'Year', 'Month', 'Day']]
# w1_station = w1[[ind[0], 'Year', 'Month', 'Day']]
# w2_station = w2[[ind[0], 'Year', 'Month', 'Day']]
# prcp_daymet = pd.merge(prcp_daymet, w0_station, on=['Year', 'Month', 'Day'], how='inner')
# prcp_maurer = pd.merge(prcp_maurer, w1_station, on=['Year', 'Month', 'Day'], how='inner')
# prcp_nldas = pd.merge(prcp_nldas, w2_station, on=['Year', 'Month', 'Day'], how='inner')
# prcp_daymet = prcp_daymet['precip_daymet'][ind_wght:ind_wght+num_days]
# w0_select = w0[ind[0]][ind_wght:ind_wght+num_days]
# prcp_maurer = prcp_maurer['precip_maurer'][ind_wght:ind_wght+num_days]
# w1_select = w1[ind[0]][ind_wght:ind_wght+num_days]
# prcp_nldas = prcp_nldas['precip_nldas'][ind_wght:ind_wght+num_days]
# w2_select = w2[ind[0]][ind_wght:ind_wght+num_days]
# fused = (w0_select*prcp_daymet) + (w1_select*prcp_maurer) + (w2_select*prcp_nldas)

pred_gage = pred[ind[0], :, 0]
pred_kz_gage = pred_kz[ind[0]]
pred_kz_select = pred_kz_gage[date_ind:date_ind+num_days]
pred_lstm_gage = pred_lstm[0,ind[0], :, 0]
pred_dpl_gage = pred_dpl[ind[0], :, 0]
obs_gage = obs[ind[0], :, 0]
pred_gage_select = pred_gage[date_ind:date_ind+num_days]
pred_lstm_gage_select = pred_lstm_gage[date_ind:date_ind+num_days]
pred_dpl_gage_select = pred_dpl_gage[date_ind:date_ind+num_days]
obs_gage_select = obs_gage[date_ind:date_ind+num_days]
ax1 = axs[1]
# ax2 = ax1.twinx()
# ax1.set_xlabel('Date', fontsize=10)
ax1.set_ylabel('Predicted Steamflow (mm/day)', fontsize=15)
ax1.tick_params(axis="y")

# ax1.plot(date[date_ind:date_ind+num_days], prcp_daymet, label = f'Daymet(std = {np.std(prcp_daymet):.3f}, mean={np.mean(prcp_daymet):.3f}', linestyle='dashed', color = 'red')
ax1.plot(date[date_ind:date_ind+num_days], pred_gage_select, label = f'Multiple Precipitation dPLHBV', linestyle='dashed', color = 'red')
# ax1.plot(date[date_ind:date_ind+num_days], prcp_nldas, label = f'NLDAS(std = {np.std(prcp_nldas):.3f}, mean={np.mean(prcp_nldas):.3f}', linestyle='dashed', color = 'blue')
ax1.plot(date[date_ind:date_ind+num_days], pred_dpl_gage_select, label = f'dPLHBV (Daymet Forced)', linestyle='dashed', color = 'blue')
# ax1.plot(date[date_ind:date_ind+num_days], prcp_maurer, label = f'Maurer(std = {np.std(prcp_maurer):.3f}, mean={np.mean(prcp_maurer):.3f}', linestyle='dashed', color = 'yellow')
ax1.plot(date[date_ind:date_ind+num_days], pred_kz_select, label = f'Kratzert et al. Multiforcing LSTM', linestyle='dashed', color = 'green')
ax1.plot(date[date_ind:date_ind+num_days], obs_gage_select, label = f'Observed', linestyle='solid', color = 'black')
ax1.tick_params(axis='x', rotation =45)

# ax2.set_ylabel('Predicted Steamflow - from Fusion(mm/day)', fontsize=14, color='magenta')
# ax1.plot(date[date_ind:date_ind+num_days], fused, label = f'Fused(std = {np.std(fused):.3f}, mean={np.mean(fused):.3f}', color='green')
# ax2.plot(date[date_ind:date_ind+num_days], pred_gage_select, color='magenta')
# ax2.tick_params(axis="y", labelcolor='magenta')

axs[1].set_title(f"(B){gages_select[1]}", fontsize=15)
# axs[1].legend(loc='upper left', fontsize='small')

ind = crd[crd['gage']==int(gages_select[2])].index
huc = crd.loc[crd['gage'] == int(gages_select[2]), 'huc'].values[0]
if len(str(huc)) == 1:
    huc = '0' + str(huc)
if len(str(gages_select[2])) == 7:
    gages_select[2] = '0' + str(gages_select[2])
# prcp_daymet_dir = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/daymet/{huc}/{gages_select[2]}_lump_cida_forcing_leap.txt"
# prcp_maurer_dir = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/maurer_extended/{huc}/{gages_select[2]}_lump_maurer_forcing_leap.txt"
# prcp_nldas_dir = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/nldas_extended/{huc}/{gages_select[2]}_lump_nldas_forcing_leap.txt"
# prcp_daymet = pd.read_csv(prcp_daymet_dir, sep=r'\s+', header=None, skiprows=4)
# prcp_maurer = pd.read_csv(prcp_maurer_dir, sep=r'\s+', header=None, skiprows=4)
# prcp_nldas = pd.read_csv(prcp_nldas_dir, sep=r'\s+', header=None, skiprows=4)
# prcp_daymet.rename(columns={0:'Year', 1:'Month', 2:'Day', 5:'precip_daymet'}, inplace=True)
# prcp_maurer.rename(columns={0:'Year', 1:'Month', 2:'Day', 5:'precip_maurer'}, inplace=True)
# prcp_nldas.rename(columns={0:'Year', 1:'Month', 2:'Day', 5:'precip_nldas'}, inplace=True)
# prcp_daymet.drop(columns=[3, 4, 6, 7, 8, 9, 10], inplace=True)
# prcp_maurer.drop(columns=[3, 4, 6, 7, 8, 9, 10], inplace=True)
# prcp_nldas.drop(columns=[3, 4, 6, 7, 8, 9, 10], inplace=True)
# w0_station = w0[[ind[0], 'Year', 'Month', 'Day']]
# w1_station = w1[[ind[0], 'Year', 'Month', 'Day']]
# w2_station = w2[[ind[0], 'Year', 'Month', 'Day']]
# prcp_daymet = pd.merge(prcp_daymet, w0_station, on=['Year', 'Month', 'Day'], how='inner')
# prcp_maurer = pd.merge(prcp_maurer, w1_station, on=['Year', 'Month', 'Day'], how='inner')
# prcp_nldas = pd.merge(prcp_nldas, w2_station, on=['Year', 'Month', 'Day'], how='inner')
# prcp_daymet = prcp_daymet['precip_daymet'][ind_wght:ind_wght+num_days]
# w0_select = w0[ind[0]][ind_wght:ind_wght+num_days]
# prcp_maurer = prcp_maurer['precip_maurer'][ind_wght:ind_wght+num_days]
# w1_select = w1[ind[0]][ind_wght:ind_wght+num_days]
# prcp_nldas = prcp_nldas['precip_nldas'][ind_wght:ind_wght+num_days]
# w2_select = w2[ind[0]][ind_wght:ind_wght+num_days]
# fused = (w0_select*prcp_daymet) + (w1_select*prcp_maurer) + (w2_select*prcp_nldas)

pred_gage = pred[ind[0], :, 0]
pred_kz_gage = pred_kz[ind[0]]
pred_kz_select = pred_kz_gage[date_ind:date_ind+num_days]
pred_lstm_gage = pred_lstm[0,ind[0], :, 0]
pred_dpl_gage = pred_dpl[ind[0], :, 0]
obs_gage = obs[ind[0], :, 0]
pred_gage_select = pred_gage[date_ind:date_ind+num_days]
pred_lstm_gage_select = pred_lstm_gage[date_ind:date_ind+num_days]
pred_dpl_gage_select = pred_dpl_gage[date_ind:date_ind+num_days]
obs_gage_select = obs_gage[date_ind:date_ind+num_days]
ax1 = axs[2]
# ax2 = ax1.twinx()
# ax1.set_xlabel('Date', fontsize=10)
# ax1.set_ylabel('Predicted Steamflow(mm/day)', fontsize=10, color='red')
ax1.tick_params(axis="y")

ax1.plot(date[date_ind:date_ind+num_days], pred_gage_select, label = f'Multiple Precipitation dPLHBV', linestyle='dashed', color = 'red')
ax1.plot(date[date_ind:date_ind+num_days], pred_dpl_gage_select, label = f'dPLHBV (Daymet Forced)', linestyle='dashed', color = 'blue')
ax1.plot(date[date_ind:date_ind+num_days], pred_kz_select, label = f'Kratzert et al. Multiforcing LSTM', linestyle='dashed', color = 'green')
ax1.plot(date[date_ind:date_ind+num_days], obs_gage_select, label = f'Observed', linestyle='solid', color = 'black')

ax1.tick_params(axis='x', rotation =45)
# ax1.plot(date[date_ind:date_ind+num_days], prcp_daymet, label = f'Daymet(std = {np.std(prcp_daymet):.3f}, mean={np.mean(prcp_daymet):.3f}', linestyle='dashed', color = 'red')
# ax1.plot(date[date_ind:date_ind+num_days], prcp_nldas, label = f'NLDAS(std = {np.std(prcp_nldas):.3f}, mean={np.mean(prcp_nldas):.3f}', linestyle='dashed', color = 'blue')
# ax1.plot(date[date_ind:date_ind+num_days], prcp_maurer, label = f'Maurer(std = {np.std(prcp_maurer):.3f}, mean={np.mean(prcp_maurer):.3f}', linestyle='dashed', color = 'yellow')
# # ax2.set_ylabel('Observed Steamflow(mm/day)', fontsize=10, color='green')
# ax1.plot(date[date_ind:date_ind+num_days], fused, label = f'Fused(std = {np.std(fused):.3f}, mean={np.mean(fused):.3f}', color='green')
# ax2.plot(date[date_ind:date_ind+num_days], pred_gage_select, color='magenta')
# ax2.tick_params(axis="y", labelcolor='magenta')
# ax2.legend(loc='upper left')
axs[2].set_title(f"(C){gages_select[2]} ", fontsize=15)
# axs[2].legend(loc='upper left', fontsize='small')

# fig.autofmt_xdate()



coords = {'A': (crd[crd['gage']==int(gages_select[0])][['LONG']].values[0][0], crd[crd['gage']==int(gages_select[0])][['LAT']].values[0][0]),
          'B': (crd[crd['gage']==int(gages_select[1])][['LONG']].values[0][0], crd[crd['gage']==int(gages_select[1])][['LAT']].values[0][0]),
          'C':(crd[crd['gage']==int(gages_select[2])][['LONG']].values[0][0], crd[crd['gage']==int(gages_select[2])][['LAT']].values[0][0])}

geometry = [Point(lon, lat) for label, (lon, lat) in coords.items()]
coords_gdf = gpd.GeoDataFrame(list(coords.keys()), geometry=geometry, columns=['Label'])
# Set the initial CRS if it isn't set already
if coords_gdf.crs is None:
    coords_gdf.set_crs(epsg=4326, inplace=True)
us_map = gpd.read_file("/data/kas7897/dPLHBVrelease/hydroDL-dev/example/dPLHBV/tl_2022_us_state/tl_2022_us_state.shp")
non_continental = ['HI','VI','MP','GU','AK','AS','PR']
for n in non_continental:
    us_map = us_map[us_map.STUSPS != n]
# us_map = us_map[us_map['name'] == 'United States of America']
us_map = us_map.to_crs(epsg=3857)
coords_gdf = coords_gdf.to_crs(epsg=3857)
us_map.plot(ax=axs[3], linewidth=1, edgecolor='grey', alpha=0.0)
coords_gdf.plot(ax=axs[3], color='red', markersize=50, label='Coordinates')
for label, lon, lat in zip(coords.keys(), coords_gdf.geometry.x, coords_gdf.geometry.y):
    axs[3].annotate(label, xy=(lon, lat), xytext=(3, 3), textcoords='offset points', color='black', fontsize=12)


ctx.add_basemap(axs[3], source=ctx.providers.Stamen.Terrain, attribution="")
axs[3].set_xticks([])
axs[3].set_yticks([])
axs[3].set_title('(D) Basin Locations', fontsize=15)
# plt.tight_layout()
plt.savefig('/home/kas7897/final_plots_fusion_paper/sf_timeseries.png',  dpi=300, bbox_inches='tight')

plt.show()

# for label, (lon, lat) in coords.items():
    # axs[3].text(lon + 2, lat, label, fontsize=12, color='black')
    # axs[3].plot(lon, lat, marker='o', color='red', markersize=8, label=label)
    # axs[3].text(lon, lat, label, fontsize=12, ha='right', va='bottom')
# Set plot title and legend
# axs[3].legend()

# Show the plot





# plt.show()


# pred = pred[:,:,0]
# # pred2 = pred_2[:,:,0]
# obs = obs[:,:,0]
# # obs2 = obs_2[:,:,0]
#
# evaDict_og = [stat.statError(pred, obs.squeeze())]
# flv = np.nanmedian(evaDict_og[0]['FLV'])
# fhv = np.nanmedian(evaDict_og[0]['FHV'])
# corr = np.nanmedian(evaDict_og[0]['Corr'])
# rmse = np.nanmedian(evaDict_og[0]['RMSE'])
# print(flv, fhv, corr, rmse)
#
#
# dates = pd.date_range(start='1995-10-01', periods=obs.shape[1], freq='D')
# pred_df_1 = pd.DataFrame(pred, columns=dates).T
# obs_df_1 = pd.DataFrame(obs, columns=dates).T
#
#
# #yearly max
# pred_yearly_max = np.array(pred_df_1.resample('Y').max().T)
# obs_yearly_max = np.array(obs_df_1.resample('Y').max().T)
#
# evaDict = [stat.statError(pred_yearly_max, obs_yearly_max.squeeze())]
# nse_hbv = evaDict[0]['NSE']
# rmse_hbv = evaDict[0]['RMSE']
# # obs_mean = np.mean(obs_yearly_max, axis=1)
# # obs_mean = obs_mean[:, np.newaxis]
# # ones = np.ones(671)
# # NSE = ones - (np.sum((pred_yearly_max - obs_yearly_max)**2, axis=1) / np.sum((obs_yearly_max - obs_mean)**2, axis=1))
# # NSE_d = ones - (np.sum(abs(pred - obs), axis=1) / np.sum(abs(obs - obs_mean), axis=1))
# print(np.nanmedian(nse_hbv), np.nanmedian(rmse_hbv))
#
# # pred_lstm = pd.read_csv("/data/kas7897/dPLHBVrelease/output/rnnStreamflow/CAMELSDemo-Multiforcing-All/TestRun/epochs300_batch100_rho365_hiddensize256_Tstart19801001_Tend19951001/All-85-95/All_19951001_20051001_ep300_Streamflow.csv")
# pred_lstm = np.load("/data/kas7897/dPLHBVrelease/output/rnnStreamflow/CAMELSDemo-Multiforcing/TestRun/epochs300_batch100_rho365_hiddensize256_Tstart19801001_Tend19951001/pred.npy", allow_pickle=True)[0]
# obs_lstm = np.load("/data/kas7897/dPLHBVrelease/output/rnnStreamflow/CAMELSDemo-Multiforcing/TestRun/epochs300_batch100_rho365_hiddensize256_Tstart19801001_Tend19951001/obs.npy", allow_pickle=True)
# pred_lstm = pred_lstm[:,:,0]
# obs_lstm = obs_lstm[:,:,0]
#
# evaDict_og = [stat.statError(pred_lstm, obs_lstm.squeeze())]
# flv = np.nanmedian(evaDict_og[0]['FLV'])
# fhv = np.nanmedian(evaDict_og[0]['FHV'])
# corr = np.nanmedian(evaDict_og[0]['Corr'])
# rmse = np.nanmedian(evaDict_og[0]['RMSE'])
# print(flv, fhv, corr, rmse)
#
# dates = pd.date_range(start='1995-10-01', periods=obs.shape[1], freq='D')
# pred_df = pd.DataFrame(pred_lstm, columns=dates).T
# obs_df = pd.DataFrame(obs_lstm, columns=dates).T
#
#
# #yearly max
# pred_yearly_max = np.array(pred_df.resample('Y').max().T)
# obs_yearly_max = np.array(obs_df.resample('Y').max().T)
# evaDict = [stat.statError(pred_yearly_max, obs_yearly_max.squeeze())]
# nse_lstm = evaDict[0]['NSE']
# rmse_lstm = evaDict[0]['RMSE']
# # obs_mean = np.mean(obs_yearly_max, axis=1)
# # obs_mean = obs_mean[:, np.newaxis]
# #
# # ones = np.ones(671)
# # NSE = ones - (np.sum((pred_yearly_max - obs_yearly_max)**2, axis=1) / np.sum((obs_yearly_max- obs_mean)**2, axis=1))
# print(np.nanmedian(nse_lstm), np.nanmedian(rmse_lstm))
# k=1