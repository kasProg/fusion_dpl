import numpy as np
import pandas as pd
# import geopandas as gpd
from mpl_toolkits.basemap import Basemap, addcyclic
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf
import os
import json
lossfactor = 200
smoothfactor = 0
# expno =4
crd = pd.read_csv("/data/kas7897/dPLHBVrelease/hydroDL-dev/gages_list.csv")
rm_window = 1
obs_sf = np.load('/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/'
                         'allprcp_withloss23smooth0/BuffOpt0/RMSE_para0.25/111111/'
                         'Train19801001_19951001Test19951001_20051001Buff5478Staind5477/obs.npy', allow_pickle=True)

inputsLst = np.array(['prcp_daymet', 'prcp_maurer','prcp_nldas', 'temp_daymet', 'pet_daymet', 'p_mean','pet_mean',
                      'p_seasonality','frac_snow','aridity','high_prec_freq','high_prec_dur',
                   'low_prec_freq','low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
                   'lai_diff', 'gvf_max', 'gvf_diff', 'dom_land_cover_frac', 'dom_land_cover', 'root_depth_50',
                   'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
                   'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'geol_1st_class', 'glim_1st_class_frac',
                   'geol_2nd_class', 'glim_2nd_class_frac', 'carbonate_rocks_frac', 'geol_porostiy', 'geol_permeability'])
# inputsLst = np.array(['prcp_daymet', 'prcp_maurer','prcp_nldas', 'temp_daymet', 'pet_daymet', 'p_mean', 'pet_mean',
#                       'p_seasonality', 'frac_snow', 'aridity', 'high_prec_freq', 'high_prec_dur',
#                  'low_prec_freq', 'low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
#                  'lai_diff', 'gvf_max', 'gvf_diff'])
# inputsLst = np.array(['frac_snow','aridity', 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
#                'lai_diff', 'gvf_max', 'gvf_diff'])
index1= np.where(inputsLst=='slope_mean')[0][0]
index2= np.where(inputsLst=='elev_mean')[0][0]
wghts_sdate = '1980-10-01'
w0_3 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts1.csv',
    header=None)
w0_1 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts1.csv',
    header=None)
w0_2 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts1.csv',
    header=None)
wghts_days = len(w0_3)
date_range_wghts = pd.date_range(start=wghts_sdate, periods=wghts_days)
w0_3['dates'] = date_range_wghts
w0_1['dates'] = date_range_wghts
w0_2['dates'] = date_range_wghts
w0_3['Day'] = w0_3['dates'].dt.day
w0_1['Day'] = w0_1['dates'].dt.day
w0_2['Day'] = w0_2['dates'].dt.day
w0_3['Mnth'] = w0_3['dates'].dt.month
w0_1['Mnth'] = w0_1['dates'].dt.month
w0_2['Mnth'] = w0_2['dates'].dt.month
w0_3['Year'] = w0_3['dates'].dt.year
w0_2['Year'] = w0_2['dates'].dt.year
w0_1['Year'] = w0_1['dates'].dt.year
w0_3.drop(columns=['dates'], inplace=True)
w0_2.drop(columns=['dates'], inplace=True)
w0_1.drop(columns=['dates'], inplace=True)

w1_3 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts2.csv',
    header=None)
w1_2 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts2.csv',
    header=None)
w1_1 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts2.csv',
    header=None)
w1_3['dates'] = date_range_wghts
w1_2['dates'] = date_range_wghts
w1_1['dates'] = date_range_wghts
w1_3['Day'] = w1_3['dates'].dt.day
w1_2['Day'] = w1_2['dates'].dt.day
w1_1['Day'] = w1_1['dates'].dt.day
w1_3['Mnth'] = w1_3['dates'].dt.month
w1_2['Mnth'] = w1_2['dates'].dt.month
w1_1['Mnth'] = w1_1['dates'].dt.month
w1_3['Year'] = w1_3['dates'].dt.year
w1_2['Year'] = w1_2['dates'].dt.year
w1_1['Year'] = w1_1['dates'].dt.year
w1_3.drop(columns=['dates'], inplace=True)
w1_2.drop(columns=['dates'], inplace=True)
w1_1.drop(columns=['dates'], inplace=True)

w2_3 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts3.csv',
    header=None)
w2_2 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts3.csv',
    header=None)
w2_1 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts3.csv',
    header=None)
w2_3['dates'] = date_range_wghts
w2_2['dates'] = date_range_wghts
w2_1['dates'] = date_range_wghts
w2_3['Day'] = w2_3['dates'].dt.day
w2_2['Day'] = w2_2['dates'].dt.day
w2_1['Day'] = w2_1['dates'].dt.day
w2_3['Mnth'] = w2_3['dates'].dt.month
w2_2['Mnth'] = w2_2['dates'].dt.month
w2_1['Mnth'] = w2_1['dates'].dt.month
w2_3['Year'] = w2_3['dates'].dt.year
w2_2['Year'] = w2_2['dates'].dt.year
w2_1['Year'] = w2_1['dates'].dt.year
w2_3.drop(columns=['dates'], inplace=True)
w2_2.drop(columns=['dates'], inplace=True)
w2_1.drop(columns=['dates'], inplace=True)

time_avg_w0_1 = pd.DataFrame(np.mean(w0_1))
time_avg_w0_2 = pd.DataFrame(np.mean(w0_2))
time_avg_w0_3 = pd.DataFrame(np.mean(w0_3))
# time_avg_w0 = pd.DataFrame(np.mean(w0[-3653:]))
time_avg_w1_1 = pd.DataFrame(np.mean(w1_1))
time_avg_w1_2 = pd.DataFrame(np.mean(w1_2))
time_avg_w1_3 = pd.DataFrame(np.mean(w1_3))
# time_avg_w1 = pd.DataFrame(np.mean(w1[-3653:]))
time_avg_w2_1 = pd.DataFrame(np.mean(w2_1))#
time_avg_w2_2 = pd.DataFrame(np.mean(w2_2))#
time_avg_w2_3 = pd.DataFrame(np.mean(w2_3))#
# time_avg_w2 = pd.DataFrame(np.mean(w2[-3563:]))
#
print('DAYMET:', np.mean(time_avg_w0_1))
print('DAYMET:', np.mean(time_avg_w0_2))
print('DAYMET:', np.mean(time_avg_w0_3))
print('MAURER:', np.mean(time_avg_w1_1))
print('MAURER:', np.mean(time_avg_w1_2))
print('MAURER:', np.mean(time_avg_w1_3))
print('NLDAS:', np.mean(time_avg_w2_1))
print('NLDAS:', np.mean(time_avg_w2_2))
print('NLDAS:', np.mean(time_avg_w2_3))
# time_avg_wsum = pd.DataFrame(np.mean(w0+w1+w2))
# time_avg_w0.rename(columns={0: 'w0'}, inplace=True)
# time_avg_w1.rename(columns={0: 'w1'}, inplace=True)
# time_avg_w2.rename(columns={0: 'w2'}, inplace=True)

std_model1 = np.empty(len(crd))
std_model2 = np.empty(len(crd))
std_model3 = np.empty(len(crd))
std_model_daymet = np.empty(len(crd))
std_model_maurer = np.empty(len(crd))
std_model_nldas = np.empty(len(crd))
mean_model1 = np.empty(len(crd))
mean_model2 = np.empty(len(crd))
mean_model3 = np.empty(len(crd))
mean_model_daymet = np.empty(len(crd))
mean_model_maurer = np.empty(len(crd))
mean_model_nldas = np.empty(len(crd))
acf_model1 = np.empty(len(crd))
acf_model2 = np.empty(len(crd))
acf_model3 = np.empty(len(crd))
acf_model_daymet = np.empty(len(crd))
acf_model_maurer = np.empty(len(crd))
acf_model_nldas = np.empty(len(crd))


highsfobs_daymet_mean = np.empty(len(crd))
highsfobs_maurer_mean = np.empty(len(crd))
highsfobs_nldas_mean = np.empty(len(crd))
highsfobs_multi_mean = np.empty(len(crd))


for i  in range(len(crd)):
    gage = crd['gage'][i]
    huc = crd.loc[crd['gage'] == int(gage), 'huc'].values[0]
    if len(str(huc)) == 1:
        huc = '0' + str(huc)
    if len(str(gage)) == 7:
        gage = '0' + str(gage)
    gage = str(gage)
    prcp_daymet_dir = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/daymet/{huc}/{gage}_lump_cida_forcing_leap.txt"
    prcp_maurer_dir = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/maurer_extended/{huc}/{gage}_lump_maurer_forcing_leap.txt"
    prcp_nldas_dir = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/nldas_extended/{huc}/{gage}_lump_nldas_forcing_leap.txt"
    # w0_station_1 = w0_1[[i, 'Year', 'Mnth', 'Day']]
    # w0_station_1.rename(columns={i: 'w0_1'}, inplace=True)
    w0_station_2 = w0_2[[i, 'Year', 'Mnth', 'Day']]
    w0_station_2.rename(columns={i: 'w0'}, inplace=True)
    # w0_station_3 = w0_3[[i, 'Year', 'Mnth', 'Day']]
    # w0_station_3.rename(columns={i: 'w0_3'}, inplace=True)
    # w1_station_1 = w1_1[[i, 'Year', 'Mnth', 'Day']]
    # w1_station_1.rename(columns={i: 'w1_1'}, inplace=True)
    w1_station_2 = w1_2[[i, 'Year', 'Mnth', 'Day']]
    w1_station_2.rename(columns={i: 'w1'}, inplace=True)
    # w1_station_3 = w1_3[[i, 'Year', 'Mnth', 'Day']]
    # w1_station_3.rename(columns={i: 'w1_3'}, inplace=True)
    # w2_station_1 = w2_1[[i, 'Year', 'Mnth', 'Day']]
    # w2_station_1.rename(columns={i: 'w2_1'}, inplace=True)
    w2_station_2 = w2_2[[i, 'Year', 'Mnth', 'Day']]
    w2_station_2.rename(columns={i: 'w2'}, inplace=True)
    # w2_station_3 = w2_3[[i, 'Year', 'Mnth', 'Day']]
    # w2_station_3.rename(columns={i: 'w2_3'}, inplace=True)

    prcp_daymet = pd.read_csv(prcp_daymet_dir, sep=r'\s+', skiprows=3)
    prcp_maurer = pd.read_csv(prcp_maurer_dir, sep=r'\s+', skiprows=3)
    prcp_nldas = pd.read_csv(prcp_nldas_dir, sep=r'\s+', skiprows=3)
    # prcp_daymet.rename(columns={0: 'Year', 1: 'Month', 2: 'Day', 5: 'precip_daymet'}, inplace=True)
    # prcp_maurer.rename(columns={0: 'Year', 1: 'Month', 2: 'Day', 5: 'precip_maurer'}, inplace=True)
    # prcp_nldas.rename(columns={0: 'Year', 1: 'Month', 2: 'Day', 5: 'precip_nldas'}, inplace=True)
    # prcp_daymet.drop(columns=[3, 4, 6, 7, 8, 9, 10], inplace=True)
    # prcp_maurer.drop(columns=[3, 4, 6, 7, 8, 9, 10], inplace=True)
    # prcp_nldas.drop(columns=[3, 4, 6, 7, 8, 9, 10], inplace=True)

    # prcp_daymet = prcp_daymet[np.where(date_range == np.datetime64('1981-01-01'))[0][0]: np.where(date_range == np.datetime64('2005-10-01'))[0][0]][5].reset_index(drop=True)
    # prcp_nldas = prcp_nldas[np.where(date_range == np.datetime64('1981-01-01'))[0][0]: np.where(date_range == np.datetime64('2005-10-01'))[0][0]][5].reset_index(drop=True)
    # prcp_maurer = prcp_maurer[np.where(date_range_maurer == np.datetime64('1981-01-01'))[0][0]: np.where(date_range_maurer == np.datetime64('2005-10-01'))[0][0]][5].reset_index(drop=True)
    # prcp_daymet = pd.merge(prcp_daymet, w0_station_1, on=['Year', 'Mnth', 'Day'], how='inner')
    prcp_daymet = pd.merge(prcp_daymet, w0_station_2, on=['Year', 'Mnth', 'Day'], how='inner')
    # prcp_daymet = pd.merge(prcp_daymet, w0_station_3, on=['Year', 'Mnth', 'Day'], how='inner')
    # prcp_maurer = pd.merge(prcp_maurer, w1_station_1, on=['Year', 'Mnth', 'Day'], how='inner')
    prcp_maurer = pd.merge(prcp_maurer, w1_station_2, on=['Year', 'Mnth', 'Day'], how='inner')
    # prcp_maurer = pd.merge(prcp_maurer, w1_station_3, on=['Year', 'Mnth', 'Day'], how='inner')
    # prcp_nldas = pd.merge(prcp_nldas, w2_station_1, on=['Year', 'Mnth', 'Day'], how='inner')
    prcp_nldas = pd.merge(prcp_nldas, w2_station_2, on=['Year', 'Mnth', 'Day'], how='inner')
    # prcp_nldas = pd.merge(prcp_nldas, w2_station_3, on=['Year', 'Mnth', 'Day'], how='inner')


    # prcp_daymet['wght_prcp_1'] = prcp_daymet['prcp(mm/day)'] * prcp_daymet['w0_1']
    prcp_daymet['wght_prcp'] = prcp_daymet['prcp(mm/day)'] * prcp_daymet['w0']
    # prcp_daymet['wght_prcp_3'] = prcp_daymet['prcp(mm/day)'] * prcp_daymet['w0_3']
    # prcp_maurer['wght_prcp_1'] = prcp_maurer['prcp(mm/day)'] * prcp_maurer['w1_1']
    prcp_maurer['wght_prcp'] = prcp_maurer['prcp(mm/day)'] * prcp_maurer['w1']
    # prcp_maurer['wght_prcp_3'] = prcp_maurer['prcp(mm/day)'] * prcp_maurer['w1_3']
    # prcp_nldas['wght_prcp_1'] = prcp_nldas['PRCP(mm/day)'] * prcp_nldas['w2_1']
    prcp_nldas['wght_prcp'] = prcp_nldas['PRCP(mm/day)'] * prcp_nldas['w2']
    # prcp_nldas['wght_prcp_3'] = prcp_nldas['PRCP(mm/day)'] * prcp_nldas['w2_3']

    # multi_prcp1 = prcp_daymet['wght_prcp_1'] + prcp_maurer['wght_prcp_1'] + prcp_nldas['wght_prcp_1']
    multi_prcp2 = prcp_daymet['wght_prcp'] + prcp_maurer['wght_prcp'] + prcp_nldas['wght_prcp']
    # multi_prcp3 = prcp_daymet['wght_prcp_3'] + prcp_maurer['wght_prcp_3'] + prcp_nldas['wght_prcp_3']

    #creating fused product
    prcp_daymet['prcp(mm/day)'] = multi_prcp2
    prcp_daymet.drop(columns=['w0', 'wght_prcp'], inplace=True)
    with open(prcp_daymet_dir, 'r') as file:
        # save first three lines which are not a part of dataframe
        header = [next(file) for _ in range(3)]

    fused_dir = f'/data/kas7897/dPLHBVrelease/hydroDL-dev/example/fused_prcp/{huc}'
    if not os.path.exists(fused_dir):
        os.makedirs(fused_dir)
    fused_output= os.path.join(fused_dir, f'{gage}_lump_fused_forcing_leap.txt')

    with open(fused_output, 'w') as file:
        # write the headers back
        file.writelines(header)
        # append the dataframe
        prcp_daymet.to_string(file, index=False, float_format="%.2f")

    print(fused_output)


    #
    # mean_model1[i] = np.mean(multi_prcp1)
    # std_model1[i] = np.std(multi_prcp1)
    # acf_model1[i] = acf(multi_prcp1, nlags=1)[1]
    mean_model2[i] = np.mean(multi_prcp2)
    std_model2[i] = np.std(multi_prcp2)
    acf_model2[i] = acf(multi_prcp2, nlags=1)[1]
    # mean_model3[i] = np.mean(multi_prcp3)
    # std_model3[i] = np.std(multi_prcp3)
    # acf_model3[i] = acf(multi_prcp3, nlags=1)[1]

    mean_model_daymet[i] = np.mean(prcp_daymet['prcp(mm/day)'])
    mean_model_maurer[i] = np.mean(prcp_maurer['prcp(mm/day)'])
    mean_model_nldas[i] = np.mean(prcp_nldas['PRCP(mm/day)'])
    std_model_daymet[i] = np.std(prcp_daymet['prcp(mm/day)'])
    std_model_maurer[i] = np.std(prcp_maurer['prcp(mm/day)'])
    std_model_nldas[i] = np.mean(prcp_nldas['PRCP(mm/day)'])
    acf_model_daymet[i] = acf(prcp_daymet['prcp(mm/day)'], nlags=1)[1]
    acf_model_maurer[i] = acf(prcp_maurer['prcp(mm/day)'], nlags=1)[1]
    acf_model_nldas[i] = acf(prcp_nldas['PRCP(mm/day)'], nlags=1)[1]





    # prcp_sf = pd.DataFrame(obs_sf[i,:,0], columns=['obs_sf'])
    # prcp_sf = pd.concat([prcp_sf, multi_prcp2[-len(obs_sf[i,:,0]):].reset_index(drop=True),
    #                      prcp_daymet['precip_daymet'][-len(obs_sf[i,:,0]):].reset_index(drop=True),
    #                      prcp_maurer['precip_maurer'][-len(obs_sf[i,:,0]):].reset_index(drop=True),
    #                      prcp_nldas['precip_nldas'][-len(obs_sf[i,:,0]):].reset_index(drop=True)], axis=1)
    # # prcp_sf_sort = prcp_sf.sort_values('obs_sf').reset_index(drop=True)
    # prcp_sf_sort = prcp_sf.sort_values('obs_sf')
    # indexhigh = round(0.98 * len(prcp_sf_sort))
    #
    # high_indices = prcp_sf_sort.index[indexhigh:]
    # buffer_highindices = []
    # for idx in high_indices:
    #     buffer_highindices.extend(range(idx - 5, idx + 1))
    #
    # high_sf_prcp = prcp_sf.loc[sorted(set(buffer_highindices)),:].copy()
    #
    #
    # highsfobs_daymet_mean[i] = np.mean(high_sf_prcp['precip_daymet'])
    # highsfobs_maurer_mean[i] = np.mean(high_sf_prcp['precip_maurer'])
    # highsfobs_nldas_mean[i] = np.mean(high_sf_prcp['precip_nldas'])
    # highsfobs_multi_mean[i] = np.mean(high_sf_prcp['wght_prcp_2'])

    # high_sf_prcp = high_sf_prcp[high_sf_prcp['precip_nldas'] != 0]
    # highsfobs_daymet_mean[i] = np.nanmean((high_sf_prcp['precip_daymet'] - high_sf_prcp['precip_nldas'])*100/ high_sf_prcp['precip_daymet'])
    # highsfobs_maurer_mean[i] = np.nanmean((high_sf_prcp['precip_maurer'] - high_sf_prcp['precip_nldas'])*100/ high_sf_prcp['precip_daymet'])
    # highsfobs_nldas_mean[i] = np.nanmean((high_sf_prcp['precip_nldas']- high_sf_prcp['precip_nldas'])*100/ high_sf_prcp['precip_daymet'])
    # highsfobs_multi_mean[i] = np.nanmean((high_sf_prcp['wght_prcp_2']- high_sf_prcp['precip_nldas'])*100/ high_sf_prcp['precip_daymet'])

    # sort_sf_pred = np.sort(pred_sf[i,:,0])
    # indexhigh = round(0.98 * len(sort_sf_pred))
    # highsf_pred[i] = np.mean(sort_sf_pred[indexhigh:])


    k=1


fig, axs = plt.subplots(3, 3, figsize=(20, 16))
# cbar_axs = [fig.add_axes([0.1, 0.1, 0.8, 0.02]) for _ in range(4)]


m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
            projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
            resolution='c', area_thresh=10000)


axs[0,0].set_title(f'(a) Mean Precipitation - Fusion 1.0', size=17)
m.ax = axs[0, 0]
m.drawmapboundary()
# m.shadedrelief(scale=0.5)
m.etopo()
# m.fillcontinents(color='#524c4c', lake_color='#46bcec')
# m.fillcontinents(color='0.1')
m.drawstates()
m.drawcoastlines()
x, y = m(crd['LONG'].values, crd['LAT'].values)
# scatter = m.scatter(x, y, c=time_avg_w0['w0'], cmap='seismic', alpha=0.9, zorder=2)
scatter = m.scatter(x, y, c=mean_model1, cmap='Reds', alpha=0.9, zorder=2)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cax1 = fig.add_axes([0.13, 0.52, 0.35, 0.02])
cax1= fig.add_axes([axs[0, 0].get_position().x0, axs[0, 0].get_position().y0 - 0.02, axs[0, 0].get_position().width, 0.02])
cbar = plt.colorbar(scatter, cax=cax1, shrink=0.9, orientation='horizontal', pad=0.05)
# cbar.set_label(f'w0(daymet), loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
cbar.ax.tick_params(labelsize=14)
# plt.title(f'(a) w\u2080 (weight associated to Daymet)', size=12)



axs[0,1].set_title(f'(b) Mean Precipitation - Fusion 2.0 (Daymet)', size=17)
m.ax = axs[0, 1]
m.drawmapboundary()
# m.fillcontinents(color='#524c4c', lake_color='#46bcec')
# m.fillcontinents(color='0.1')
# m.shadedrelief(scale=0.5)
m.etopo()
m.drawcoastlines()
m.drawstates()
x, y = m(crd['LONG'].values, crd['LAT'].values)
# scatter = m.scatter(x, y, c=time_avg_w0['w0'], cmap='seismic', alpha=0.9, zorder=2)
scatter = m.scatter(x, y, c=mean_model2, cmap='Reds', alpha=0.9, zorder=2)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cax2 = fig.add_axes([0.13, 0.52, 0.35, 0.02])
cax2 = fig.add_axes([axs[0, 1].get_position().x0, axs[0, 1].get_position().y0 - 0.02, axs[0, 1].get_position().width, 0.02])
cbar = plt.colorbar(scatter, cax=cax2, shrink=0.9, orientation='horizontal', pad=0.05)
# cbar.set_label(f'w0(daymet), loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
cbar.ax.tick_params(labelsize=14)


axs[0,2].set_title(f'(c) Mean Precipitation - Fusion 2.0 (NLDAS)', size=17)
m.ax = axs[0, 2]
m.drawmapboundary()
# m.fillcontinents(color='#524c4c', lake_color='#46bcec')
# m.fillcontinents(color='0.1')
m.shadedrelief(scale=0.5)
m.etopo()
m.drawcoastlines()
m.drawstates()
x, y = m(crd['LONG'].values, crd['LAT'].values)
scatter = m.scatter(x, y, c=mean_model3, cmap='Reds', alpha=0.9, zorder=2)
# scatter = m.scatter(x, y, c=time_avg_slope_w1[0], cmap='seismic', alpha=0.9, zorder=2)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cax3 = fig.add_axes([0.55, 0.52, 0.35, 0.02])
cax3= fig.add_axes([axs[0, 2].get_position().x0, axs[0, 2].get_position().y0 - 0.02, axs[0, 2].get_position().width, 0.02])
cbar = plt.colorbar(scatter, cax=cax3, shrink=0.9, orientation='horizontal', pad=0.05)
# cbar.set_label(f'w1(maurer), loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
cbar.ax.tick_params(labelsize=14)
# plt.title(f'(b) w\u2081 (weight associated to Maurer)', size=12)





axs[1,0].set_title(f'(d) Std. Precipitation - Fusion 1.0', size=17)
m.ax = axs[1, 0]
m.drawmapboundary()
# m.fillcontinents(color='#524c4c', lake_color='#46bcec')
# m.fillcontinents(color='0.1')
# m.shadedrelief(scale=0.5)
m.etopo()
m.drawcoastlines()
m.drawstates()

x, y = m(crd['LONG'].values, crd['LAT'].values)
# scatter = m.scatter(x, y, c=time_avg_w2['w2'], cmap='seismic', alpha=0.9, zorder=2)
scatter = m.scatter(x, y, c=std_model1, cmap='Greens', alpha=0.9, zorder=2, vmin=0.0, vmax=20.0)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cax4 = fig.add_axes([0.13, 0.1, 0.35, 0.02])
cax4= fig.add_axes([axs[1, 0].get_position().x0, axs[1, 0].get_position().y0 - 0.02, axs[1, 0].get_position().width, 0.02])
cbar = plt.colorbar(scatter, cax=cax4, shrink=0.9, orientation='horizontal', pad=0.05)
# cbar.set_label(f'w2(nldas), loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
# plt.title(f'(c) w\u2082 (weight associated to NLDAS2)', size=12)
cbar.ax.tick_params(labelsize=14)

axs[1,1].set_title(f'(e) Std. Precipitation - Fusion 2.0 (Daymet)', size=17)
m.ax = axs[1, 1]
m.drawmapboundary()
# m.fillcontinents(color='#524c4c', lake_color='#46bcec')
# m.shadedrelief(scale=0.5)
m.etopo()
# m.fillcontinents(color='0.1')
m.drawcoastlines()
m.drawstates()
x, y = m(crd['LONG'].values, crd['LAT'].values)
# scatter = m.scatter(x, y, c=time_avg_w0['w0'], cmap='seismic', alpha=0.9, zorder=2)
scatter = m.scatter(x, y, c=std_model2, cmap='Greens', alpha=0.9, zorder=2, vmin=0.0, vmax=20.0)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cax5 = fig.add_axes([0.13, 0.52, 0.35, 0.02])
cax5= fig.add_axes([axs[1, 1].get_position().x0, axs[1, 1].get_position().y0 - 0.02, axs[1, 1].get_position().width, 0.02])
cbar = plt.colorbar(scatter, cax=cax5, shrink=0.9, orientation='horizontal', pad=0.05)
# cbar.set_label(f'w0(daymet), loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
cbar.ax.tick_params(labelsize=14)


axs[1,2].set_title(f'(f) Std. Precipitation - Fusion 2.0 (NLDAS)',size=17)
m.ax = axs[1, 2]
m.drawmapboundary()
# m.fillcontinents(color='#524c4c', lake_color='#46bcec')
# m.shadedrelief(scale=0.5)
m.etopo()
# m.fillcontinents(color='0.1')
m.drawcoastlines()
m.drawstates()
x, y = m(crd['LONG'].values, crd['LAT'].values)
scatter = m.scatter(x, y, c=std_model3, cmap='Greens', alpha=0.9, zorder=2, vmin=0.0, vmax=20.0)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cax6 = fig.add_axes([0.55, 0.1, 0.35, 0.02])
cax6= fig.add_axes([axs[1, 2].get_position().x0, axs[1, 2].get_position().y0 - 0.02, axs[1, 2].get_position().width, 0.02])
cbar = plt.colorbar(scatter, cax=cax6, shrink=0.9, orientation='horizontal', pad=0.05)
# cbar.set_label(f'wsum, loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
# plt.title(f'Sum of Weights (w\u2080 + w\u2081 + w\u2082)')
cbar.ax.tick_params(labelsize=14)

# plt.tight_layout()


axs[2,0].set_title(f'(g) ACF Precipitation - Fusion 1.0', size=17)
m.ax = axs[2, 0]
m.drawmapboundary()
# m.shadedrelief(scale=0.5)
m.etopo()
# m.fillcontinents(color='#524c4c', lake_color='#46bcec')
# m.fillcontinents(color='0.1')
m.drawcoastlines()
m.drawstates()
x, y = m(crd['LONG'].values, crd['LAT'].values)
# scatter = m.scatter(x, y, c=time_avg_w2['w2'], cmap='seismic', alpha=0.9, zorder=2)
scatter = m.scatter(x, y, c=acf_model1, cmap='Blues', alpha=0.9, zorder=2)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cax7 = fig.add_axes([0.13, 0.1, 0.35, 0.02])
cax7= fig.add_axes([axs[2, 0].get_position().x0, axs[2, 0].get_position().y0 - 0.02, axs[2, 0].get_position().width, 0.02])
cbar = plt.colorbar(scatter, cax=cax7, shrink=0.9, orientation='horizontal', pad=0.05)
# cbar.set_label(f'w2(nldas), loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
# plt.title(f'(c) w\u2082 (weight associated to NLDAS2)', size=12)
cbar.ax.tick_params(labelsize=14)


axs[2,1].set_title(f'(h) ACF Precipitation - Fusion 2.0 (Daymet)', size=17)
m.ax = axs[2, 1]
m.drawmapboundary()
# m.fillcontinents(color='#524c4c', lake_color='#46bcec')
# m.shadedrelief(scale=0.5)
m.etopo()
# m.fillcontinents(color='0.1')
m.drawcoastlines()
m.drawstates()
x, y = m(crd['LONG'].values, crd['LAT'].values)
# scatter = m.scatter(x, y, c=time_avg_w0['w0'], cmap='seismic', alpha=0.9, zorder=2)
scatter = m.scatter(x, y, c=acf_model2, cmap='Blues', alpha=0.9, zorder=2)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cax8 = fig.add_axes([0.13, 0.52, 0.35, 0.02])
cax8= fig.add_axes([axs[2, 1].get_position().x0, axs[2, 1].get_position().y0 - 0.02, axs[2, 1].get_position().width, 0.02])
cbar = plt.colorbar(scatter, cax=cax8, shrink=0.9, orientation='horizontal', pad=0.05)
# cbar.set_label(f'w0(daymet), loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
cbar.ax.tick_params(labelsize=14)



axs[2,2].set_title(f'(i) ACF Precipitation - Fusion 2.0 (NLDAS)',size=17)
m.ax = axs[2, 2]
m.drawmapboundary()
# m.fillcontinents(color='#524c4c', lake_color='#46bcec')
# m.fillcontinents(color='0.1')
# m.shadedrelief(scale=0.5)
m.etopo()
m.drawcoastlines()
m.drawstates()
x, y = m(crd['LONG'].values, crd['LAT'].values)
scatter = m.scatter(x, y, c=acf_model3, cmap='Blues', alpha=0.9, zorder=2)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cax9 = fig.add_axes([0.55, 0.1, 0.35, 0.02])
cax9= fig.add_axes([axs[2, 2].get_position().x0, axs[2, 2].get_position().y0 - 0.02, axs[2, 2].get_position().width, 0.02])
cbar = plt.colorbar(scatter, cax=cax9, shrink=0.9, orientation='horizontal', pad=0.05)
# cbar.set_label(f'wsum, loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
# plt.title(f'Sum of Weights (w\u2080 + w\u2081 + w\u2082)')
cbar.ax.tick_params(labelsize=14)

# plt.title(f'loss_factor = {lossfactor}')
plt.savefig(f'/home/kas7897/prcp_stats.png', dpi=300, bbox_inches='tight')

plt.show()

group1 = [mean_model2, mean_model_daymet, mean_model_maurer, mean_model_nldas]
# print("multi:",np.mean(highsfobs_multi_mean))
# print("daymet:",np.mean(highsfobs_daymet_mean))
# print("maurer:",np.mean(highsfobs_maurer_mean))
# print("nldas:",np.mean(highsfobs_nldas_mean))
# group1 = [highsfobs_multi_mean, highsfobs_daymet_mean, highsfobs_maurer_mean, highsfobs_nldas_mean]
group2 = [std_model2, std_model_daymet, std_model_maurer, std_model_nldas]
group3 = [acf_model2, acf_model_daymet, acf_model_maurer, acf_model_nldas]
# data = [group1, group2, group3]
# data = [group1, group2]
#
# fig, ax = plt.subplots()
#
# # This offset is used to separate the groups
# # offsets = [0, 4, 8]
# offsets = [2, 6]
#
# for i, data_group in enumerate(data):
#     # The position for each box within the group
#     positions = [j + offsets[i] for j in range(len(data_group))]
#     ax.boxplot(data_group, positions=positions, widths=0.6)
#
# # Custom x-axis labels
# tick_positions = [j + offset for offset in offsets for j in range(3)]
# tick_labels = [
#     "Mean Fusion 1.0", "Mean Fusion 2.0(Daymet)", "Mean Fusion 2.0(NLDAS)",
#     "Std Fusion 1.0", "Std Fusion 2.0(Daymet)", "Std Fusion 2.0(NLDAS)"
# ]
#
# ax.set_xticks(tick_positions)
# ax.set_xticklabels(tick_labels, rotation=45, ha='right')  # rotated for better readability
#
# ax.set_title("Precipitation Statistics")
# plt.tight_layout()
# plt.show()

fig, ax = plt.subplots(3, 1, figsize=(5, 12))
colors = ['red', 'blue', 'green', 'yellow']
data = [group1]
offsets = [0]

for i, data_group in enumerate(data):
    # The position for each box within the group
    positions = [j + offsets[i] for j in range(len(data_group))]
    bp = ax[0].boxplot(data_group, positions=positions, widths=0.6, patch_artist=True)
    # Set the colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

# Custom x-axis labels
# tick_positions = [j + offset for offset in offsets for j in range(3)]
tick_labels = [
    "Fused", "Daymet", "Maurer", "NLDAS"
]

# ax[0].set_xticks(tick_positions)
ax[0].set_xticklabels(tick_labels)  # rotated for better readability

ax[0].set_title("(a)Mean (mm/day)", size=12)
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots()
# colors1 = ['red', 'blue', 'green', 'yellow']

data = [group2]
offsets = [1]

for i, data_group in enumerate(data):
    # The position for each box within the group
    positions = [j + offsets[i] for j in range(len(data_group))]
    bp = ax[1].boxplot(data_group, positions=positions, widths=0.6, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

# Custom x-axis labels
# tick_positions = [j + offset for offset in offsets for j in range(3)]
tick_labels = [
    "Fused", "Daymet", "Maurer", "NLDAS"
]

# ax[1].set_xticks(tick_positions)
ax[1].set_xticklabels(tick_labels)  # rotated for better readability
ax[1].set_title("(b)Standard Deviation", size=12)

# ax.set_title("Precipitation Statistics")
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots()

data = [group3]
offsets = [1]

for i, data_group in enumerate(data):
    # The position for each box within the group
    positions = [j + offsets[i] for j in range(len(data_group))]
    bp = ax[2].boxplot(data_group, positions=positions, widths=0.6, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

# Custom x-axis labels
# tick_positions = [j + offset for offset in offsets for j in range(3)]
tick_labels = [
    "Fused", "Daymet", "Maurer ", "NLDAS"
]

# ax[2].set_xticks(tick_positions)
ax[2].set_xticklabels(tick_labels)  # rotated for better readability
ax[2].set_title("(c)Autocorrelation (lag=1)", size=12)

# ax.set_title("Precipitation Statistics")
# plt.tight_layout()

plt.savefig('/home/kas7897/final_plots_fusion_paper/prcp_box_plots.png',  dpi=300, bbox_inches='tight')
plt.show()


# ax.set_xticks([1, 5, 9])  # Centering the x-tick labels under each group
# ax.set_xticklabels(['Mean', 'Std', 'ACF'])
#
# ax.set_title("Grouped Boxplots with 9 Arrays")
# plt.show()

