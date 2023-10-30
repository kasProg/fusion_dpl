# import torch
# from hydroDL import *
# from hydroDL.master import loadModel

# model = loadModel('/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_withloss/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/', epoch=50)
#
# para = model.parameters()
# print(len(list(para)))

import numpy as np
import pandas as pd
# import geopandas as gpd
from mpl_toolkits.basemap import Basemap, addcyclic
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from hydroDL.post import plot, stat



# from dtw import dtw


lossfactor = 0
smoothfactor = 0
expno = 2

crd = pd.read_csv("/data/kas7897/dPLHBVrelease/hydroDL-dev/gages_list.csv")



# pred = np.load('/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss60/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/pred50.npy', allow_pickle=True)
# obs = np.load('/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss60/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/obs.npy', allow_pickle=True)
eva0 = np.load(f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/nldas_extendedwithloss{lossfactor}smooth{smoothfactor}_{expno}/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/Eva50.npy', allow_pickle=True)
daymet_eva = np.load(f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/daymet/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/Eva50.npy', allow_pickle=True)
# eva1 = np.load(f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/maurer_extendedwithloss{lossfactor}smooth{smoothfactor}_{expno}/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/Eva50.npy', allow_pickle=True)
# eva2 = np.load(f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/nldas_extendedwithloss{lossfactor}smooth{smoothfactor}_{expno}/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/Eva50.npy', allow_pickle=True)

delta_nse_daymet = eva0[0]['NSE'] - daymet_eva[0]['NSE']
nse = daymet_eva[0]['NSE']


# pred = pred[:,:,0]
# obs = obs[:,:,0]
# obs_mean = np.mean(obs, axis=1)
# obs_mean = obs_mean[:, np.newaxis]
# ones = np.ones(671)# w1 = pd.read_csv(
#     f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts2.csv',
#     header=None)
# w2 = pd.read_csv(
#     f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts3.csv',
#     header=None)
# NSE = ones - (np.sum((pred - obs)**2, axis=1) / np.sum((obs - obs_mean)**2, axis=1))
# NSE_d = ones - (np.sum(abs(pred - obs), axis=1) / np.sum(abs(obs - obs_mean), axis=1))
# print(np.nanmedian(NSE), np.nanmedian(NSE_d))

m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
            projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
            resolution='c', area_thresh=10000)
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#524c4c', lake_color='#46bcec')
# m.fillcontinents(color='0.1')
m.drawcoastlines()

x, y = m(crd['LONG'].values, crd['LAT'].values)

m.scatter(x, y, c=delta_nse_daymet, cmap='seismic', alpha=0.9, zorder=2, vmin= -0.6, vmax= 0.6)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
cbar = plt.colorbar(shrink= 0.93,  orientation='horizontal', pad=0.05, extend = 'both')
cbar.set_label(f'delta_NSE_daymet', size=7)


cbar.ax.tick_params(labelsize=7)
plt.show()

# loss = pd.read_csv('/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss15/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/run.csv', header=None, delimiter=' ')
w0 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/daymetwithloss{lossfactor}smooth{smoothfactor}_{expno}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts1.csv',
    header=None)
# w0_pet = pd.read_csv(
#     f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/daymetwithloss{lossfactor}smooth{smoothfactor}_{expno}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/pet_wghts1.csv',
#     header=None)
w1 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/maurer_extendedwithloss{lossfactor}smooth{smoothfactor}_{expno}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts1.csv',
    header=None)
# w1_pet = pd.read_csv(
#     f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/maurer_extendedwithloss{lossfactor}smooth{smoothfactor}_{expno}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/pet_wghts1.csv',
#     header=None)
w2 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/nldas_extendedwithloss{lossfactor}smooth{smoothfactor}_{expno}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts1.csv',
    header=None)
# w2_pet = pd.read_csv(
#     f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/nldas_extendedwithloss{lossfactor}smooth{smoothfactor}_{expno}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/pet_wghts1.csv',
#     header=None)
dates = np.arange('1980-10-01', '2005-10-01', dtype='datetime64[D]')

# gage = '01123000'
# gage = '11274630'

time_avg_w0 = pd.DataFrame(np.mean(w0))
# time_avg_w0_pet = pd.DataFrame(np.mean(w0_pet))
# time_avg_w0 = pd.DataFrame(np.mean(w0[-3653:]))
time_avg_w1 = pd.DataFrame(np.mean(w1))
# time_avg_w1_pet = pd.DataFrame(np.mean(w1_pet))
# time_avg_w1 = pd.DataFrame(np.mean(w1[-3653:]))
time_avg_w2 = pd.DataFrame(np.mean(w2))
# time_avg_w2_pet = pd.DataFrame(np.mean(w2_pet))
# time_avg_w2 = pd.DataFrame(np.mean(w2[-3563:]))

print('DAYMET:', np.mean(time_avg_w0))
# print('DAYMET-PET:', np.mean(time_avg_w0_pet))
print('MAURER:', np.mean(time_avg_w1))
# print('MAURER-PET:', np.mean(time_avg_w1_pet))
print('NLDAS:', np.mean(time_avg_w2))
# print('NLDAS-PET:', np.mean(time_avg_w2_pet))
# time_avg_wsum = pd.DataFrame(np.mean(w0+w1+w2))
time_avg_w0.rename(columns={0: 'w0'}, inplace=True)
time_avg_w1.rename(columns={0: 'w1'}, inplace=True)
time_avg_w2.rename(columns={0: 'w2'}, inplace=True)

# to plot prcp weights spatially
# m = Basemap(projection='merc', llcrnrlon=crd['LONG'].min()-0.1, llcrnrlat=crd['LAT'].min()-0.1, urcrnrlon=crd['LONG'].max()+0.1, urcrnrlat=crd['LAT'].max()+0.1, resolution='c')
m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
            projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
            resolution='c', area_thresh=10000)
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#524c4c', lake_color='#46bcec')
# m.fillcontinents(color='0.1')
m.drawcoastlines()

x, y = m(crd['LONG'].values, crd['LAT'].values)

m.scatter(x, y, c=time_avg_w0['w0'], cmap='seismic', alpha=0.9, zorder=2)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
cbar = plt.colorbar(shrink=0.92, orientation='horizontal', pad=0.05)
cbar.set_label(f'w0(daymet), loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
cbar.ax.tick_params(labelsize=7)
plt.show()

# to plot prcp weights spatially
# m = Basemap(projection='merc', llcrnrlon=crd['LONG'].min()-0.1, llcrnrlat=crd['LAT'].min()-0.1, urcrnrlon=crd['LONG'].max()+0.1, urcrnrlat=crd['LAT'].max()+0.1, resolution='c')
m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
            projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
            resolution='c', area_thresh=10000)
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#524c4c', lake_color='#46bcec')
# m.fillcontinents(color='0.1')
m.drawcoastlines()

x, y = m(crd['LONG'].values, crd['LAT'].values)

m.scatter(x, y, c=time_avg_w1['w1'], cmap='seismic', alpha=0.9, zorder=2)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
cbar = plt.colorbar(shrink=0.92, orientation='horizontal', pad=0.05)
cbar.set_label(f'w1(maurer), loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
cbar.ax.tick_params(labelsize=7)
plt.show()
# plt.savefig('/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_withloss/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/w0.png', bbox_inches = 'tight')

# to plot prcp weights spatially
# m = Basemap(projection='merc', llcrnrlon=crd['LONG'].min()-0.1, llcrnrlat=crd['LAT'].min()-0.1, urcrnrlon=crd['LONG'].max()+0.1, urcrnrlat=crd['LAT'].max()+0.1, resolution='c')
m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
            projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
            resolution='c', area_thresh=10000)
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#524c4c', lake_color='#46bcec')
# m.fillcontinents(color='0.1')
m.drawcoastlines()

x, y = m(crd['LONG'].values, crd['LAT'].values)

m.scatter(x, y, c=time_avg_w2['w2'], cmap='seismic', alpha=0.9, zorder=2)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
cbar = plt.colorbar(shrink=0.92, orientation='horizontal', pad=0.05)
cbar.set_label(f'w2(nldas), loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
cbar.ax.tick_params(labelsize=7)
plt.show()
#
prcp_gagewise_daymet = np.zeros(len(crd))
prcp_gagewise_maurer = np.zeros(len(crd))
prcp_gagewise_nldas = np.zeros(len(crd))

eff_wght_daymet = np.zeros(len(crd))
eff_wght_maurer = np.zeros(len(crd))
eff_wght_nldas = np.zeros(len(crd))

eff_prcp_daymet = np.zeros(len(crd))
eff_prcp_maurer = np.zeros(len(crd))
eff_prcp_nldas = np.zeros(len(crd))

igage = 0

start_date = '1980-01-01'
end_date = '2022-12-31'
#gages = ['01123000', '11274630', '14301000']

# for gage in gages:
#     huc = crd.loc[crd['gage'] == int(gage), 'huc'].values[0]
#     if len(str(huc)) == 1:
#         huc = '0' + str(huc)
#     prcp_daymet_dir = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/daymet/{huc}/{gage}_lump_cida_forcing_leap.txt"
#     prcp_maurer_dir = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/maurer_extended/{huc}/{gage}_lump_maurer_forcing_leap.txt"
#     prcp_nldas_dir = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/nldas_extended/{huc}/{gage}_lump_nldas_forcing_leap.txt"
#
#     # prcp_daymet = pd.read_csv(prcp_daymet_dir, sep=r'\s+', header=None, skiprows=4)[5][:9131] #slicing from 1980 to 2004
#     prcp_daymet = pd.read_csv(prcp_daymet_dir, sep=r'\s+', header=None, skiprows=4)[5][274:9405].reset_index(drop=True) #slicing from 1980 to 2004
#     # prcp_daymet = pd.read_csv(prcp_daymet_dir, sep=r'\s+', header=None, skiprows=4) #slicing from 1980 to 2004
#     prcp_maurer = pd.read_csv(prcp_maurer_dir, sep=r'\s+', header=None, skiprows=4)[5][274:9405].reset_index(drop=True)
#     prcp_nldas = pd.read_csv(prcp_nldas_dir, sep=r'\s+', header=None, skiprows=4)[5][274:9405].reset_index(drop=True)
#
#
#     # prcp_daymet_mean =  np.mean(prcp_daymet)
#     # prcp_maurer_mean =  np.mean(prcp_maurer)
#     # prcp_nldas_mean =  np.mean(prcp_nldas)
#
#     # prcp_gagewise_daymet[igage] =  np.mean(prcp_daymet)
#     # prcp_gagewise_maurer[igage] =  np.mean(prcp_maurer)
#     # prcp_gagewise_nldas[igage] =  np.mean(prcp_nldas)
#     # igage = igage + 1
#     # print(time_avg_w0)
#     # k = time_avg_w0[0:]
#     # print(time_avg.shape()
#     # print(crd)
#     # gages_prcp = np.array([len(prcp_dataframe)])
#
#     k = 1
#     prcp_index = crd.loc[crd['gage'] == int(gage)].index[0]
#     w0_filtered = w0[prcp_index]
#     w1_filtered = w1[prcp_index]
#     w2_filtered = w2[prcp_index]
#     # w0_filtered[prcp_daymet==0] = np.nan
#     # w1_filtered[prcp_maurer==0] = np.nan
#     # w2_filtered[prcp_nldas==0] = np.nan
#
#
#
#     #to plot loss streamflow vs prcp
#     # sf = loss[6].astype('str').str.extractall('(\d+)').unstack().fillna('').sum(axis=1).astype(float)/1000
#     # prcp = loss[9].astype('str').str.extractall('(\d+)').unstack().fillna('').sum(axis=1).astype(float)/1000
#     #
#     #
#     # plt.plot(prcp, label='precipitation loss')
#     # plt.plot(sf, label='streamflow_loss')
#     # plt.title('loss_factor=15')
#     # plt.legend()
#     # plt.show()
#     #to plot prcp weight temporally
#     # wsum_filtered = w0_filtered + w1_filtered + w2_filtered
#     #rolling mean
#
#     #uncomment here
#     rm_window=1
#     w0_filtered_rm = w0_filtered.rolling(window=rm_window).mean()
#     w1_filtered_rm = w1_filtered.rolling(window=rm_window).mean()
#     w2_filtered_rm = w2_filtered.rolling(window=rm_window).mean()
#     # wsum_filtered_rm = wsum_filtered.rolling(window=rm_window).mean()
#
#     fig = plt.figure(figsize=(12, 6))
#     #
#     plt.plot(w0_filtered_rm, label='w0_non0(daymet)')
#     plt.plot(w1_filtered_rm, label='w1_non0(maurer)')
#     plt.plot(w2_filtered_rm, label='w2_non0(nldas)')
#     # plt.plot(wsum_filtered, label='wsum_non0')
#     plt.title(f'{gage}_with rolling mean {rm_window} days; loss factor: {lossfactor}; smoothing factor: {smoothfactor}')
#     # # Add a legend
#     # plt.ylim((0.6, 1.2))
#     plt.legend(loc='upper right')
#     plt.show()
#
#     # plt.savefig()
#     # plt.savefig('/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_withloss/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/20gage.png', bbox_inches = 'tight')
#
#     # Add axis labels and a title
#     # plt.title(crd)


corr1 = np.zeros(len(crd))
corr2 = np.zeros(len(crd))
rmse1 = np.zeros(len(crd))
rmse2 = np.zeros(len(crd))
rel_rmse = np.zeros(len(crd))
error1 = np.zeros(len(crd))
error2 = np.zeros(len(crd))
mae1 = np.zeros(len(crd))
mae2 = np.zeros(len(crd))
for i in range(len(crd)):
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

    prcp_daymet = pd.read_csv(prcp_daymet_dir, sep=r'\s+', header=None, skiprows=4)[5][274:9405].reset_index(drop=True) #slicing from 1980 to 2004
    # prcp_daymet = pd.read_csv(prcp_daymet_dir, sep=r'\s+', header=None, skiprows=4) #slicing from 1980 to 2004
    # prcp_daymet = prcp_daymet[prcp_daymet[0] ==year]
    # prcp_daymet = prcp_daymet[5]

    prcp_maurer = pd.read_csv(prcp_maurer_dir, sep=r'\s+', header=None, skiprows=4)[5][274:9405].reset_index(drop=True)
    # prcp_maurer = pd.read_csv(prcp_maurer_dir, sep=r'\s+', header=None, skiprows=4)
    # prcp_maurer = prcp_maurer[prcp_maurer[0] ==year]
    # prcp_maurer = prcp_maurer[5]

    prcp_nldas = pd.read_csv(prcp_nldas_dir, sep=r'\s+', header=None, skiprows=4)[5][274:9405].reset_index(drop=True)
    # prcp_nldas = pd.read_csv(prcp_nldas_dir, sep=r'\s+', header=None, skiprows=4)
    # prcp_nldas = prcp_nldas[prcp_nldas[0] ==year]
    # prcp_nldas = prcp_nldas[prcp_nldas[0] ==year]

    # prcp_nldas_df = pd.DataFrame(prcp_nldas, index=dates)

    eff_prcp_daymet[i] = np.sum(prcp_daymet*w0[i])
    eff_wght_daymet[i] = np.sum(prcp_daymet*w0[i])/np.sum(prcp_daymet)
    eff_wght_maurer[i] = np.sum(prcp_maurer*w1[i])/np.sum(prcp_maurer)
    eff_prcp_maurer[i] = np.sum(prcp_maurer*w1[i])
    eff_wght_nldas[i] = np.sum(prcp_nldas*w2[i])/np.sum(prcp_nldas)
    eff_prcp_nldas[i] = np.sum(prcp_nldas*w2[i])

    prcp_gagewise_daymet[i] =  np.sum(prcp_daymet)
    prcp_gagewise_maurer[i] =  np.sum(prcp_maurer)# prcp_daymet_mean =  np.mean(prcp_daymet)
    prcp_gagewise_nldas[i] =  np.sum(prcp_nldas)# prcp_maurer_mean =  np.mean(prcp_maurer)

    eff_nldas = prcp_nldas*w2[i]
    # eff_nldas.set_index(dates, drop=True, inplace=True)
    # eff_nldas_ymean = eff_nldas.resample('Y').mean()
    eff_daymet = prcp_daymet*w0[i]
    # eff_daymet.set_index(dates, drop=True, inplace=True)
    # eff_daymet_ymean = eff_daymet.resample('Y').mean()

    corr1[i] = pearsonr(prcp_daymet, prcp_nldas)[0]
    corr2[i] = pearsonr(eff_daymet, eff_nldas)[0]
    error1[i] = np.nanmedian((prcp_daymet-eff_nldas)/prcp_daymet)
    # error1[i] = error1[~np.isnan(error1)]
    # error1[i] = error1[~np.isinf(error1)]
    error2[i] = np.nanmedian((prcp_daymet-prcp_nldas)/prcp_daymet)
    # error2[i] = error2[~np.isnan(error2)]
    # error2[i] = error2[~np.isinf(error2)]
    # rel_rmse[i] = np.sqrt(mean_squared_error(error1, error2))
    rmse1[i] = np.sqrt(mean_squared_error(prcp_daymet, prcp_nldas))
    rmse2[i] = np.sqrt(mean_squared_error(eff_daymet, eff_nldas))

    mae1[i] = mean_absolute_error(prcp_daymet, prcp_nldas)
    mae2[i] = mean_absolute_error(eff_daymet, eff_nldas)
#
# print('DAYMET:', np.mean(eff_wght_daymet))
# print('MAURER:', np.mean(eff_wght_maurer))
# print('NLDAS:', np.mean(eff_wght_nldas))
# diff_wght = eff_wght_daymet-eff_wght_nldas
#
# fig, axs = plt.subplots(2, 2, figsize=(10, 8))
# m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
#             projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
#             resolution='c', area_thresh=10000)
#
# axs[0,0].set_title(f"(a) w'\u2080 (effective weight associated to daymet)", size=12)
# m.ax = axs[0,0]
# m.drawmapboundary(fill_color='#46bcec')
# m.fillcontinents(color='#524c4c', lake_color='#46bcec')
# # m.fillcontinents(color='0.1')
# m.drawcoastlines()
#
# x, y = m(crd['LONG'].values, crd['LAT'].values)
#
# scatter = m.scatter(x, y, c=eff_wght_daymet, cmap='seismic', alpha=0.9, zorder=2)
# # cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cax1 = fig.add_axes([0.13, 0.52, 0.35, 0.02])
# cbar = plt.colorbar(scatter, cax=cax1, shrink=0.9, orientation='horizontal', pad=0.05)
# # cbar.set_label(f'wsum, loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
# # plt.title(f"w'\u2080 (effective weight associated to daymet)", size=12)
# cbar.ax.tick_params(labelsize=10)
# # plt.show()
#
# axs[0,1].set_title(f"(b) w'\u2081 (effective weight associated to maurer)", size=12)
# m.ax = axs[0,1]
# # m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
# #             projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
# #             resolution='c', area_thresh=10000)
# m.drawmapboundary(fill_color='#46bcec')
# m.fillcontinents(color='#524c4c', lake_color='#46bcec')
# # m.fillcontinents(color='0.1')
# m.drawcoastlines()
#
# x, y = m(crd['LONG'].values, crd['LAT'].values)
#
# scatter = m.scatter(x, y, c=eff_wght_maurer, cmap='seismic', alpha=0.9, zorder=2)
# # cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cax2 = fig.add_axes([0.55, 0.52, 0.35, 0.02])
# cbar = plt.colorbar(scatter, cax=cax2, shrink=0.9, orientation='horizontal', pad=0.05)
# # cbar.set_label(f'wsum, loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
# # plt.title(f"w'\u2081 (effective weight associated to maurer)", size=12)
# cbar.ax.tick_params(labelsize=10)
# # plt.show()
#
# axs[1,0].set_title(f"(c) w'\u2082 (effective weight associated to NLDAS2)", size=12)
# m.ax = axs[1, 0]
# # m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
# #             projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
# #             resolution='c', area_thresh=10000)
# m.drawmapboundary(fill_color='#46bcec')
# m.fillcontinents(color='#524c4c', lake_color='#46bcec')
# # m.fillcontinents(color='0.1')
# m.drawcoastlines()
#
# x, y = m(crd['LONG'].values, crd['LAT'].values)
#
# scatter = m.scatter(x, y, c=eff_wght_nldas, cmap='seismic', alpha=0.9, zorder=2)
# # cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cax3 = fig.add_axes([0.13, 0.1, 0.35, 0.02])
# cbar = plt.colorbar(scatter, cax=cax3, shrink=0.9, orientation='horizontal', pad=0.05)
# # cbar.set_label(f'wsum, loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
# # plt.title(f"w'\u2082 (effective weight associated to NLDAS-2)", size=12)
# cbar.ax.tick_params(labelsize=10)
#
# axs[1,1].set_title(f"(d) w'\u2080 - w'\u2082", size=12)
# m.ax = axs[1, 1]
# # m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
# #             projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
# #             resolution='c', area_thresh=10000)
# m.drawmapboundary(fill_color='#46bcec')
# m.fillcontinents(color='#524c4c', lake_color='#46bcec')
# # m.fillcontinents(color='0.1')
# m.drawcoastlines()
#
# x, y = m(crd['LONG'].values, crd['LAT'].values)
#
# scatter = m.scatter(x, y, c=diff_wght, cmap='seismic', alpha=0.9, zorder=2)
# # cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cax3 = fig.add_axes([0.55, 0.1, 0.35, 0.02])
# cbar = plt.colorbar(scatter, cax=cax3, shrink=0.9, orientation='horizontal', pad=0.05)
# # cbar.set_label(f'wsum, loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
# # plt.title(f"w'\u2082 (effective weight associated to NLDAS-2)", size=12)
# cbar.ax.tick_params(labelsize=10)
#
# plt.show()
#
#
# diff = abs(prcp_gagewise_daymet - prcp_gagewise_nldas)
# eff_diff = abs(prcp_gagewise_daymet - eff_prcp_nldas)
# diff_corr = corr2-corr1
# diff_rmse = rmse1-rmse2
# percent_chng_daymet = (eff_prcp_daymet - prcp_gagewise_daymet)/prcp_gagewise_daymet *100
# percent_chng_maurer = (eff_prcp_maurer - prcp_gagewise_maurer)/prcp_gagewise_maurer *100
# percent_chng_nldas = (eff_prcp_nldas - prcp_gagewise_nldas)/prcp_gagewise_nldas *100
# # print(min(eff_diff))
# # print(max(eff_diff))
#
# fig, axs = plt.subplots(1, 2, figsize=(10, 8))
#
# m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
#             projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
#             resolution='c', area_thresh=10000)
#
# axs[0].set_title(f'correlation difference , loss factor {lossfactor}; smoothing factor {smoothfactor}', size=12)
# m.ax = axs[0]
# m.drawmapboundary(fill_color='#46bcec')
# m.fillcontinents(color='#524c4c', lake_color='#46bcec')
# # m.fillcontinents(color='0.1')
# m.drawcoastlines()
# x, y = m(crd['LONG'].values, crd['LAT'].values)
# scatter = m.scatter(x, y, c=diff_corr, cmap='seismic', alpha=0.9, zorder=2, vmin = -0.002, vmax=0.002)
# # cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cbar = plt.colorbar(scatter, shrink=0.92, orientation='horizontal', pad=0.05, extend='both')
# # cbar.set_label(f'correlation difference , loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
# cbar.ax.tick_params(labelsize=7)
#
# axs[1].set_title(f'correlation difference , loss factor {lossfactor}; smoothing factor {smoothfactor}', size=12)
# m.ax = axs[1]
# m.drawmapboundary(fill_color='#46bcec')
# m.fillcontinents(color='#524c4c', lake_color='#46bcec')
# # m.fillcontinents(color='0.1')
# m.drawcoastlines()
# x, y = m(crd['LONG'].values, crd['LAT'].values)
# scatter = m.scatter(x, y, c=diff_rmse, cmap='seismic', alpha=0.9, zorder=2, vmin = -0.002, vmax=0.002)
# # cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cbar = plt.colorbar(scatter, shrink=0.92, orientation='horizontal', pad=0.05, extend='both')
# # cbar.set_label(f'correlation difference , loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
# cbar.ax.tick_params(labelsize=7)
#
#
# plt.show()


# lower_limit = -2000
# upper_limit = 2000
#
# # Count the number of values within the limits
# num_within_limits = np.count_nonzero((eff_diff >= lower_limit) & (eff_diff <= upper_limit))
#
# # Calculate the percentage of values within the limits
# percentage_within_limits = num_within_limits / eff_diff.size * 100
#
# print(f"{percentage_within_limits:.2f}
# percent = np.sum(eff_diff < diff) / len(diff) * 100
percent1 = np.sum(rmse1 > rmse2) / len(rmse1) * 100
percent2 = np.sum(corr1 < corr2) / len(corr1) * 100
percent3 = np.sum(mae1 > mae2) / len(mae1) * 100

percent4 = np.sum(error1<error2)/len(error1)
percent5 = np.sum(error1>error2)/len(error1)
percent6 = np.sum(error1==error2)/len(error1)
print(f"corr_{percent2:.2f}%")
print(f"rmse_{percent1:.2f}%")
print(f"rmse_{percent3:.2f}%")
print(np.median(corr1))
print(np.median(corr2))


# select_gages = eff_diff < diff
gages1 = [crd['gage'][580], crd['gage'][5], crd['gage'][154], crd['gage'][340], crd['gage'][460], crd['gage'][608], crd['gage'][555]]
for i in range(len(gages1)):
    gage = gages1[i]
    huc = crd.loc[crd['gage'] == int(gage), 'huc'].values[0]
    if len(str(huc)) == 1:
        huc = '0' + str(huc)
    if len(str(gage)) == 7:
        gage = '0' + str(gage)
    gage = str(gage)
    prcp_daymet_dir = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/daymet/{huc}/{gage}_lump_cida_forcing_leap.txt"
    prcp_maurer_dir = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/maurer_extended/{huc}/{gage}_lump_maurer_forcing_leap.txt"
    prcp_nldas_dir = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/nldas_extended/{huc}/{gage}_lump_nldas_forcing_leap.txt"
    # prcp_daymet = pd.read_csv(prcp_daymet_dir, sep=r'\s+', header=None, skiprows=4)[5][:9131] #slicing from 1980 to 20
    prcp_daymet = pd.read_csv(prcp_daymet_dir, sep=r'\s+', header=None, skiprows=4)[5][274:9405].reset_index(drop=True)
    # prcp_daymet = pd.read_csv(prcp_daymet_dir, sep=r'\s+', header=None, skiprows=4) #slicing from 1980 to 2004
    prcp_maurer = pd.read_csv(prcp_maurer_dir, sep=r'\s+', header=None, skiprows=4)[5][274:9405].reset_index(drop=True)
    prcp_nldas = pd.read_csv(prcp_nldas_dir, sep=r'\s+', header=None, skiprows=4)[5][274:9405].reset_index(drop=True)

    k = 1
    prcp_index = crd.loc[crd['gage'] == int(gage)].index[0]
    # w0_filtered = w0[prcp_index]*prcp_daymet
    w0_filtered = w0[prcp_index]
    # w1_filtered = w1[prcp_index]*prcp_maurer
    w1_filtered = w1[prcp_index]
    # w2_filtered = w2[prcp_index]*prcp_nldas
    w2_filtered = w2[prcp_index]
    # w0_filtered[prcp_daymet==0] = np.nan
    # w1_filtered[prcp_maurer==0] = np.nan
    # w2_filtered[prcp_nldas==0] = np.nan

    corr_coef1 = pearsonr(prcp_daymet, prcp_nldas)
    corr_coef2 = pearsonr(prcp_daymet, w2_filtered)
    # corr_coef1 = correlate(prcp_daymet, prcp_nldas,mode='same')
    # corr_coef2 = correlate(prcp_daymet, w2_filtered, mode='same')
    # dtw1 = dtw_distance(prcp_daymet, prcp_nldas)
    # dtw2 = dtw_distance(prcp_daymet, w2_filtered)

    #uncomment here
    rm_window=1
    w0_filtered_rm = w0_filtered.rolling(window=rm_window).mean()                                                       
    w1_filtered_rm = w1_filtered.rolling(window=rm_window).mean()
    w2_filtered_rm = w2_filtered.rolling(window=rm_window).mean()

    prcp_daymet_rm = prcp_daymet.rolling(window=rm_window).mean()
    prcp_nldas_rm = prcp_nldas.rolling(window=rm_window).mean()
    # wsum_filtered_rm = wsum_filtered.rolling(window=rm_window).mean()

    fig = plt.figure(figsize=(12, 6))
    #
    plt.plot(prcp_nldas_rm, w2_filtered_rm, 'o')
    # plt.plot(y = w0_filtered_rm[:730], x= prcp_daymet_rm[:730], label='w0 vs obs daymet')
    plt.xlabel('nldas(mm/day)')
    plt.ylabel('w2')
    # plt.plot(, label='applied daymet (0.9-1.1)')
    # plt.plot(prcp_nldas_rm[:730], label='obs nldas')
    # plt.plot(w2_filtered_rm[:730], label='applied nldas(0.8-1.2)')
    # plt.plot(wsum_filtered, label='wsum_non0')
    plt.title(f'{gage}_with rolling mean {rm_window} days; loss factor: {lossfactor}; smoothing factor: {smoothfactor}')
    # # Add a legend
    # plt.ylim((0.6, 1.2))
    # plt.legend(loc='upper right')
    plt.show()