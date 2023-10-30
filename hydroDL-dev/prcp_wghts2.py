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
import json
from statsmodels.tsa.stattools import acf

from matplotlib.colors import Normalize

# from hydroDL.post import plot, stat

lossfactor = 23
smoothfactor = 0
# expno =4
crd = pd.read_csv("/data/kas7897/dPLHBVrelease/hydroDL-dev/gages_list.csv")
rm_window = 1

pred = np.load('/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/allPET200_extended_withloss200smooth0/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/pred50.npy', allow_pickle=True)
# pred = np.load("/data/kas7897/dPLHBVrelease/output/rnnStreamflow/CAMELSDemo-daymet/TestRun/epochs300_batch100_rho365_hiddensize256_Tstart19801001_Tend19951001/pred.npy", allow_pickle=True)
obs = np.load('/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/allPET200_extended_withloss200smooth0/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/obs.npy', allow_pickle=True)
# obs = np.load("/data/kas7897/dPLHBVrelease/output/rnnStreamflow/CAMELSDemo-daymet/TestRun/epochs300_batch100_rho365_hiddensize256_Tstart19801001_Tend19951001/obs.npy", allow_pickle=True)


# eva0 = [stat.statError(pred[:,:,0], obs.squeeze())]
# eva0=[stat.statError(x.squeeze(), obs.squeeze()) for x in pred]

center_crd = crd[(crd['LONG'] >= -110) & (crd['LONG'] <= -90)]

eva0 = np.load(f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/Eva50.npy', allow_pickle=True)
# eva0 = np.load(f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/maurer_extendedwithloss0smooth0_11/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/Eva50.npy', allow_pickle=True)
daymet_eva = np.load(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/daymet/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/Eva50.npy', allow_pickle=True)
eva_nldas = np.load(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/nldas/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/Eva50.npy', allow_pickle=True)
# eva1 = np.load(f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/maurer_extendedwithloss{lossfactor}smooth{smoothfactor}_{expno}/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/Eva50.npy', allow_pickle=True)
# eva2 = np.load(f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/nldas_extendedwithloss{lossfactor}smooth{smoothfactor}_{expno}/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/Eva50.npy', allow_pickle=True)
#
gages_select = ['06879650', '07145700', '06470800']
ind1 = crd[crd['gage']==int(gages_select[0])].index
ind2 = crd[crd['gage']==int(gages_select[1])].index
ind3 = crd[crd['gage']==int(gages_select[2])].index


lstm_daymet = np.load("/data/kas7897/dPLHBVrelease/output/rnnStreamflow/CAMELSDemo-daymet/TestRun/epochs300_batch100_rho365_hiddensize256_Tstart19801001_Tend19951001/Eva300.npy", allow_pickle=True)
delta_nse_daymet = eva0[0]['NSE'] - daymet_eva[0]['NSE']
delta_kge_daymet = eva0[0]['KGE'] - daymet_eva[0]['KGE']
delta_highRMSE_daymet = daymet_eva[0]['lowRMSE']-eva0[0]['lowRMSE']
crd['NSE'] = delta_nse_daymet
crd['KGE'] = delta_kge_daymet
delta_kge_daymet_center = delta_kge_daymet[center_crd.index]
delta_nse_daymet_center = delta_nse_daymet[center_crd.index]
center_crd['KGE'] = delta_kge_daymet_center
center_crd['NSE'] = delta_nse_daymet_center
nse = daymet_eva[0]['NSE']



max_ind_multi = np.argsort(eva0[0]['NSE'])[-50:]
max_ind_lstmDaymet = np.argsort(lstm_daymet[0]['NSE'])[-50:]
max_ind_Daymet = np.argsort(daymet_eva[0]['NSE'])[-50:]
max_stations_multi = crd['huc'][max_ind_multi]
max_stations_lstmDaymet = crd['huc'][max_ind_lstmDaymet]
max_stations_Daymet = crd['huc'][max_ind_Daymet]

data = np.array([1, 2, 2, 3, 4, 4, 4, 5, 6, 6, 7])

# Count occurrences of each unique value
unique, counts = np.unique(max_stations_multi, return_counts=True)

# Calculate the percentage of occurrence for each value
total_values = len(max_stations_multi)
percentage_dict = {value: (count / total_values) * 100 for value, count in zip(unique, counts)}
print("For multi-forcing:")

print(percentage_dict)

unique, counts = np.unique(max_stations_lstmDaymet, return_counts=True)

# Calculate the percentage of occurrence for each value
total_values = len(max_stations_multi)
percentage_dict = {value: (count / total_values) * 100 for value, count in zip(unique, counts)}
print("For LSTM Daymet:")

print(percentage_dict)

unique, counts = np.unique(max_stations_Daymet, return_counts=True)

# Calculate the percentage of occurrence for each value
total_values = len(max_stations_multi)
percentage_dict = {value: (count / total_values) * 100 for value, count in zip(unique, counts)}
print("For dPLHBV daymet:")

print(percentage_dict)
# Calculate the percentage of repetition for each value

# # pred = pred[:,:,0]
# # obs = obs[:,:,0]
# # obs_mean = np.mean(obs, axis=1)
# # obs_mean = obs_mean[:, np.newaxis]
# # ones = np.ones(671)# w1 = pd.read_csv(
# #     f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts2.csv',
# #     header=None)
# # w2 = pd.read_csv(
# #     f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts3.csv',
# #     header=None)
# # NSE = ones - (np.sum((pred - obs)**2, axis=1) / np.sum((obs - obs_mean)**2, axis=1))
# # NSE_d = ones - (np.sum(abs(pred - obs), axis=1) / np.sum(abs(obs - obs_mean), axis=1))
# # print(np.nanmedian(NSE), np.nanmedian(NSE_d))

subsetPath = "/data/kas7897/dPLHBVrelease/hydroDL-dev/example/dPLHBV/Sub531ID.txt"
with open(subsetPath, 'r') as fp:
    sub531IDLst = json.load(fp)  # Subset 531 ID List
# get the evaluation metrics on 531 subset\
logtestIDLst = list(crd['gage'])
[C, ind1, SubInd] = np.intersect1d(sub531IDLst, logtestIDLst, return_indices=True)
evaframe = pd.DataFrame(eva0[0])
evaframeSub = evaframe.loc[SubInd, list(evaframe.keys())]
evaS531Dict = [{col:evaframeSub[col].values for col in evaframeSub}] # 531 subset evaDict

# print NSE median value of testing basins
# print('Testing finished! Evaluation results saved in\n', outpath)
# print('For basins of whole CAMELS, NSE median:', np.nanmedian(evaDict[0]['NSE']))
print('For basins of 531 subset, NSE median:', np.nanmedian(evaS531Dict[0]['NSE']))
print('For basins of 531 subset, KGE median:', np.nanmedian(evaS531Dict[0]['KGE']))



fig, axs = plt.subplots(1, 2, figsize=(10, 8))
# cbar_axs = [fig.add_axes([0.1, 0.1, 0.8, 0.02]) for _ in range(4)]


m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
            projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
            resolution='c', area_thresh=10000)

# axs[0,0].set_title(f'(a) NSE of dPLHBV (simple) trained on Daymet', size=12)
# m.ax = axs[0, 0]
# # m.drawmapboundary()
# m.etopo(scale=0.5, alpha=0.5)
# m.drawcoastlines()
# m.drawcountries(linewidth=1)
# m.drawstates(linewidth=0.4)
#
# x, y = m(crd['LONG'].values, crd['LAT'].values)
#
# scatter = m.scatter(x, y, c=daymet_eva[0]['highRMSE'], cmap='seismic', alpha=0.9, zorder=2,s=20)
# # cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cax1 = fig.add_axes([0.13, 0.52, 0.35, 0.02])
# cbar = plt.colorbar(scatter, cax=cax1, shrink=0.9, orientation='horizontal', pad=0.05)
# # cbar.set_label(f'w0(daymet), loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
# cbar.ax.tick_params(labelsize=10)
# # plt.title(f'(a) w\u2080 (weight associated to Daymet)', size=12)
#
#
# # plt.show()
#
# # to plot prcp weights spatially
# # m = Basemap(projection='merc', llcrnrlon=crd['LONG'].min()-0.1, llcrnrlat=crd['LAT'].min()-0.1, urcrnrlon=crd['LONG'].max()+0.1, urcrnrlat=crd['LAT'].max()+0.1, resolution='c')
# # m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
# #             projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
# #             resolution='c', area_thresh=10000)
# axs[0,1].set_title(f'(b) NSE of LSTM model trained on Daymet', size=12)
# m.ax = axs[0, 1]
# m.etopo(scale=0.5, alpha=0.5)
# m.drawcoastlines()
# m.drawcountries(linewidth=1)
# m.drawstates(linewidth=0.4)
#
#
# x, y = m(crd['LONG'].values, crd['LAT'].values)
#
# scatter = m.scatter(x, y, c=lstm_daymet[0]['highRMSE'], cmap='seismic', alpha=0.9, zorder=2, s=20)
# # cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cax2 = fig.add_axes([0.55, 0.52, 0.35, 0.02])
# cbar = plt.colorbar(scatter, cax=cax2, shrink=0.9, orientation='horizontal', pad=0.05)
# # cbar.set_label(f'w1(maurer), loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
# cbar.ax.tick_params(labelsize=10)
# plt.title(f'(b) w\u2081 (weight associated to Maurer)', size=12)

# plt.show()
# plt.savefig('/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_withloss/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/w0.png', bbox_inches = 'tight')

# to plot prcp weights spatially
# m = Basemap(projection='merc', llcrnrlon=crd['LONG'].min()-0.1, llcrnrlat=crd['LAT'].min()-0.1, urcrnrlon=crd['LONG'].max()+0.1, urcrnrlat=crd['LAT'].max()+0.1, resolution='c')
# m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
#             projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
#             resolution='c', area_thresh=10000)
axs[0].set_title(f'(a) NSE of Fusion δHBV (trained on Multiple P)', size=12)
m.ax = axs[0]
m.etopo(scale=0.5, alpha=0.5)
m.drawcoastlines()
m.drawcountries(linewidth=1)
m.drawstates(linewidth=0.4)

x, y = m(crd['LONG'].values, crd['LAT'].values)

scatter = m.scatter(x, y, c=eva0[0]['NSE'], cmap='seismic', alpha=0.9, zorder=2, vmin = 0.0, s=20)
cax3 = fig.add_axes([0.13, 0.32, 0.35, 0.02])
cbar = plt.colorbar(scatter, cax=cax3, shrink=0.9, orientation='horizontal', pad=0.05, extend='min')
cbar.ax.tick_params(labelsize=10)

axs[1].set_title(f'(b) \u0394NSE i.e NSE(Fusion δHBV) minus NSE(δHBV)',size=12)
m.ax = axs[1]
m.etopo(scale=0.5, alpha=0.5)
m.drawcoastlines()
m.drawcountries(linewidth=1)
m.drawstates(linewidth=0.4)

x, y = m(crd['LONG'].values, crd['LAT'].values)

scatter = m.scatter(x, y, c=delta_nse_daymet, cmap='seismic', alpha=0.9, zorder=2, vmin=-0.35, vmax=0.35,s=20)
cax4 = fig.add_axes([0.55, 0.32, 0.35, 0.02])
cbar = plt.colorbar(scatter, cax=cax4, shrink=0.9, orientation='horizontal', pad=0.05, extend='both')

cbar.ax.tick_params(labelsize=10)
# plt.savefig('/home/kas7897/final_plots_fusion_paper/nse_analysis_2.png', dpi=300, bbox_inches='tight')

plt.show()
# plt.show()




# loss = pd.read_csv('/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss15/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/run.csv', header=None, delimiter=' ')
w0 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts1.csv',
    header=None)

w1 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts2.csv',
    header=None)

w2 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts3.csv',
    header=None)


gages = ['01123000', '11274630', '14301000']
# gage = '01123000'
# gage = '11274630'

time_avg_w0 = pd.DataFrame(np.mean(w0))
# time_avg_w0 = pd.DataFrame(np.mean(w0[-3653:]))
time_avg_w1 = pd.DataFrame(np.mean(w1))
# time_avg_w1 = pd.DataFrame(np.mean(w1[-3653:]))
time_avg_w2 = pd.DataFrame(np.mean(w2))
# time_avg_w2 = pd.DataFrame(np.mean(w2[-3563:]))

print('DAYMET:', np.mean(time_avg_w0))
print('MAURER:', np.mean(time_avg_w1))
print('NLDAS:', np.mean(time_avg_w2))
time_avg_wsum = pd.DataFrame(np.mean(w0+w1+w2))
time_avg_w0.rename(columns={0: 'w0'}, inplace=True)
time_avg_w1.rename(columns={0: 'w1'}, inplace=True)
time_avg_w2.rename(columns={0: 'w2'}, inplace=True)

dom_wght = np.argmax(np.array([time_avg_w0['w0'], time_avg_w1['w1'], time_avg_w2['w2']]), axis=0) + 1


fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title(f'Dominant weight(1:Daymet; 2:Maurer; 3:NLDAS)', size=12)

m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
            projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
            resolution='c', area_thresh=10000)
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#D3D3D3', lake_color='#46bcec')
# m.fillcontinents(color='0.1')
m.drawcoastlines()
m.drawstates()

# Plot data points
for value, lat, lon in zip(dom_wght, crd['LAT'].values, crd['LONG'].values):
    x, y = m(lon, lat)
    # x, y = m(crd['LONG'].values, crd['LAT'].values)

    m.scatter(x, y)
    if value==1:
        ax.text(x, y + 0.1, str(value), color = 'red',fontsize=12)  # places the number slightly to the right and above the point
    elif value==2:
        ax.text(x + 0.1, y + 0.1, str(value), color = 'blue',fontsize=12)  # places the number slightly to the right and above the point
    elif value==3:
        ax.text(x + 0.1, y + 0.1, str(value), color='green', fontsize=12)  # places the number slightly to the right and above the point
plt.show()
# fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# ax1 = axes[0]
# m1 = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
#             projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
#             resolution='c', area_thresh=10000)
# m1.drawmapboundary(fill_color='#46bcec')
# m1.fillcontinents(color='#524c4c', lake_color='#46bcec')
# # m.fillcontinents(color='0.1')
# m1.drawcoastlines()
# x1, y1 = m1(crd['LONG'].values, crd['LAT'].values)
# # m1.scatter(x1, y1, c=daymet_eva[0]['NSE'], cmap='seismic', alpha=0.9, zorder=2, vmin=-0.5, vmax=0.5)
# m1.scatter(x1, y1, c=time_avg_slope_wo[0], cmap='seismic', alpha=0.9, zorder=2, vmin=-0.5, vmax=0.5)
# # m.scatter(x, y, c=delta_nse_daymet, cmap='seismic', alpha=0.9, zorder=2)
# # cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# # cbar = plt.colorbar(shrink= 0.93,  orientation='horizontal', pad=0.05, extend = 'both')
# # cbar.ax.tick_params()
# cbar1 = plt.colorbar(shrink= 0.93,  orientation='horizontal', pad=0.05, extend='both')
# # plt.title(f'(d) NSE for Model Trained on Daymet Only', size=12)
# plt.title(f'(a)Gradients of Slope associated to Weight of Daymet', size=12)
# cbar1.ax.tick_params(labelsize=10)
#
# ax2 = axes[1]
# m2 = Basemap(ax=ax2, llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
#             projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
#             resolution='c', area_thresh=10000)
# m2.drawmapboundary(fill_color='#46bcec')
# m2.fillcontinents(color='#524c4c', lake_color='#46bcec')
# # m.fillcontinents(color='0.1')
# m2.drawcoastlines()
# x2, y2 = m2(crd['LONG'].values, crd['LAT'].values)
# # m2.scatter(x2, y2, c=eva0[0]['NSE'], cmap='seismic', alpha=0.9, zorder=2)
# m2.scatter(x2, y2, c=time_avg_slope_w1[0], cmap='seismic', alpha=0.9, zorder=2)
# # m.scatter(x, y, c=delta_nse_daymet, cmap='seismic', alpha=0.9, zorder=2)
# # cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# # cbar = plt.colorbar(shrink= 0.93,  orientation='horizontal', pad=0.05, extend = 'both')
# # cbar.ax.tick_params()
# cbar2 = plt.colorbar(ax=ax2, shrink= 0.93,  orientation='horizontal', pad=0.05)
# # ax2.set_title(f'(b) NSE for Model Trained on Multiple Forcings', size=12)
# ax2.set_title(f'(b) Gradients of Slope associated to Weight of Maurer', size=12)
# cbar2.ax.tick_params(labelsize=10)
#
# ax3 = axes[2]
# m3 = Basemap(ax=ax3, llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
#             projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
#             resolution='c', area_thresh=10000)
# m3.drawmapboundary(fill_color='#46bcec')
# m3.fillcontinents(color='#524c4c', lake_color='#46bcec')
# m3.drawcoastlines()
# x3, y3 = m2(crd['LONG'].values, crd['LAT'].values)
# # m3.scatter(x3, y3, c=delta_nse_daymet, cmap='seismic', alpha=0.9, zorder=2, vmin=-0.5, vmax=-0.5)
# m3.scatter(x3, y3, c=time_avg_slope_w2[0], cmap='seismic', alpha=0.9, zorder=2, vmin=-0.5, vmax=-0.5)
# # m.scatter(x, y, c=delta_nse_daymet, cmap='seismic', alpha=0.9, zorder=2)
# # cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cbar3 = plt.colorbar(ax=ax3, shrink= 0.93,  orientation='horizontal', pad=0.05, extend = 'both')
# # cbar2 = plt.colorbar(shrink= 0.93,  orientation='horizontal', pad=0.05)
# # ax3.set_title(f'(c) \u0394NSE ', size=12)
# ax3.set_title(f'(c) Gradients of Slope associated to Weight of NLDAS2', size=12)
# cbar3.ax.tick_params(labelsize=10)
#
# plt.tight_layout()
# plt.show()

# to plot prcp weights spatially
# m = Basemap(projection='merc', llcrnrlon=crd['LONG'].min()-0.1, llcrnrlat=crd['LAT'].min()-0.1, urcrnrlon=crd['LONG'].max()+0.1, urcrnrlat=crd['LAT'].max()+0.1, resolution='c')
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
# cbar_axs = [fig.add_axes([0.1, 0.1, 0.8, 0.02]) for _ in range(4)]

cmap=plt.cm.get_cmap('CMRmap').reversed()
m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
            projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
            resolution='c', area_thresh=10000)

axs[0,0].set_title(f"(a) w\u2080 (weight associated to Daymet)", size=12)
m.ax = axs[0, 0]
m.etopo(scale=0.5, alpha=0.5)
m.drawcoastlines()
m.drawcountries(linewidth=1)
m.drawstates(linewidth=0.4)

x, y = m(crd['LONG'].values, crd['LAT'].values)

scatter = m.scatter(x, y, c=time_avg_w0['w0'], cmap=cmap, alpha=0.9, zorder=2, s=20)
# scatter = m.scatter(x, y, c=time_avg_slope_wo[0], cmap='seismic', alpha=0.9, zorder=2)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
cax1 = fig.add_axes([0.13, 0.52, 0.35, 0.02])
cbar = plt.colorbar(scatter, cax=cax1, shrink=0.9, orientation='horizontal', pad=0.05)
# cbar.set_label(f'w0(daymet), loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
cbar.ax.tick_params(labelsize=10)
# plt.title(f'(a) w\u2080 (weight associated to Daymet)', size=12)


# plt.show()

# to plot prcp weights spatially
# m = Basemap(projection='merc', llcrnrlon=crd['LONG'].min()-0.1, llcrnrlat=crd['LAT'].min()-0.1, urcrnrlon=crd['LONG'].max()+0.1, urcrnrlat=crd['LAT'].max()+0.1, resolution='c')
# m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
#             projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
#             resolution='c', area_thresh=10000)
axs[0,1].set_title(f'(b) w\u2081 (weight associated to Maurer)', size=12)
m.ax = axs[0, 1]
m.etopo(scale=0.5, alpha=0.5)
m.drawcoastlines()
m.drawcountries(linewidth=1)
m.drawstates(linewidth=0.4)

x, y = m(crd['LONG'].values, crd['LAT'].values)

scatter = m.scatter(x, y, c=time_avg_w1['w1'], cmap=cmap, alpha=0.9, zorder=2, s=20)
# scatter = m.scatter(x, y, c=time_avg_slope_w1[0], cmap='seismic', alpha=0.9, zorder=2)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
cax2 = fig.add_axes([0.55, 0.52, 0.35, 0.02])
cbar = plt.colorbar(scatter, cax=cax2, shrink=0.9, orientation='horizontal', pad=0.05)
# cbar.set_label(f'w1(maurer), loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
cbar.ax.tick_params(labelsize=10)
# plt.title(f'(b) w\u2081 (weight associated to Maurer)', size=12)

# plt.show()
# plt.savefig('/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_withloss/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/w0.png', bbox_inches = 'tight')

# to plot prcp weights spatially
# m = Basemap(projection='merc', llcrnrlon=crd['LONG'].min()-0.1, llcrnrlat=crd['LAT'].min()-0.1, urcrnrlon=crd['LONG'].max()+0.1, urcrnrlat=crd['LAT'].max()+0.1, resolution='c')
# m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
#             projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
#             resolution='c', area_thresh=10000)
axs[1,0].set_title(f'(c) w\u2082 (weight associated to NLDAS2)', size=12)
m.ax = axs[1, 0]
m.etopo(scale=0.5, alpha=0.5)
m.drawcoastlines()
m.drawcountries(linewidth=1)
m.drawstates(linewidth=0.4)

x, y = m(crd['LONG'].values, crd['LAT'].values)

scatter = m.scatter(x, y, c=time_avg_w2['w2'], cmap=cmap, alpha=0.9, zorder=2, s=20)
# scatter = m.scatter(x, y, c=time_avg_slope_w2[0], cmap='seismic', alpha=0.9, zorder=2)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
cax3 = fig.add_axes([0.13, 0.1, 0.35, 0.02])
cbar = plt.colorbar(scatter, cax=cax3, shrink=0.9, orientation='horizontal', pad=0.05)
# cbar.set_label(f'w2(nldas), loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
# plt.title(f'(c) w\u2082 (weight associated to NLDAS2)', size=12)
cbar.ax.tick_params(labelsize=10)

# plt.show()

# m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
#             projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
#             resolution='c', area_thresh=10000)
axs[1,1].set_title(f'(d) Sum of Weights (w\u2080 + w\u2081 + w\u2082)', size=12)
m.ax = axs[1, 1]
m.etopo(scale=0.5, alpha=0.5)
m.drawcoastlines()
m.drawcountries(linewidth=1)
m.drawstates(linewidth=0.4)

x, y = m(crd['LONG'].values, crd['LAT'].values)

scatter = m.scatter(x, y, c=time_avg_wsum[0], cmap='seismic', alpha=0.9, zorder=2, s=20)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
cax4 = fig.add_axes([0.55, 0.1, 0.35, 0.02])
cbar = plt.colorbar(scatter, cax=cax4, shrink=0.9, orientation='horizontal', pad=0.05)
# cbar.set_label(f'wsum, loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
# plt.title(f'Sum of Weights (w\u2080 + w\u2081 + w\u2082)')
cbar.ax.tick_params(labelsize=10)

# plt.tight_layout()
plt.show()
# plt.show()

diff_wghts = time_avg_w0['w0'] - time_avg_w2['w2']
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

m.scatter(x, y, c=diff_wghts, cmap='seismic', alpha=0.9, zorder=2)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
cbar = plt.colorbar(shrink=0.92, orientation='horizontal', pad=0.05)
cbar.set_label(f'w0(daymet) - w2(nldas), loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
cbar.ax.tick_params(labelsize=7)
plt.show()




# crd = pd.read_csv("/data/kas7897/dPLHBVrelease/hydroDL-dev/gages_list.csv")
prcp_gagewise_daymet = np.zeros(len(crd))
prcp_gagewise_maurer = np.zeros(len(crd))
prcp_gagewise_nldas = np.zeros(len(crd))
prcp_gagewise_fused = np.zeros(len(crd))

eff_wght_daymet = np.zeros(len(crd))
eff_wght_maurer = np.zeros(len(crd))
eff_wght_nldas = np.zeros(len(crd))

std_daymet = np.zeros(len(crd))
std_maurer = np.zeros(len(crd))
std_nldas = np.zeros(len(crd))
std_fused = np.zeros(len(crd))

acf_daymet = np.zeros(len(crd))
acf_maurer = np.zeros(len(crd))
acf_nldas = np.zeros(len(crd))
acf_fused = np.zeros(len(crd))

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

    prcp_daymet = pd.read_csv(prcp_daymet_dir, sep=r'\s+', header=None, skiprows=4)[5][274:9405].reset_index(drop=True)  #slicing from 1980 to 2004
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


    # prcp_daymet_mean =  np.mean(prcp_daymet)
    # prcp_maurer_mean =  np.mean(prcp_maurer)
    # prcp_nldas_mean =  np.mean(prcp_nldas)

    fused = prcp_daymet*w0[i] + prcp_maurer*w1[i] + prcp_nldas*w2[i]
    prcp_gagewise_daymet[i] =  np.mean(prcp_daymet)
    prcp_gagewise_maurer[i] =  np.mean(prcp_maurer)
    prcp_gagewise_nldas[i] =  np.mean(prcp_nldas)
    prcp_gagewise_fused[i] =  np.mean(fused)

    std_daymet[i] = np.std(prcp_daymet)
    std_maurer[i] = np.std(prcp_maurer)
    std_nldas[i] = np.std(prcp_nldas)
    std_fused[i] = np.std(fused)

    acf_daymet[i] = acf(prcp_daymet, nlags=1)[1]
    acf_maurer[i] = acf(prcp_maurer, nlags=1)[1]
    acf_nldas[i] = acf(prcp_nldas, nlags=1)[1]
    acf_fused[i] = acf(fused, nlags=1)[1]

    eff_wght_daymet[i] = np.sum(prcp_daymet * w0[i]) / np.sum(prcp_daymet)
    eff_wght_maurer[i] = np.sum(prcp_maurer * w1[i]) / np.sum(prcp_maurer)
    eff_wght_nldas[i] = np.sum(prcp_nldas * w2[i]) / np.sum(prcp_maurer)



group1 = [prcp_gagewise_fused, prcp_gagewise_daymet, prcp_gagewise_maurer, prcp_gagewise_nldas]
group2 = [std_fused, std_daymet, std_maurer, std_nldas]
# group3 = [acf_ghcn_3day, acf_multi_3day, acf_daymet_3day, acf_maurer_3day, acf_nldas_3day]
group3 = [acf_fused, acf_daymet, acf_maurer, acf_nldas]

fig, ax = plt.subplots(3, 1, figsize=(5, 12))
fig.subplots_adjust(hspace=0.4)
colors = ['red', 'blue', 'green', 'yellow', 'violet']
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
# tick_positions = [j + offset for offset in offsets for j in range(5)]
tick_labels = [
    "Fused P", "Daymet", "Maurer", "NLDAS"
]

# ax[0].set_xticks(tick_positions)
ax[0].set_xticklabels(tick_labels)  # rotated for better readability

ax[0].set_title("(a)Mean", size=12)
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
    "Fused P", "Daymet", "Maurer", "NLDAS"
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
    "Fused P", "Daymet", "Maurer", "NLDAS"
]

# ax[2].set_xticks(tick_positions)
ax[2].set_xticklabels(tick_labels)  # rotated for better readability
ax[2].set_title("(c)Autocorrelation (lag=1)", size=12)

# ax.set_title("Precipitation Statistics")
# plt.tight_layout()

plt.savefig('/home/kas7897/final_plots_fusion_paper/prcp_stats_box_plots.png',  dpi=300, bbox_inches='tight')
plt.show()




fig, axs = plt.subplots(3, 2, figsize=(10,8))
fig.subplots_adjust(hspace=0.5)

m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
            projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
            resolution='c', area_thresh=10000)

axs[0,0].set_title(f'(a) NSE of Fusion-δHBV (trained on Multiple P)', size=12)
m.ax = axs[0,0]
m.etopo(scale=0.5, alpha=0.5)
m.drawcoastlines()
m.drawcountries(linewidth=1)
m.drawstates(linewidth=0.4)

x, y = m(crd['LONG'].values, crd['LAT'].values)

scatter = m.scatter(x, y, c=eva0[0]['NSE'], cmap='seismic', alpha=0.9, zorder=2, vmin = 0.0, s=20)
# cax1 = fig.add_axes([0.13, 0.25, 0.35, 0.02])
# cax1 = fig.add_axes([0.13, 0.70, 0.35, 0.02])
cax1 = fig.add_axes([0.125, 0.65, 0.35, 0.02])
cbar = plt.colorbar(scatter, cax=cax1, shrink=0.9, orientation='horizontal', pad=0.05, extend='min')
cbar.ax.tick_params(labelsize=10)

axs[0,1].set_title(f'(b) \u0394NSE i.e NSE(Fusion-δHBV) minus NSE(δHBV)',size=12)
m.ax = axs[0,1]
m.etopo(scale=0.5, alpha=0.5)
m.drawcoastlines()
m.drawcountries(linewidth=1)
m.drawstates(linewidth=0.4)

x, y = m(crd['LONG'].values, crd['LAT'].values)

scatter = m.scatter(x, y, c=delta_nse_daymet, cmap='seismic', alpha=0.9, zorder=2, vmin=-0.35, vmax=0.35,s=20)
# cax2 = fig.add_axes([0.55, 0.25, 0.35, 0.02])
# cax2 = fig.add_axes([0.55, 0.70, 0.35, 0.02])
cax2 = fig.add_axes([0.55, 0.65, 0.35, 0.02])

cbar = plt.colorbar(scatter, cax=cax2, shrink=0.9, orientation='horizontal', pad=0.05, extend='both')

cbar.ax.tick_params(labelsize=10)

axs[1,0].set_title(r"(c) Effective w'$_{Daymet}$", size=12)
m.ax = axs[1, 0]
m.etopo(scale=0.5, alpha=0.5)
m.drawcoastlines()
m.drawcountries(linewidth=1)
m.drawstates(linewidth=0.4)

x, y = m(crd['LONG'].values, crd['LAT'].values)

scatter = m.scatter(x, y, c=eff_wght_daymet, cmap=cmap, alpha=0.9, zorder=2, s=20)
# scatter = m.scatter(x, y, c=time_avg_slope_wo[0], cmap='seismic', alpha=0.9, zorder=2)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cax3 = fig.add_axes([0.13, 0.52, 0.35, 0.02])
# cax3 = fig.add_axes([0.13, 0.48, 0.35, 0.02])
cax3 = fig.add_axes([0.125, 0.37, 0.35, 0.02])

cbar = plt.colorbar(scatter, cax=cax3, shrink=0.9, orientation='horizontal', pad=0.05)
# cbar.set_label(f'w0(daymet), loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
cbar.ax.tick_params(labelsize=10)
# plt.title(f'(a) w\u2080 (weight associated to Daymet)', size=12)



axs[1,1].set_title(r"(d) Effective w'$_{Maurer}$", size=12)
m.ax = axs[1, 1]
m.etopo(scale=0.5, alpha=0.5)
m.drawcoastlines()
m.drawcountries(linewidth=1)
m.drawstates(linewidth=0.4)

x, y = m(crd['LONG'].values, crd['LAT'].values)

scatter = m.scatter(x, y, c=eff_wght_maurer, cmap=cmap, alpha=0.9, zorder=2, s=20)

cax4 = fig.add_axes([0.55, 0.37, 0.35, 0.02])

cbar = plt.colorbar(scatter, cax=cax4, shrink=0.9, orientation='horizontal', pad=0.05)
cbar.ax.tick_params(labelsize=10)


axs[2,0].set_title(r"(e) Effective w'$_{NLDAS}$", size=12)
m.ax = axs[2, 0]
m.etopo(scale=0.5, alpha=0.5)
m.drawcoastlines()
m.drawcountries(linewidth=1)
m.drawstates(linewidth=0.4)

x, y = m(crd['LONG'].values, crd['LAT'].values)

scatter = m.scatter(x, y, c=eff_wght_nldas, cmap=cmap, alpha=0.9, zorder=2, s=20)
# scatter = m.scatter(x, y, c=time_avg_slope_w2[0], cmap='seismic', alpha=0.9, zorder=2)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cax5 = fig.add_axes([0.13, 0.26, 0.35, 0.02])
cax5 = fig.add_axes([0.125, 0.08, 0.35, 0.02])
cbar = plt.colorbar(scatter, cax=cax5, shrink=0.9, orientation='horizontal', pad=0.05)
# cbar.set_label(f'w2(nldas), loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
# plt.title(f'(c) w\u2082 (weight associated to NLDAS2)', size=12)
cbar.ax.tick_params(labelsize=10)

# plt.show()

# m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
#             projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
#             resolution='c', area_thresh=10000)
axs[2,1].set_title(r"(f) Sum of Effective Weights (w'$_{Daymet}$ + w'$_{Maurer}$ + w'$_{NLDAS}$)", size=12)
m.ax = axs[2, 1]
m.etopo(scale=0.5, alpha=0.5)
m.drawcoastlines()
m.drawcountries(linewidth=1)
m.drawstates(linewidth=0.4)

x, y = m(crd['LONG'].values, crd['LAT'].values)

scatter = m.scatter(x, y, c=(eff_wght_daymet+eff_wght_maurer+eff_wght_nldas), vmin = 0.92, vmax=1.08, cmap='seismic', alpha=0.9, zorder=2, s=20)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cax6 = fig.add_axes([0.55, 0.1, 0.35, 0.02])
# cax6 = fig.add_axes([0.55, 0.26, 0.35, 0.02])
cax6 = fig.add_axes([0.55, 0.08, 0.35, 0.02])

cbar = plt.colorbar(scatter, cax=cax6, shrink=0.9, orientation='horizontal', pad=0.05, extend='both')
# cbar.set_label(f'wsum, loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
# plt.title(f'Sum of Weights (w\u2080 + w\u2081 + w\u2082)')
cbar.ax.tick_params(labelsize=10)
plt.savefig('/home/kas7897/final_plots_fusion_paper/nse_eff_wght.png', dpi=300, bbox_inches='tight')

# plt.tight_layout()
plt.show()




for gage in gages:

    huc = crd.loc[crd['gage'] == int(gage), 'huc'].values[0]
    if len(str(huc)) == 1:
        huc = '0' + str(huc)
    prcp_daymet_dir = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/daymet/{huc}/{gage}_lump_cida_forcing_leap.txt"
    prcp_daymet = pd.read_csv(prcp_daymet_dir, sep=r'\s+', header=None, skiprows=4)[274:9405].reset_index(drop=True)

    # print(time_avg_w0)
    # k = time_avg_w0[0:]
    # print(time_avg.shape()
    # print(crd)
    k=1
    prcp_index = crd.loc[crd['gage'] == int(gage)].index[0]
    w0_filtered = w0[prcp_index]
    w1_filtered = w1[prcp_index]
    w2_filtered = w2[prcp_index]
    w0_filtered[prcp_daymet==0] = np.nan
    w1_filtered[prcp_daymet==0] = np.nan
    w2_filtered[prcp_daymet==0] = np.nan



    #to plot loss streamflow vs prcp
    # sf = loss[6].astype('str').str.extractall('(\d+)').unstack().fillna('').sum(axis=1).astype(float)/1000
    # prcp = loss[9].astype('str').str.extractall('(\d+)').unstack().fillna('').sum(axis=1).astype(float)/1000
    #
    #
    # plt.plot(prcp, label='precipitation loss')
    # plt.plot(sf, label='streamflow_loss')
    # plt.title('loss_factor=15')
    # plt.legend()
    # plt.show()



    #to plot prcp weight temporally


    # wsum_filtered = w0_filtered + w1_filtered + w2_filtered
    #rolling mean
    w0_filtered_rm = w0_filtered.rolling(window=rm_window).mean()
    w1_filtered_rm = w1_filtered.rolling(window=rm_window).mean()
    w2_filtered_rm = w2_filtered.rolling(window=rm_window).mean()
    # wsum_filtered_rm = wsum_filtered.rolling(window=rm_window).mean()

    fig = plt.figure(figsize=(12, 6))
    #
    plt.plot(w0_filtered_rm, label='w0_non0(daymet)')
    plt.plot(w1_filtered_rm, label='w1_non0(maurer)')
    plt.plot(w2_filtered_rm, label='w2_non0(nldas)')
    # plt.plot(wsum_filtered, label='wsum_non0')
    plt.title(f'{gage}_with rolling mean {rm_window} days; loss factor: {lossfactor}; smoothing factor: {smoothfactor}')
    # # Add a legend
    plt.legend(loc='upper right')
    plt.show()
    # plt.savefig()
    # plt.savefig('/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_withloss/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/20gage.png', bbox_inches = 'tight')

    # Add axis labels and a title
    # plt.title(crd)