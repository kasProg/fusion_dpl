import numpy as np
import pandas as pd
# import geopandas as gpd
from mpl_toolkits.basemap import Basemap, addcyclic
import matplotlib.pyplot as plt
import numpy as np
import json
lossfactor = 23
smoothfactor = 0
# expno =4
crd = pd.read_csv("/data/kas7897/dPLHBVrelease/hydroDL-dev/gages_list.csv")
rm_window = 1
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
w0 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts1.csv',
    header=None)
slope_w0 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/{index1}_grad_daymet.csv',
    header=None)
elev_w0 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/{index2}_grad_daymet.csv',
    header=None)
w1 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts2.csv',
    header=None)
slope_w1 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/{index1}_grad_maurer.csv',
    header=None)
elev_w1 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/{index2}_grad_maurer.csv',
    header=None)
w2 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts3.csv',
    header=None)
slope_w2 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/{index1}_grad_nldas.csv',
    header=None)
elev_w2 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/{index2}_grad_nldas.csv',
    header=None)

time_avg_w0 = pd.DataFrame(np.mean(w0))
time_avg_slope_wo = pd.DataFrame(np.mean(slope_w0))
time_avg_elev_wo = pd.DataFrame(np.mean(elev_w0))
time_avg_slope_w1 = pd.DataFrame(np.mean(slope_w1))
time_avg_elev_w1 = pd.DataFrame(np.mean(elev_w1))
time_avg_slope_w2 = pd.DataFrame(np.mean(slope_w2))
time_avg_elev_w2 = pd.DataFrame(np.mean(elev_w2))
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


fig, axs = plt.subplots(3, 3, figsize=(20, 16))
# cbar_axs = [fig.add_axes([0.1, 0.1, 0.8, 0.02]) for _ in range(4)]


m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
            projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
            resolution='c', area_thresh=10000)
# cax1 = fig.add_axes([0.12, 0.32, 0.35, 0.02])
# cax1 = fig.add_axes([0.12, 0.32, 0.35, 0.02])
# cax2 = fig.add_axes([0.55, 0.32, 0.35, 0.02])
# cax3 = fig.add_axes([0.12, 0.1, 0.35, 0.02])
# cax4 = fig.add_axes([0.55, 0.1, 0.35, 0.02])
# cax5 = fig.add_axes([0.12, 0.1, 0.35, 0.02])
# cax6 = fig.add_axes([0.55, 0.1, 0.35, 0.02])
# cax7 = fig.add_axes([0.12, 0.1, 0.35, 0.02])
# cax8 = fig.add_axes([0.55, 0.1, 0.35, 0.02])
# cax9 = fig.add_axes([0.12, 0.1, 0.35, 0.02])

axs[0,0].set_title(f'(a) {inputsLst[index1]} Gradients to Weight (Daymet)', size=17)
m.ax = axs[0, 0]
m.etopo(scale=0.5, alpha=0.5)
m.drawcoastlines()
m.drawcountries(linewidth=1)
m.drawstates(linewidth=0.4)
x, y = m(crd['LONG'].values, crd['LAT'].values)
# scatter = m.scatter(x, y, c=time_avg_w0['w0'], cmap='seismic', alpha=0.9, zorder=2)
scatter = m.scatter(x, y, c=time_avg_slope_wo[0], cmap='seismic', alpha=0.9, zorder=2, s=20)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cax1 = fig.add_axes([0.13, 0.52, 0.35, 0.02])
cax1= fig.add_axes([axs[0, 0].get_position().x0, axs[0, 0].get_position().y0 - 0.02, axs[0, 0].get_position().width, 0.02])
cbar = plt.colorbar(scatter, cax=cax1, shrink=0.9, orientation='horizontal', pad=0.05)
# cbar.set_label(f'w0(daymet), loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
cbar.ax.tick_params(labelsize=14)
# plt.title(f'(a) w\u2080 (weight associated to Daymet)', size=12)



axs[0,1].set_title(f'(b) {inputsLst[index2]} Gradients to Weight (Daymet)', size=17)
m.ax = axs[0, 1]
m.etopo(scale=0.5, alpha=0.5)
m.drawcoastlines()
m.drawcountries(linewidth=1)
m.drawstates(linewidth=0.4)
x, y = m(crd['LONG'].values, crd['LAT'].values)
# scatter = m.scatter(x, y, c=time_avg_w0['w0'], cmap='seismic', alpha=0.9, zorder=2)
scatter = m.scatter(x, y, c=time_avg_elev_wo[0], cmap='seismic', alpha=0.9, zorder=2, s=20)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cax2 = fig.add_axes([0.13, 0.52, 0.35, 0.02])
cax2 = fig.add_axes([axs[0, 1].get_position().x0, axs[0, 1].get_position().y0 - 0.02, axs[0, 1].get_position().width, 0.02])
cbar = plt.colorbar(scatter, cax=cax2, shrink=0.9, orientation='horizontal', pad=0.05)
# cbar.set_label(f'w0(daymet), loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
cbar.ax.tick_params(labelsize=14)


axs[0,2].set_title(f'(c) w\u2080 (weight associated to Daymet)', size=17)
m.ax = axs[0, 2]
m.etopo(scale=0.5, alpha=0.5)
m.drawcoastlines()
m.drawcountries(linewidth=1)
m.drawstates(linewidth=0.4)
x, y = m(crd['LONG'].values, crd['LAT'].values)
scatter = m.scatter(x, y, c=time_avg_w0['w0'], cmap='seismic', alpha=0.9, zorder=2, s=20)
# scatter = m.scatter(x, y, c=time_avg_slope_w1[0], cmap='seismic', alpha=0.9, zorder=2)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cax3 = fig.add_axes([0.55, 0.52, 0.35, 0.02])
cax3= fig.add_axes([axs[0, 2].get_position().x0, axs[0, 2].get_position().y0 - 0.02, axs[0, 2].get_position().width, 0.02])
cbar = plt.colorbar(scatter, cax=cax3, shrink=0.9, orientation='horizontal', pad=0.05)
# cbar.set_label(f'w1(maurer), loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
cbar.ax.tick_params(labelsize=14)
# plt.title(f'(b) w\u2081 (weight associated to Maurer)', size=12)





axs[1,0].set_title(f'(d) {inputsLst[index1]} Gradients to Weight (Maurer)', size=17)
m.ax = axs[1, 0]
m.etopo(scale=0.5, alpha=0.5)
m.drawcoastlines()
m.drawcountries(linewidth=1)
m.drawstates(linewidth=0.4)
x, y = m(crd['LONG'].values, crd['LAT'].values)
# scatter = m.scatter(x, y, c=time_avg_w2['w2'], cmap='seismic', alpha=0.9, zorder=2)
scatter = m.scatter(x, y, c=time_avg_slope_w1[0], cmap='seismic', alpha=0.9, zorder=2, s=20)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cax4 = fig.add_axes([0.13, 0.1, 0.35, 0.02])
cax4= fig.add_axes([axs[1, 0].get_position().x0, axs[1, 0].get_position().y0 - 0.02, axs[1, 0].get_position().width, 0.02])
cbar = plt.colorbar(scatter, cax=cax4, shrink=0.9, orientation='horizontal', pad=0.05)
# cbar.set_label(f'w2(nldas), loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
# plt.title(f'(c) w\u2082 (weight associated to NLDAS2)', size=12)
cbar.ax.tick_params(labelsize=14)

axs[1,1].set_title(f'(e) {inputsLst[index2]} Gradients to Weight (Maurer)', size=17)
m.ax = axs[1, 1]
m.etopo(scale=0.5, alpha=0.5)
m.drawcoastlines()
m.drawcountries(linewidth=1)
m.drawstates(linewidth=0.4)
x, y = m(crd['LONG'].values, crd['LAT'].values)
# scatter = m.scatter(x, y, c=time_avg_w0['w0'], cmap='seismic', alpha=0.9, zorder=2)
scatter = m.scatter(x, y, c=time_avg_elev_w1[0], cmap='seismic', alpha=0.9, zorder=2, s=20)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cax5 = fig.add_axes([0.13, 0.52, 0.35, 0.02])
cax5= fig.add_axes([axs[1, 1].get_position().x0, axs[1, 1].get_position().y0 - 0.02, axs[1, 1].get_position().width, 0.02])
cbar = plt.colorbar(scatter, cax=cax5, shrink=0.9, orientation='horizontal', pad=0.05)
# cbar.set_label(f'w0(daymet), loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
cbar.ax.tick_params(labelsize=14)


axs[1,2].set_title(f'(f) w\u2081 (weight associated to Maurer)',size=17)
m.ax = axs[1, 2]
m.etopo(scale=0.5, alpha=0.5)
m.drawcoastlines()
m.drawcountries(linewidth=1)
m.drawstates(linewidth=0.4)
x, y = m(crd['LONG'].values, crd['LAT'].values)
scatter = m.scatter(x, y, c=time_avg_w1['w1'], cmap='seismic', alpha=0.9, zorder=2, s=20)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cax6 = fig.add_axes([0.55, 0.1, 0.35, 0.02])
cax6= fig.add_axes([axs[1, 2].get_position().x0, axs[1, 2].get_position().y0 - 0.02, axs[1, 2].get_position().width, 0.02])
cbar = plt.colorbar(scatter, cax=cax6, shrink=0.9, orientation='horizontal', pad=0.05)
# cbar.set_label(f'wsum, loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
# plt.title(f'Sum of Weights (w\u2080 + w\u2081 + w\u2082)')
cbar.ax.tick_params(labelsize=14)

# plt.tight_layout()


axs[2,0].set_title(f'(g) {inputsLst[index1]} Gradients to Weight (NLDAS2)', size=17)
m.ax = axs[2, 0]
m.etopo(scale=0.5, alpha=0.5)
m.drawcoastlines()
m.drawcountries(linewidth=1)
m.drawstates(linewidth=0.4)
x, y = m(crd['LONG'].values, crd['LAT'].values)
# scatter = m.scatter(x, y, c=time_avg_w2['w2'], cmap='seismic', alpha=0.9, zorder=2)
scatter = m.scatter(x, y, c=time_avg_slope_w2[0], cmap='seismic', alpha=0.9, zorder=2, s=20)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cax7 = fig.add_axes([0.13, 0.1, 0.35, 0.02])
cax7= fig.add_axes([axs[2, 0].get_position().x0, axs[2, 0].get_position().y0 - 0.02, axs[2, 0].get_position().width, 0.02])
cbar = plt.colorbar(scatter, cax=cax7, shrink=0.9, orientation='horizontal', pad=0.05)
# cbar.set_label(f'w2(nldas), loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
# plt.title(f'(c) w\u2082 (weight associated to NLDAS2)', size=12)
cbar.ax.tick_params(labelsize=14)


axs[2,1].set_title(f'(h) {inputsLst[index2]} Gradients to Weight (NLDAS2)', size=17)
m.ax = axs[2, 1]
m.etopo(scale=0.5, alpha=0.5)
m.drawcoastlines()
m.drawcountries(linewidth=1)
m.drawstates(linewidth=0.4)
x, y = m(crd['LONG'].values, crd['LAT'].values)
# scatter = m.scatter(x, y, c=time_avg_w0['w0'], cmap='seismic', alpha=0.9, zorder=2)
scatter = m.scatter(x, y, c=time_avg_elev_w2[0], cmap='seismic', alpha=0.9, zorder=2, s=20)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cax8 = fig.add_axes([0.13, 0.52, 0.35, 0.02])
cax8= fig.add_axes([axs[2, 1].get_position().x0, axs[2, 1].get_position().y0 - 0.02, axs[2, 1].get_position().width, 0.02])
cbar = plt.colorbar(scatter, cax=cax8, shrink=0.9, orientation='horizontal', pad=0.05)
# cbar.set_label(f'w0(daymet), loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
cbar.ax.tick_params(labelsize=14)



axs[2,2].set_title(f'(i) w\u2082 (weight associated to NLDAS2)',size=17)
m.ax = axs[2, 2]
m.etopo(scale=0.5, alpha=0.5)
m.drawcoastlines()
m.drawcountries(linewidth=1)
m.drawstates(linewidth=0.4)
x, y = m(crd['LONG'].values, crd['LAT'].values)
scatter = m.scatter(x, y, c=time_avg_w2['w2'], cmap='seismic', alpha=0.9, zorder=2, s=20)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cax9 = fig.add_axes([0.55, 0.1, 0.35, 0.02])
cax9= fig.add_axes([axs[2, 2].get_position().x0, axs[2, 2].get_position().y0 - 0.02, axs[2, 2].get_position().width, 0.02])
cbar = plt.colorbar(scatter, cax=cax9, shrink=0.9, orientation='horizontal', pad=0.05)
# cbar.set_label(f'wsum, loss factor {lossfactor}; smoothing factor {smoothfactor}', size=7)
# plt.title(f'Sum of Weights (w\u2080 + w\u2081 + w\u2082)')
cbar.ax.tick_params(labelsize=14)

# plt.title(f'loss_factor = {lossfactor}')
plt.savefig(f'/home/kas7897/final_plots_fusion_paper/gradients_{inputsLst[index1]}&{inputsLst[index2]}_loss{lossfactor}.png', dpi=300, bbox_inches='tight')

plt.show()



# from mpl_toolkits.basemap import Basemap
# import matplotlib.pyplot as plt
#
# # Create a figure with subplots
# fig, axs = plt.subplots(3, 2, figsize=(12, 18))
#
# # Basemap parameters
# m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
#             projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
#             resolution='c', area_thresh=10000)
#
# # Create scatter plots with Basemap
# scatter_params = [
#     (axs[0, 0], time_avg_slope_wo[0], 'Daymet', 'w0'),
#     (axs[0, 1], time_avg_w0['w0'], 'Daymet', 'w0'),
#     (axs[1, 0], time_avg_slope_w1[0], 'Maurer', 'w1'),
#     (axs[1, 1], time_avg_w1['w1'], 'Maurer', 'w1'),
#     (axs[2, 0], time_avg_slope_w2[0], 'Maurer', 'w2'),
#     (axs[2, 1], time_avg_w2['w2'], 'NLDAS2', 'w2')
# ]
#
# for ax, cmap_data, title, label in scatter_params:
#     m.ax = ax
#     m.drawmapboundary(fill_color='#46bcec')
#     m.fillcontinents(color='#524c4c', lake_color='#46bcec')
#     m.drawcoastlines()
#     x, y = m(crd['LONG'].values, crd['LAT'].values)
#     scatter = m.scatter(x, y, c=cmap_data, cmap='seismic', alpha=0.9, zorder=2)
#     cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0 - 0.02, ax.get_position().width, 0.02])
#     cbar = plt.colorbar(scatter, cax=cax, orientation='horizontal')
#     cbar.ax.tick_params(labelsize=14)
#     cbar.set_label(f'{label} ({title})', size=14)
#     ax.set_title(f'({label}) Slope Gradients associated to weight of {title}', size=12)
#
# # Adjust layout and show the plot
# # plt.tight_layout()
# plt.savefig('/data/kas7897/dPLHBVrelease/scatter_plots_with_basemap.png', dpi=300, bbox_inches='tight')
#
# plt.show()
