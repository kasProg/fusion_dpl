
import pandas as pd
# import geopandas as gpd
from mpl_toolkits.basemap import Basemap, addcyclic
import matplotlib.pyplot as plt
import numpy as np
crd = pd.read_csv("/data/kas7897/dPLHBVrelease/hydroDL-dev/gages_list.csv")
prcp_gagewise_daymet = np.zeros(len(crd))
prcp_gagewise_maurer = np.zeros(len(crd))
prcp_gagewise_nldas = np.zeros(len(crd))
# igage = 0
year = 2000
huc1='01'
huc2='18'
huc3='17'
gage1 = '01123000'
gage2 = '11274630'
gage3 = '14301000'
prcp_daymet_dir1 = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/daymet/{huc1}/{gage1}_lump_cida_forcing_leap.txt"
prcp_daymet_dir2 = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/daymet/{huc2}/{gage2}_lump_cida_forcing_leap.txt"
prcp_daymet_dir3 = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/daymet/{huc3}/{gage3}_lump_cida_forcing_leap.txt"
prcp_maurer_dir1 = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/maurer_extended/{huc1}/{gage1}_lump_maurer_forcing_leap.txt"
prcp_maurer_dir2 = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/maurer_extended/{huc2}/{gage2}_lump_maurer_forcing_leap.txt"
prcp_maurer_dir3 = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/maurer_extended/{huc3}/{gage3}_lump_maurer_forcing_leap.txt"
prcp_nldas_dir1 = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/nldas_extended/{huc1}/{gage1}_lump_nldas_forcing_leap.txt"
prcp_nldas_dir2 = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/nldas_extended/{huc2}/{gage2}_lump_nldas_forcing_leap.txt"
prcp_nldas_dir3 = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/nldas_extended/{huc3}/{gage3}_lump_nldas_forcing_leap.txt"

prcp_daymet1 = pd.read_csv(prcp_daymet_dir1, sep=r'\s+', header=None, skiprows=4)[5][:9131].reset_index(drop=True)
prcp_daymet2 = pd.read_csv(prcp_daymet_dir2, sep=r'\s+', header=None, skiprows=4)[5][:9131].reset_index(drop=True)
prcp_daymet3 = pd.read_csv(prcp_daymet_dir3, sep=r'\s+', header=None, skiprows=4)[5][:9131].reset_index(drop=True)
prcp_maurer1 = pd.read_csv(prcp_maurer_dir1, sep=r'\s+', header=None, skiprows=4)[5][:9131].reset_index(drop=True)
prcp_maurer2 = pd.read_csv(prcp_maurer_dir2, sep=r'\s+', header=None, skiprows=4)[5][:9131].reset_index(drop=True)
prcp_maurer3 = pd.read_csv(prcp_daymet_dir3, sep=r'\s+', header=None, skiprows=4)[5][:9131].reset_index(drop=True)
prcp_nldas1 = pd.read_csv(prcp_nldas_dir1, sep=r'\s+', header=None, skiprows=4)[5][:9131].reset_index(drop=True)
prcp_nldas2 = pd.read_csv(prcp_nldas_dir2, sep=r'\s+', header=None, skiprows=4)[5][:9131].reset_index(drop=True)
prcp_nldas3 = pd.read_csv(prcp_nldas_dir3, sep=r'\s+', header=None, skiprows=4)[5][:9131].reset_index(drop=True)

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

    prcp_daymet = pd.read_csv(prcp_daymet_dir, sep=r'\s+', header=None, skiprows=4)[5][:9131] #slicing from 1980 to 2004
    # prcp_daymet = pd.read_csv(prcp_daymet_dir, sep=r'\s+', header=None, skiprows=4) #slicing from 1980 to 2004
    # prcp_daymet = prcp_daymet[prcp_daymet[0] ==year]
    # prcp_daymet = prcp_daymet[5]

    prcp_maurer = pd.read_csv(prcp_maurer_dir, sep=r'\s+', header=None, skiprows=4)[5][:9131]
    # prcp_maurer = pd.read_csv(prcp_maurer_dir, sep=r'\s+', header=None, skiprows=4)
    # prcp_maurer = prcp_maurer[prcp_maurer[0] ==year]
    # prcp_maurer = prcp_maurer[5]

    prcp_nldas = pd.read_csv(prcp_nldas_dir, sep=r'\s+', header=None, skiprows=4)[5][:9131]
    # prcp_nldas = pd.read_csv(prcp_nldas_dir, sep=r'\s+', header=None, skiprows=4)
    # prcp_nldas = prcp_nldas[prcp_nldas[0] ==year]
    # prcp_nldas = prcp_nldas[prcp_nldas[0] ==year]


    # prcp_daymet_mean =  np.mean(prcp_daymet)
    # prcp_maurer_mean =  np.mean(prcp_maurer)
    # prcp_nldas_mean =  np.mean(prcp_nldas)

    prcp_gagewise_daymet[i] =  np.sum(prcp_daymet)
    prcp_gagewise_maurer[i] =  np.sum(prcp_maurer)
    prcp_gagewise_nldas[i] =  np.sum(prcp_nldas)
    # igage = igage + 1

#converting to mm/year
# prcp_gagewise_daymet = prcp_gagewise_daymet*365
# prcp_gagewise_maurer = prcp_gagewise_maurer*365
# prcp_gagewise_nldas = prcp_gagewise_nldas*365


m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
            projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
            resolution='c', area_thresh=10000)
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#524c4c', lake_color='#46bcec')
# m.fillcontinents(color='0.1')
m.drawcoastlines()

x, y = m(crd['LONG'].values, crd['LAT'].values)

m.scatter(x, y, c=prcp_gagewise_daymet, cmap='OrRd', alpha=0.9, zorder=2, vmax = 6*365)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
cbar = plt.colorbar(shrink=0.92, orientation='horizontal', pad=0.05, extend = 'max')
cbar.set_label(f'daymet 1980-2004 mean', size=7)
cbar.ax.tick_params(labelsize=7)
plt.show()

m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
            projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
            resolution='c', area_thresh=10000)
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#524c4c', lake_color='#46bcec')
# m.fillcontinents(color='0.1')
m.drawcoastlines()

x, y = m(crd['LONG'].values, crd['LAT'].values)

m.scatter(x, y, c=prcp_gagewise_maurer, cmap='OrRd', alpha=0.9, zorder=2, vmax =6*365)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
cbar = plt.colorbar(shrink=0.92, orientation='horizontal', pad=0.05, extend = 'max')
cbar.set_label(f'maurer 1980-2004 mean', size=7)
cbar.ax.tick_params(labelsize=7)
plt.show()

m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
            projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
            resolution='c', area_thresh=10000)
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#524c4c', lake_color='#46bcec')
# m.fillcontinents(color='0.1')
m.drawcoastlines()

x, y = m(crd['LONG'].values, crd['LAT'].values)

m.scatter(x, y, c=prcp_gagewise_nldas, cmap='OrRd', alpha=0.9, zorder=2, vmax = 6*365)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
cbar = plt.colorbar(shrink=0.92, orientation='horizontal', pad=0.05, extend ='max')
cbar.set_label(f'nldas 1980-2004 mean', size=7)
cbar.ax.tick_params(labelsize=7)
plt.show()

diff = prcp_gagewise_daymet - prcp_gagewise_nldas
print(min(diff))
print(max(diff))
m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
            projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
            resolution='c', area_thresh=10000)
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#524c4c', lake_color='#46bcec')
# m.fillcontinents(color='0.1')
m.drawcoastlines()

x, y = m(crd['LONG'].values, crd['LAT'].values)

m.scatter(x, y, c=diff, cmap='seismic', alpha=0.9, zorder=2, vmin = -5000, vmax = 5000)
# cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
cbar = plt.colorbar(shrink=0.92, orientation='horizontal', pad=0.05, extend='both')
cbar.set_label(f'total (daymet-nldas)', size=7)
cbar.ax.tick_params(labelsize=7)
plt.show()

lower_limit = -2000
upper_limit = 2000

# Count the number of values within the limits
num_within_limits = np.count_nonzero((diff >= lower_limit) & (diff <= upper_limit))

# Calculate the percentage of values within the limits
percentage_within_limits = num_within_limits / diff.size * 100

print(f"{percentage_within_limits:.2f}% of values are within the limits.")