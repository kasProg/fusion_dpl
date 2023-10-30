import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


lossfactor = 0
smoothfactor = 0
expno = 2


w0 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/daymetwithloss{lossfactor}smooth{smoothfactor}_{4}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts1.csv',
    header=None)
w1 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/maurer_extendedwithloss{lossfactor}smooth{smoothfactor}_{expno}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts1.csv',
    header=None)
w2 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/nldas_extendedwithloss{lossfactor}smooth{smoothfactor}_{expno}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts1.csv',
    header=None)


# Load CAMELS basins CSV file
camels_df = gpd.read_file('/data/kas7897/dPLHBVrelease/hydroDL-dev/gages_list.csv')
camels_df.crs = "EPSG:4326"
camels_df['geometry'] = camels_df.apply(lambda row: Point(float(row['LONG']), float(row['LAT'])), axis=1)

# Load US region shapefile
# us_regions = gpd.read_file("/data/kas7897/cb_2019_us_region_5m/cb_2019_us_region_5m.shp")
# us_regions = gpd.read_file("/data/kas7897/cb_2018_us_division_500k/cb_2018_us_division_500k.shp")
us_regions = gpd.read_file("/data/kas7897/US_physio_shp/physio.shp")
camels_df = camels_df.to_crs(us_regions.crs)

basins_with_regions = gpd.sjoin(camels_df, us_regions, how='left', op='within')
print(basins_with_regions)
# Northeast, South, Midwest, West
# New England, Middle Atlantic, East North Central, West North Central, South Atlantic, Mountain, Pacific, East South Central, West South Central
region = 'Pacific'

# camels_region = basins_with_regions[basins_with_regions['SECTION']=='FLORIDIAN'  basins_with_regions['SECTION']=='SEA ISLAND'].reset_index(drop=True)
camels_region = basins_with_regions[(basins_with_regions['SECTION'] == 'FLORIDIAN') | (basins_with_regions['SECTION'] == 'SEA ISLAND')].reset_index(drop=True)
# camels_region = basins_with_regions[(basins_with_regions['SECTION'] == 'GREAT BASIN') | (basins_with_regions['SECTION'] == 'PAYETTE')|
#                                     (basins_with_regions['SECTION'] == 'HARNEY') | (basins_with_regions['SECTION'] == 'SNAKE RIVER PLAIN')|
#                                     (basins_with_regions['PROVINCE'] == 'MIDDLE ROCKY MOUNTAINS')| (basins_with_regions['SECTION'] == 'HIGH PLATEAUS OF UTAH')].reset_index(drop=True)
# camels_region = basins_with_regions[(basins_with_regions['SECTION'] == 'CALIFORNIA COAST RANGES') | (basins_with_regions['SECTION'] == 'LOS ANGELES RANGES')| (basins_with_regions['SECTION'] == 'KLAMATH MOUNTAINS')].reset_index(drop=True)

mean_daymet = np.zeros(len(camels_region))
eff_mean_daymet = np.zeros(len(camels_region))
mean_nldas = np.zeros(len(camels_region))
eff_mean_nldas = np.zeros(len(camels_region))


for i in range(len(camels_region)):
    gage = camels_region['gage'][i]
    huc = camels_region['huc'][i]
    if len(str(huc)) == 1:
        huc = '0' + str(huc)

    prcp_daymet_dir = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/daymet/{huc}/{gage}_lump_cida_forcing_leap.txt"
    prcp_maurer_dir = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/maurer_extended/{huc}/{gage}_lump_maurer_forcing_leap.txt"
    prcp_nldas_dir = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/nldas_extended/{huc}/{gage}_lump_nldas_forcing_leap.txt"

    prcp_daymet = pd.read_csv(prcp_daymet_dir, sep=r'\s+', header=None, skiprows=4)[274:9405].reset_index(drop=True)  # slicing from 1980 to 2004
    prcp_maurer = pd.read_csv(prcp_maurer_dir, sep=r'\s+', header=None, skiprows=4)[274:9405].reset_index(drop=True)
    prcp_nldas = pd.read_csv(prcp_nldas_dir, sep=r'\s+', header=None, skiprows=4)[274:9405].reset_index(drop=True)

    mean_yearly_daymet = prcp_daymet.groupby(0).sum()
    mean_yearly_nldas = prcp_nldas.groupby(0).sum()
    mean_daymet[i] = np.mean(mean_yearly_daymet[5])
    mean_nldas[i] = np.mean(mean_yearly_nldas[5])

    prcp_daymet_copy = prcp_daymet.copy()
    prcp_nldas_copy = prcp_nldas.copy()
    prcp_index = camels_df.loc[camels_df['gage'] == gage].index[0]
    prcp_daymet_copy[5] = w0[prcp_index]*prcp_daymet[5]
    prcp_nldas_copy[5] = w2[prcp_index]*prcp_nldas[5]

    eff_mean_yearly_daymet = prcp_daymet_copy.groupby(0).sum()
    eff_mean_yearly_nldas = prcp_nldas_copy.groupby(0).sum()
    eff_mean_daymet[i] = np.mean(eff_mean_yearly_daymet[5])
    eff_mean_nldas[i] = np.mean(eff_mean_yearly_nldas[5])






k=1
# fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# box_props = dict(whiskers=dict(color='r', linestyle='--'),
#                  medians=dict(color='b', linestyle='-'),
#                  fliers=dict(marker='o', markersize=5, markerfacecolor='r'),
#                  boxes=dict(facecolor='lightblue', color='black', linewidth=2))
# boxprops = dict(facecolor='lightblue', color='black', linewidth=2)
# eff_boxprops = dict(facecolor='lightgreen', color='black', linewidth=2)
#
# # plt.boxplot([mean_daymet, eff_mean_daymet, mean_nldas, eff_mean_nldas], positions=[1, 1.5, 2.5, 3], widths=0.3, boxprops=[boxprops, eff_boxprops, boxprops, eff_boxprops],
# #             whiskerprops=dict(color='r', linestyle='--'),
# #             medianprops=dict(color='b', linestyle='-'),
# #             flierprops=dict(marker='o', markersize=5, markerfacecolor='r'))\


# plt.boxplot([mean_daymet, eff_mean_daymet, mean_nldas, eff_mean_nldas], positions=[1, 1.5, 2.5, 3], widths=0.2)
# # plt.xlabel('Data')
# plt.ylabel('Mean of Total-Annual Precipitation (mm/year)', size = 20)
# # plt.title('South-East Coast')
# plt.title('West (Around the Great Basin)', size =25)
# # plt.title('West Coast')
# plt.xticks([1, 1.5, 2.5, 3], ['Daymet', 'Applied Daymet', 'NLDAS-2', 'Effective NLDAS-2'], size=20)
# #plt.grid(True)  # Add gridlines
#
# # Customize the box colors
# box_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']  # Define custom box colors
# for box, color in zip(plt['boxes'], box_colors):
#     box.set(color='black', linewidth=2)  # Set box outline color and width
#     box.set(facecolor=color)  # Set box fill color
# #
# plt.show()



# Perform the boxplot and store the result
boxplot_dict = plt.boxplot([mean_daymet, eff_mean_daymet, mean_nldas, eff_mean_nldas], positions=[1, 1.5, 2.5, 3], widths=0.2, patch_artist=True)

# Customize the plot
plt.ylabel('Mean of Total-Annual Precipitation (mm/year)', fontsize=20)
# plt.title('West (Around the Great Basin)', fontsize=25)
# plt.title('West Coast', fontsize=25)
plt.title('South-East Coast', fontsize=25)

plt.xticks([1, 1.5, 2.5, 3], ['Daymet', 'Applied\nDaymet', 'NLDAS2', 'Applied\nNLDAS2'], fontsize=20)
plt.yticks(fontsize=15)
plt.grid(True)  # Add gridlines

# Customize the box colors
box_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']  # Define custom box colors
for box, color in zip(boxplot_dict['boxes'], box_colors):
    box.set(color='black', linewidth=2)  # Set box outline color and width
    box.set(facecolor=color)  # Set box fill color

plt.show()
