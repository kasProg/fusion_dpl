import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
# import geopandas as gpd
import os
# from shapely.geometry import Point
from mpl_toolkits.basemap import Basemap, addcyclic
import matplotlib.pyplot as plt
import xarray as xr
import json
from statsmodels.tsa.stattools import acf
import matplotlib.lines as mlines



# true_prcp = xr.open_dataset("/data/kas7897/cpc_prcp/subset.nc")
# true_prcp = xr.open_dataset("/data/kas7897/cpc_prcp/1980.nc")
# #
# true_prcp_df = true_prcp.precip[:].to_dataframe().dropna()
# true_prcp_df = true_prcp_df.dropna()
# true_prcp_df = true_prcp_df.reset_index()
#
# true_coords = true_prcp_df.drop_duplicates(subset=['lat', 'lon'])[['lat', 'lon']]


folder_path = '/data/kas7897/camels_shapefiles_new'

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

# Load data from CSVs A and B into pandas DataFrames
crd_camels = pd.read_csv("/data/kas7897/dPLHBVrelease/hydroDL-dev/gages_list.csv")
# df_a = pd.read_csv('path/to/csv_a.csv')  # DataFrame A with columns: 'Latitude_A', 'Longitude_A', and 'Value_A'
# df_b = pd.read_csv('path/to/csv_b.csv')  # DataFrame B with columns: 'Latitude_B', 'Longitude_B', and 'Value_B'
crd_GHCN = pd.read_csv("/data/kas7897/GHCN_Data/Station_Data/Station_Metadata.csv")

# geometry = [Point(xy) for xy in zip(crd_GHCN['Longitude'], crd_GHCN['Latitude'])]
# crd_GHCN_geo = gpd.GeoDataFrame(crd_GHCN, crs='EPSG:4326', geometry=geometry)
# # Create an empty dictionary to store grouped coordinates
# grouped_coordinates = {}

# names = []
# for filename in os.listdir(folder_path):
#     if filename.endswith('.shp'):
#         # Read the shapefile
#         shapefile_path = os.path.join(folder_path, filename)
#         gdf_shapefile = gpd.read_file(shapefile_path)
#         # crd_GHCN_geo = gpd.GeoDataFrame(crd_GHCN, crs=gdf_shapefile.crs, geometry=geometry)
#
#         gdf_shapefile = gdf_shapefile.to_crs(crd_GHCN_geo.crs)
#         # Perform spatial join
#         # joined_data = gpd.sjoin(crd_GHCN_geo, gdf_shapefile, how='left', op='within')
#
#         # Create a key for the dictionary using the shapefile name (without the '.shp' extension)
#         shapefile_name = os.path.splitext(filename)[0]
#         # names.append(os.path.splitext(filename)[0])
#         merged_polygon = gdf_shapefile.unary_union
#
#         # names.append(os.path.splitext(filename)[0])
#         grouped_coordinates[shapefile_name] = []
#
#         # Loop through each coordinate in df_coordinates
#         for idx, coord in crd_GHCN_geo.iterrows():
#             # Check if the coordinate falls within any polygon of the shapefile
#             if coord['geometry'].within(merged_polygon):
#                 grouped_coordinates[shapefile_name].append(coord['Station_ID'])
#                 print(coord['Station_ID'])
#         k=1

output_file_path = '/data/kas7897/grouped_GHCN_crd.json'

# # Save the dictionary to a JSON file
# with open(output_file_path, 'w') as f:
#     json.dump(grouped_coordinates, f)

with open(output_file_path, 'r') as f:
    grouped_dict = json.load(f)

keys_to_remove = [key for key, value in grouped_dict.items() if isinstance(value, list) and len(value) == 0]

# Remove the keys with empty lists
for key in keys_to_remove:
    del grouped_dict[key]
stainds = list(grouped_dict.keys())

rmse_daymet = np.empty(len(grouped_dict))
acf_daymet = np.empty(len(grouped_dict))
rmse_daymet_3day = np.empty(len(grouped_dict))
hrmse_daymet = np.empty(len(grouped_dict))
lrmse_daymet = np.empty(len(grouped_dict))
bias_daymet = np.empty(len(grouped_dict))
abias_daymet = np.empty(len(grouped_dict))
rmse_wdaymet = np.empty(len(grouped_dict))
corr_daymet = np.empty(len(grouped_dict))
corr_daymet_3day = np.empty(len(grouped_dict))
corr_wdaymet = np.empty(len(grouped_dict))

rmse_maurer = np.empty(len(grouped_dict))
rmse_maurer_3day = np.empty(len(grouped_dict))
hrmse_maurer = np.empty(len(grouped_dict))
lrmse_maurer = np.empty(len(grouped_dict))
bias_maurer = np.empty(len(grouped_dict))
abias_maurer = np.empty(len(grouped_dict))
rmse_wmaurer = np.empty(len(grouped_dict))
corr_maurer = np.empty(len(grouped_dict))
corr_maurer_3day = np.empty(len(grouped_dict))
corr_wmaurer = np.empty(len(grouped_dict))
acf_maurer = np.empty(len(grouped_dict))

rmse_nldas = np.empty(len(grouped_dict))
acf_nldas = np.empty(len(grouped_dict))
rmse_nldas_3day = np.empty(len(grouped_dict))
hrmse_nldas = np.empty(len(grouped_dict))
lrmse_nldas = np.empty(len(grouped_dict))
bias_nldas = np.empty(len(grouped_dict))
abias_nldas = np.empty(len(grouped_dict))
rmse_wnldas = np.empty(len(grouped_dict))
corr_nldas = np.empty(len(grouped_dict))
corr_nldas_3day = np.empty(len(grouped_dict))
corr_wnldas = np.empty(len(grouped_dict))

rmse_multi = np.empty(len(grouped_dict))
acf_multi = np.empty(len(grouped_dict))
acf_GHCN = np.empty(len(grouped_dict))
rmse_multi_3day = np.empty(len(grouped_dict))
hrmse_multi = np.empty(len(grouped_dict))
lrmse_multi = np.empty(len(grouped_dict))
bias_multi = np.empty(len(grouped_dict))
abias_multi = np.empty(len(grouped_dict))
corr_multi = np.empty(len(grouped_dict))

corr_multi_3day = np.empty(len(grouped_dict))
bias_multi_3day = np.empty(len(grouped_dict))
abias_multi_3day = np.empty(len(grouped_dict))
bias_daymet_3day = np.empty(len(grouped_dict))
abias_daymet_3day = np.empty(len(grouped_dict))
bias_maurer_3day = np.empty(len(grouped_dict))
abias_maurer_3day = np.empty(len(grouped_dict))
bias_nldas_3day = np.empty(len(grouped_dict))
abias_nldas_3day = np.empty(len(grouped_dict))


mean_nldas_3day = np.empty(len(grouped_dict))
std_nldas_3day = np.empty(len(grouped_dict))
acf_nldas_3day = np.empty(len(grouped_dict))

mean_daymet_3day = np.empty(len(grouped_dict))
std_daymet_3day = np.empty(len(grouped_dict))
acf_daymet_3day = np.empty(len(grouped_dict))

mean_maurer_3day = np.empty(len(grouped_dict))
std_maurer_3day = np.empty(len(grouped_dict))
acf_maurer_3day = np.empty(len(grouped_dict))

mean_multi_3day = np.empty(len(grouped_dict))
std_multi_3day = np.empty(len(grouped_dict))
acf_multi_3day = np.empty(len(grouped_dict))

mean_ghcn_3day = np.empty(len(grouped_dict))
std_ghcn_3day = np.empty(len(grouped_dict))
acf_ghcn_3day = np.empty(len(grouped_dict))

mean_avg_3day = np.empty(len(grouped_dict))
std_avg_3day = np.empty(len(grouped_dict))
acf_avg = np.empty(len(grouped_dict))
bias_avg_3day = np.empty(len(grouped_dict))
abias_avg_3day = np.empty(len(grouped_dict))
corr_avg_3day = np.empty(len(grouped_dict))

rmse_nldas_maurer = np.empty(len(grouped_dict))
corr_nldas_maurer = np.empty(len(grouped_dict))

lossfactor=23
smoothfactor=0
# expno=11
wghts_sdate = '1980-10-01'


w0 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts1.csv',
    header=None)
wghts_days = len(w0)
date_range_wghts = pd.date_range(start=wghts_sdate, periods=wghts_days)
w0['dates'] = date_range_wghts
w0['Day'] = w0['dates'].dt.day
w0['Month'] = w0['dates'].dt.month
w0['Year'] = w0['dates'].dt.year
w0.drop(columns=['dates'], inplace=True)


w1 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts2.csv',
    header=None)

w1['dates'] = date_range_wghts
w1['Day'] = w1['dates'].dt.day
w1['Month'] = w1['dates'].dt.month
w1['Year'] = w1['dates'].dt.year
w1.drop(columns=['dates'], inplace=True)


w2 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts3.csv',
    header=None)
# w2 = pd.read_csv(
#     f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/nldas_extended|maurer_extendedwithloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts1.csv',
#     header=None)
w2['dates'] = date_range_wghts
w2['Day'] = w2['dates'].dt.day
w2['Month'] = w2['dates'].dt.month
w2['Year'] = w2['dates'].dt.year
w2.drop(columns=['dates'], inplace=True)



for i in range(len(grouped_dict)):
    st_ind = stainds[i]
    # print(i)
     #dealing with camels stations with no neighbours
    # camels_gage = str(crd_camels['gage'][st_ind])
    # if len(camels_gage)!=8:
    #     camels_gage = '0' + camels_gage
    camels_index = crd_camels[crd_camels['gage'] == int(st_ind)].index
    huc = crd_camels.loc[crd_camels['gage'] == int(st_ind), 'huc'].values[0]
    if len(str(huc)) == 1:
        huc = '0' + str(huc)

    w0_station = w0[[camels_index[0], 'Year', 'Month', 'Day']]
    w1_station = w1[[camels_index[0], 'Year', 'Month', 'Day']]
    w2_station = w2[[camels_index[0], 'Year', 'Month', 'Day']]

    prcp_daymet_dir = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/daymet/{huc}/{st_ind}_lump_cida_forcing_leap.txt"
    prcp_maurer_dir = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/maurer_extended/{huc}/{st_ind}_lump_maurer_forcing_leap.txt"
    prcp_nldas_dir = f"/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/nldas_extended/{huc}/{st_ind}_lump_nldas_forcing_leap.txt"
    prcp_daymet = pd.read_csv(prcp_daymet_dir, sep=r'\s+', header=None, skiprows=4)
    prcp_maurer = pd.read_csv(prcp_maurer_dir, sep=r'\s+', header=None, skiprows=4)
    prcp_nldas = pd.read_csv(prcp_nldas_dir, sep=r'\s+', header=None, skiprows=4)
    prcp_daymet.rename(columns={0:'Year', 1:'Month', 2:'Day', 5:'precip_daymet'}, inplace=True)
    prcp_maurer.rename(columns={0:'Year', 1:'Month', 2:'Day', 5:'precip_maurer'}, inplace=True)
    prcp_nldas.rename(columns={0:'Year', 1:'Month', 2:'Day', 5:'precip_nldas'}, inplace=True)
    prcp_daymet.drop(columns=[3, 4, 6, 7, 8, 9, 10], inplace=True)
    prcp_maurer.drop(columns=[3, 4, 6, 7, 8, 9, 10], inplace=True)
    prcp_nldas.drop(columns=[3, 4, 6, 7, 8, 9, 10], inplace=True)

    # prcp_daymet = prcp_daymet[np.where(date_range == np.datetime64('1981-01-01'))[0][0]: np.where(date_range == np.datetime64('2005-10-01'))[0][0]][5].reset_index(drop=True)
    # prcp_nldas = prcp_nldas[np.where(date_range == np.datetime64('1981-01-01'))[0][0]: np.where(date_range == np.datetime64('2005-10-01'))[0][0]][5].reset_index(drop=True)
    # prcp_maurer = prcp_maurer[np.where(date_range_maurer == np.datetime64('1981-01-01'))[0][0]: np.where(date_range_maurer == np.datetime64('2005-10-01'))[0][0]][5].reset_index(drop=True)
    prcp_daymet = pd.merge(prcp_daymet, w0_station, on=['Year', 'Month', 'Day'], how='inner')
    prcp_maurer = pd.merge(prcp_maurer, w1_station, on=['Year', 'Month', 'Day'], how='inner')
    prcp_nldas = pd.merge(prcp_nldas, w2_station, on=['Year', 'Month', 'Day'], how='inner')

    ghcn_stations = grouped_dict[st_ind]
    for j in ghcn_stations:
        st_data = pd.read_csv(f"/data/kas7897/GHCN_Data/Station_Data/{j}.csv")
        prcp_daymet = pd.merge(prcp_daymet, st_data, on=['Year', 'Month', 'Day'], how='left')
        prcp_maurer = pd.merge(prcp_maurer, st_data, on=['Year', 'Month', 'Day'], how='left')
        prcp_nldas = pd.merge(prcp_nldas, st_data, on=['Year', 'Month', 'Day'], how='left')
        prcp_daymet.drop(columns=['TMin', 'TMax'], inplace=True)
        prcp_maurer.drop(columns=['TMin', 'TMax'], inplace=True)
        prcp_nldas.drop(columns=['TMin', 'TMax'], inplace=True)
        prcp_daymet.rename(columns={'Precip':j}, inplace=True)
        prcp_maurer.rename(columns={'Precip':j}, inplace=True)
        prcp_nldas.rename(columns={'Precip':j}, inplace=True)


        # prcp_daymet = prcp_daymet[prcp_daymet[j] != -9999]
        # prcp_maurer = prcp_maurer[prcp_maurer[j] != -9999]
        # prcp_nldas = prcp_nldas[prcp_nldas[j] != -9999]
        prcp_daymet.replace(-9999, np.nan, inplace=True)
        prcp_maurer.replace(-9999, np.nan, inplace=True)
        prcp_nldas.replace(-9999, np.nan, inplace=True)

    if (prcp_daymet[j] == -9999).any():
        print("There are -9999 values in the column.")

    prcp_daymet['avg_prcp'] = prcp_daymet.iloc[:, -len(ghcn_stations):].mean(axis=1, skipna=True)
    prcp_maurer['avg_prcp'] = prcp_maurer.iloc[:, -len(ghcn_stations):].mean(axis=1,skipna=True)
    prcp_nldas['avg_prcp'] = prcp_nldas.iloc[:, -len(ghcn_stations):].mean(axis=1,skipna=True)

    prcp_daymet['wght_prcp'] = prcp_daymet['precip_daymet']*prcp_daymet[camels_index[0]]
    prcp_maurer['wght_prcp'] = prcp_maurer['precip_maurer']*prcp_maurer[camels_index[0]]
    prcp_nldas['wght_prcp'] = prcp_nldas['precip_nldas']*prcp_nldas[camels_index[0]]

    prcp_avg = (prcp_daymet['precip_daymet'] + prcp_maurer['precip_maurer'] + prcp_nldas['precip_nldas'])/3
    multi_prcp = prcp_daymet['wght_prcp'] + prcp_maurer['wght_prcp'] + prcp_nldas['wght_prcp']
    # nldas_maurer_prcp = prcp_maurer['wght_prcp'] + prcp_nldas['wght_prcp']

    true_prcp_3day = prcp_daymet['avg_prcp'].rolling(3).mean()
    true_prcp_3day.dropna(inplace=True)
    multi_prcp_3day = multi_prcp.rolling(3).mean()
    multi_prcp_3day = multi_prcp_3day[true_prcp_3day.index]
    prcp_daymet_3day = prcp_daymet['precip_daymet'].rolling(3).mean()
    prcp_daymet_3day = prcp_daymet_3day[true_prcp_3day.index]
    # prcp_daymet_3day.dropna(inplace=True)
    prcp_maurer_3day = prcp_maurer['precip_maurer'].rolling(3).mean()
    prcp_maurer_3day = prcp_maurer_3day[true_prcp_3day.index]
    # prcp_maurer_3day.dropna(inplace=True)
    prcp_nldas_3day = prcp_nldas['precip_nldas'].rolling(3).mean()
    prcp_nldas_3day = prcp_nldas_3day[true_prcp_3day.index]
    prcp_avg_3day = prcp_avg.rolling(3).mean()
    prcp_avg_3day = prcp_avg_3day[true_prcp_3day.index]


    rmse_multi_3day[i] = rmse(multi_prcp_3day, true_prcp_3day)
    corr_multi_3day[i] = np.corrcoef(multi_prcp_3day, true_prcp_3day)[0,1]
    rmse_daymet_3day[i] = rmse(prcp_daymet_3day, true_prcp_3day)
    corr_daymet_3day[i] = np.corrcoef(prcp_daymet_3day, true_prcp_3day)[0,1]
    rmse_maurer_3day[i] = rmse(prcp_maurer_3day, true_prcp_3day)
    corr_maurer_3day[i] = np.corrcoef(prcp_maurer_3day, true_prcp_3day)[0,1]
    rmse_nldas_3day[i] = rmse(prcp_nldas_3day, true_prcp_3day)
    corr_nldas_3day[i] = np.corrcoef(prcp_nldas_3day, true_prcp_3day)[0,1]

    corr_avg_3day[i] = np.corrcoef(prcp_avg_3day, true_prcp_3day)[0,1]

    bias_multi_3day[i] = np.nanmean(multi_prcp_3day - true_prcp_3day)
    abias_multi_3day[i] = np.nanmean(np.abs(multi_prcp_3day - true_prcp_3day))
    bias_daymet_3day[i] = np.nanmean(prcp_daymet_3day - true_prcp_3day)
    abias_daymet_3day[i] = np.nanmean(np.abs(prcp_daymet_3day - true_prcp_3day))
    bias_maurer_3day[i] = np.nanmean(prcp_maurer_3day - true_prcp_3day)
    abias_maurer_3day[i] = np.nanmean(np.abs(prcp_maurer_3day - true_prcp_3day))
    bias_nldas_3day[i] = np.nanmean(prcp_nldas_3day - true_prcp_3day)
    abias_nldas_3day[i] = np.nanmean(np.abs(prcp_nldas_3day - true_prcp_3day))

    abias_avg_3day[i] = np.nanmean(np.abs(prcp_avg_3day - true_prcp_3day))
    bias_avg_3day[i] = np.nanmean(prcp_avg_3day - true_prcp_3day)

    mean_ghcn_3day[i] = np.nanmean(true_prcp_3day)
    std_ghcn_3day[i] = np.nanstd(true_prcp_3day)
    # acf_ghcn_3day[i] = acf(true_prcp_3day, nlags=1, missing='conservative')[1]

    std_avg_3day[i] = np.nanstd(std_avg_3day)

    mean_daymet_3day[i] = np.nanmean(prcp_daymet_3day)
    std_daymet_3day[i] = np.nanstd(prcp_daymet_3day)
    # acf_daymet_3day[i] = acf(prcp_daymet_3day, nlags=1)[1]

    mean_maurer_3day[i] = np.nanmean(prcp_maurer_3day)
    std_maurer_3day[i] = np.nanstd(prcp_maurer_3day)
    # acf_maurer_3day[i] = acf(prcp_maurer_3day, nlags=1)[1]

    mean_nldas_3day[i] = np.nanmean(prcp_nldas_3day)
    std_nldas_3day[i] = np.nanstd(prcp_nldas_3day)
    # acf_nldas_3day[i] = acf(prcp_nldas_3day, nlags=1)[1]

    mean_multi_3day[i] = np.nanmean(multi_prcp_3day)
    std_multi_3day[i] = np.nanstd(multi_prcp_3day)
    # acf_multi_3day[i] = acf(multi_prcp_3day, nlags=1)[1]

    acf_multi[i] = acf(multi_prcp, nlags=1)[1]
    acf_avg[i] = acf(prcp_avg, nlags=1)[1]
    acf_daymet[i] = acf(prcp_daymet['precip_daymet'], nlags=1)[1]
    acf_maurer[i] = acf(prcp_maurer['precip_maurer'], nlags=1)[1]
    acf_nldas[i] = acf(prcp_nldas['precip_nldas'], nlags=1)[1]
    acf_GHCN[i] = acf(prcp_daymet['avg_prcp'], nlags=1, missing = 'conservative')[1]


    prcp_daymet.dropna(inplace=True)
    prcp_maurer.dropna(inplace=True)
    prcp_nldas.dropna(inplace=True)
    multi_prcp = multi_prcp[prcp_daymet.index]




    rmse_multi[i] = rmse(multi_prcp, prcp_daymet['avg_prcp'])
    # rmse_nldas_maurer[i] = rmse(nldas_maurer_prcp, prcp_daymet['avg_prcp'])
    corr_multi[i] = np.corrcoef(multi_prcp, prcp_maurer['avg_prcp'])[0,1]
    # corr_nldas_maurer[i] = np.corrcoef(nldas_maurer_prcp, prcp_maurer['avg_prcp'])[0,1]
    bias_multi[i] = np.nanmean(multi_prcp - prcp_daymet['avg_prcp'])
    abias_multi[i] = np.nanmean(np.abs(multi_prcp - prcp_daymet['avg_prcp']))

    rmse_daymet[i] = rmse(prcp_daymet['precip_daymet'], prcp_daymet['avg_prcp'])
    bias_daymet[i] = np.nanmean(prcp_daymet['precip_daymet']-prcp_daymet['avg_prcp'])
    abias_daymet[i] = np.nanmean(np.abs(prcp_daymet['precip_daymet']-prcp_daymet['avg_prcp']))
    rmse_wdaymet[i] = rmse(prcp_daymet['wght_prcp'], prcp_daymet['avg_prcp'])
    corr_daymet[i] = np.corrcoef(prcp_daymet['precip_daymet'], prcp_daymet['avg_prcp'])[0,1]
    corr_wdaymet[i] = np.corrcoef(prcp_daymet['wght_prcp'], prcp_daymet['avg_prcp'])[0,1]

    rmse_maurer[i] = rmse(prcp_maurer['precip_maurer'], prcp_maurer['avg_prcp'])
    bias_maurer[i] = np.nanmean(prcp_maurer['precip_maurer'] - prcp_maurer['avg_prcp'])
    abias_maurer[i] = np.nanmean(np.abs(prcp_maurer['precip_maurer'] - prcp_maurer['avg_prcp']))
    rmse_wmaurer[i] = rmse(prcp_maurer['wght_prcp'], prcp_maurer['avg_prcp'])
    corr_maurer[i] = np.corrcoef(prcp_maurer['precip_maurer'], prcp_maurer['avg_prcp'])[0,1]
    corr_wmaurer[i] = np.corrcoef(prcp_maurer['wght_prcp'], prcp_maurer['avg_prcp'])[0,1]

    rmse_nldas[i] = rmse(prcp_nldas['precip_nldas'], prcp_nldas['avg_prcp'])
    bias_nldas[i] = np.nanmean(prcp_nldas['precip_nldas'] - prcp_nldas['avg_prcp'])
    abias_nldas[i] = np.nanmean(np.abs(prcp_nldas['precip_nldas'] - prcp_nldas['avg_prcp']))
    rmse_wnldas[i] = rmse(prcp_nldas['wght_prcp'], prcp_nldas['avg_prcp'])
    corr_nldas[i] = np.corrcoef(prcp_nldas['precip_nldas'], prcp_nldas['avg_prcp'])[0,1]
    corr_wnldas[i] = np.corrcoef(prcp_nldas['wght_prcp'], prcp_nldas['avg_prcp'])[0,1]



    daymet_sort = np.sort(prcp_daymet['precip_daymet'])
    maurer_sort = np.sort(prcp_maurer['precip_maurer'])
    nldas_sort = np.sort(prcp_nldas['precip_nldas'])
    multi_sort = np.sort(multi_prcp)
    true_sort = np.sort(prcp_daymet['avg_prcp'])

    lowindex = round(0.3 * len(daymet_sort))
    highindex = round(0.98 * len(daymet_sort))

    low_daymet = daymet_sort[:lowindex]
    low_maurer = maurer_sort[:lowindex]
    low_nldas = nldas_sort[:lowindex]
    low_multi = multi_sort[:lowindex]
    low_true = true_sort[:lowindex]

    high_daymet = daymet_sort[highindex:]
    high_maurer = maurer_sort[highindex:]
    high_nldas = nldas_sort[highindex:]
    high_multi = multi_sort[highindex:]
    high_true = true_sort[highindex:]

    lrmse_multi[i] = rmse(low_multi, low_true)
    hrmse_multi[i] = rmse(high_multi, high_true)
    lrmse_daymet[i] = rmse(low_daymet, low_true)
    hrmse_daymet[i] = rmse(high_daymet, high_true)
    lrmse_maurer[i] = rmse(low_maurer, low_true)
    hrmse_maurer[i] = rmse(high_maurer, high_true)
    lrmse_nldas[i] = rmse(low_nldas, low_true)
    hrmse_nldas[i] = rmse(high_nldas, high_true)

k=1
crd_topo = pd.read_csv("/scratch/Camels/camels_attributes_v2.0/camels_attributes_v2.0/camels_topo.txt", sep=';')
stainds_int = [int(item) for item in stainds]
stainds_int_df = pd.DataFrame({'gauge_id': stainds_int})
# stainds_int_df1 = pd.DataFrame({'gage': stainds_int})
# stainds_int_df1 = stainds_int_df1.merge(crd_camels, on='gage', how='left')
stainds_int_df = stainds_int_df.merge(crd_topo, on='gauge_id', how='left')
stainds_int_df_filter = stainds_int_df[stainds_int_df['elev_mean']>1000]
rmse_better_Nldasdaymet = [stainds_int for stainds_int, flag in zip(stainds_int, rmse_daymet_3day<rmse_nldas_3day) if flag]
rmse_better_fuseDdaymet = [stainds_int for stainds_int, flag in zip(stainds_int, rmse_daymet_3day<rmse_multi_3day) if flag]
corr_better_Nldasdaymet = [stainds_int for stainds_int, flag in zip(stainds_int, corr_daymet_3day<corr_nldas_3day) if flag]
corr_better_fuseDdaymet = [stainds_int for stainds_int, flag in zip(stainds_int, corr_daymet_3day<corr_multi_3day) if flag]
# filtered_crd = crd_camels[crd_camels['gage'].isin(corr_better_fuseDdaymet)].reset_index(drop=True)
filtered_crd = crd_camels[crd_camels['gage'].isin(stainds_int)].reset_index(drop=True)
# filtered_crd = filteredcrd.loc[filtered_crd['gage'].isin(stainds_int)]
# rmse_better_Nldasdaymet = stainds_int[rmse_daymet_3day<rmse_nldas_3day]
# rmse_better_Nldasdaymet = filtered_crd[:][rmse_daymet_3day<rmse_nldas_3day]
# corr_better_Nldasdaymet = filtered_crd[:][corr_daymet_3day<corr_nldas_3day]
# corr_better_Nldasdaymet = stainds_int[corr_daymet_3day<corr_nldas_3day]
print(stainds_int)
print(filtered_crd)
# m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
#             projection='lcc', lat_1=33, lat_2=45, lon_0=-95,
#             resolution='c', area_thresh=10000)
# m.etopo(scale=0.5, alpha=0.5)
# # m.drawmapboundary(fill_color='#46bcec')
# # m.fillcontinents(color='#524c4c', lake_color='#46bcec')
# # m.fillcontinents(color='0.1')
# m.drawcoastlines()
#
# # x, y = m(filtered_crd['LONG'].values, filtered_crd['LAT'].values)
# # x, y = m(stainds_int_df['LONG'].values, stainds_int_df['LAT'].values)
# x, y = m(stainds_int_df_filter['gauge_lon'].values, stainds_int_df_filter['gauge_lat'].values)
#
# m.scatter(x, y, c = (rmse_multi_3day[stainds_int_df_filter.index]-rmse_daymet_3day[stainds_int_df_filter.index]), cmap='seismic', alpha=0.9, zorder=2)
# # cbar = plt.colorbar(label='wsum (sum of weights)', shrink=0.92, orientation='horizontal', pad=0.05,  fontsize=16)
# cbar = plt.colorbar(shrink=0.92, orientation='horizontal', pad=0.05)
# cbar.set_label(f'RMSE Difference for basins at elevation >1000m (Fused minus Daymet)', size=7)
# cbar.ax.tick_params(labelsize=7)
# plt.show()

print('3-day Corr Multi:', np.median(corr_multi_3day), '3-day RMSE Multi:', np.median(rmse_multi_3day))
print('3-day Corr Daymet:', np.median(corr_daymet_3day), '3-day RMSE Daymet:', np.median(rmse_daymet_3day))
print('3-day Corr Maurer:', np.median(corr_maurer_3day), '3-day RMSE Maurer:', np.median(rmse_maurer_3day))
print('3-day Corr Nldas:', np.median(corr_nldas_3day), '3-day RMSE Nldas:', np.median(rmse_nldas_3day))


print('Low_RMSE_multi', np.median(lrmse_multi))
print('Low_RMSE_daymet', np.median(lrmse_daymet))
print('Low_RMSE_maurer', np.median(lrmse_maurer))
print('Low_RMSE_nldas', np.median(lrmse_nldas))

print('High_RMSE_multi', np.median(hrmse_multi))
print('High_RMSE_daymet', np.median(hrmse_daymet))
print('High_RMSE_maurer', np.median(hrmse_maurer))
print('High_RMSE_nldas', np.median(hrmse_nldas))

print('RMSE_multi:', np.median(rmse_multi))

print('Bias_multi:', np.median(bias_multi))
print('ABias_multi:', np.median(abias_multi))
print('Bias_daymet:', np.median(bias_daymet))
print('ABias_daymet:', np.median(abias_daymet))
print('Bias_maurer:', np.median(bias_maurer))
print('ABias_maurer:', np.median(abias_maurer))
print('Bias_nldas:', np.median(bias_nldas))
print('ABias_nldas:', np.median(abias_nldas))
# print('RMSE_multi:', np.median(rmse_nldas_maurer))
print('Corr_multi:', np.median(corr_multi))
# print('Corr_multi:', np.median(corr_nldas_maurer))

print('RMSE_Daymet:', np.median(rmse_daymet), 'RMSE_Weighted_Daymet:', np.median(rmse_wdaymet))
print('CORR_Daymet:', np.median(corr_daymet), 'CORR_Weighted_Daymet:', np.median(corr_wdaymet))
print('RMSE_Maurer:', np.median(rmse_maurer), 'RMSE_Weighted_Maurer:', np.median(rmse_wmaurer))
print('CORR_Maurer:', np.median(corr_maurer), 'CORR_Weighted_Maurer:', np.median(corr_wmaurer))
print('RMSE_NLDAS:', np.median(rmse_nldas), 'RMSE_Weighted_NLDAS:', np.median(rmse_wnldas))
print('CORR_NLDAS:', np.median(corr_nldas), 'CORR_Weighted_NLDAS:', np.median(corr_wnldas))
#
# plt.scatter(rmse_multi_3day, rmse_daymet_3day, label=f'Daymet (median={np.median(rmse_daymet_3day):.3f})', marker='o')
# plt.scatter(rmse_multi_3day, rmse_maurer_3day, label=f'Maurer (median={np.median(rmse_maurer_3day):.3f})', marker='s')
# plt.scatter(rmse_multi_3day, rmse_nldas_3day, label=f'NLDAS (median={np.median(rmse_nldas_3day):.3f})', marker='^')
# plt.plot(rmse_multi_3day, rmse_multi_3day, color='gray', linestyle='--', label=f'Fused Precipitation (median={np.median(rmse_multi_3day):.3f})')
# plt.legend(loc = 'lower right',  prop={'size': 8})
#
# plt.xlabel('GHCN RMSE with Fused Data (3-Day Moving Average)')
# plt.ylabel('GHCN RMSE with Forcing Dataset (3-Day Moving Average)')
# plt.title('GHCN RMSE Scatter Plot ')
#
# plt.scatter(corr_multi_3day, corr_daymet_3day, label=f'Daymet (median={np.median(corr_daymet_3day):.3f})', marker='o')
# plt.scatter(corr_multi_3day, corr_maurer_3day, label=f'Maurer (median={np.median(corr_maurer_3day):.3f})', marker='s')
# plt.scatter(corr_multi_3day, corr_nldas_3day, label=f'NLDAS (median={np.median(corr_nldas_3day):.3f})', marker='^')
# plt.plot(corr_multi_3day, corr_multi_3day, color='gray', linestyle='--', label=f'Fused Precipitation (median={np.median(corr_multi_3day):.3f})')
# plt.legend(loc = 'lower right',  prop={'size': 8})
#
# plt.xlabel('GHCN Correlation with Fused Data (3-Day Moving Average)')
# plt.ylabel('GHCN Correlation with Forcing Dataset (3-Day Moving Average)')
# plt.title('GHCN Correlation Scatter Plot ')
# plt.show()
# fig, axs = plt.subplots(1, 3, figsize=(12, 6))
fig, axs = plt.subplots(1, 3, figsize=(15,5))

# Plot the first scatter plot in the first subplot
axs[0].scatter(bias_avg_3day, bias_daymet_3day, marker='o', color='red',
               label=f'Daymet (mean Bias:{np.mean(abias_daymet_3day):.3f}, mean Correlation:{np.mean(corr_daymet_3day):.3f})')
axs[0].scatter(bias_avg_3day, bias_maurer_3day, marker='s', color='green',
               label=f'Maurer (mean Bias:{np.mean(abias_maurer_3day):.3f}, mean Correlation:{np.mean(corr_maurer_3day):.3f})')
axs[0].scatter(bias_avg_3day, bias_nldas_3day, marker='^', color='blue',
               label=f'NLDAS (mean Bias:{np.mean(abias_nldas_3day):.3f}, mean Correlation:{np.mean(corr_nldas_3day):.3f})')
axs[0].scatter(bias_avg_3day, bias_multi_3day, marker='*', color='black',
               label=f'Fused P (mean Bias:{np.mean(abias_multi_3day):.3f}, mean Correlation:{np.mean(corr_multi_3day):.3f})')
axs[0].plot(bias_avg_3day, bias_avg_3day, color='gray', linestyle='--',
            label=f'Avg P (mean Bias:{np.mean(abias_avg_3day):.3f}, mean Correlation:{np.mean(corr_avg_3day):.3f})')
# axs[0].legend(loc='lower right', prop={'size': 10})
axs[0].set_xlabel('GHCN Bias with Avg P', size = 14)
axs[0].set_ylabel('GHCN Bias with P datasets', size = 14)
axs[0].set_title('(a) GHCN Bias  Scatter Plot', size = 16)
axs[0].tick_params(labelsize=12)


# Plot the second scatter plot in the second subplot
axs[1].scatter(corr_avg_3day, corr_daymet_3day, marker='o', color='red',
               label=f'Daymet (mean Bias:{np.mean(abias_daymet_3day):.3f}, mean Correlation:{np.mean(corr_daymet_3day):.3f})')
axs[1].scatter(corr_avg_3day, corr_maurer_3day, marker='s', color='green',
               label=f'Maurer (mean Bias:{np.mean(abias_maurer_3day):.3f}, mean Correlation:{np.mean(corr_maurer_3day):.3f})')
axs[1].scatter(corr_avg_3day, corr_nldas_3day, marker='^', color='blue',
               label=f'NLDAS (mean Bias:{np.mean(abias_nldas_3day):.3f}, mean Correlation:{np.mean(corr_nldas_3day):.3f})')
axs[1].scatter(corr_avg_3day, corr_multi_3day, color='black', marker='*',
               label=f'Fused P (mean Bias:{np.mean(abias_multi_3day):.3f}, mean Correlation:{np.mean(corr_multi_3day):.3f})')
axs[1].plot(corr_avg_3day, corr_avg_3day, color='gray', linestyle='--',
            label=f'Avg P (mean Bias:{np.mean(abias_avg_3day):.3f}, mean Correlation:{np.mean(corr_avg_3day):.3f})')

# axs[1].legend(loc='lower right', prop={'size': 10})
axs[1].set_xlabel('GHCN Correlation with Avg P', size = 14)
axs[1].set_ylabel('GHCN Correlation with P datasets', size = 14)
axs[1].set_title('(b) GHCN Correlation Scatter Plot', size = 16)
axs[1].tick_params(labelsize=12)


axs[2].scatter(corr_avg_3day, corr_daymet_3day, marker='o', color='red',
               label=f'Daymet (mean Bias:{np.mean(abias_daymet_3day):.3f}, mean Correlation:{np.mean(corr_daymet_3day):.3f})')
axs[2].scatter(corr_avg_3day, corr_maurer_3day, marker='s', color='green',
               label=f'Maurer (mean Bias:{np.mean(abias_maurer_3day):.3f}, mean Correlation:{np.mean(corr_maurer_3day):.3f})')
axs[2].scatter(corr_avg_3day, corr_nldas_3day, marker='^', color='blue',
               label=f'NLDAS (mean Bias:{np.mean(abias_nldas_3day):.3f}, mean Correlation:{np.mean(corr_nldas_3day):.3f})')
axs[2].scatter(corr_avg_3day, corr_multi_3day, marker='*', color='black',
               label=f'Fused P (mean Bias:{np.mean(abias_multi_3day):.3f}, mean Correlation:{np.mean(corr_multi_3day):.3f})')
axs[2].plot(corr_avg_3day, corr_avg_3day, color='gray', linestyle='--',
            label=f'Avg P (mean Bias:{np.mean(abias_avg_3day):.3f}, mean Correlation:{np.mean(corr_avg_3day):.3f})')
# axs[2].legend(loc='upper right', prop={'size': 15})
axs[2].set_xlabel('GHCN Correlation with Avg P', size = 14)
# axs[2].set_ylabel('GHCN Correlation with different P datasets(3-Day Moving Average)', size = 12)
axs[2].set_title('(c) GHCN Correlation Scatter Plot (Magnified)', size = 16)
axs[2].set_ylim(0.72, 1.0)
axs[2].set_xlim(0.72, 1.0)
axs[2].tick_params(labelsize=12)


# fig.legend(handles=[line, circle, square, triangle, star], loc='lower center', ncol=5, prop={'size': 10}, bbox_to_anchor=(0.5, -0.15))
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', prop={'size': 15})
fig.tight_layout()
plt.subplots_adjust(bottom=0.45)
plt.savefig('/home/kas7897/final_plots_fusion_paper/bias_corr_avg_GHCN.png',  dpi=300, bbox_inches='tight')

# Show the plots
plt.show()

# fig, ax = plt.subplots()

# data = [group]
# offsets = [1]
#
# for i, data_group in enumerate(data):
#     # The position for each box within the group
#     positions = [j + offsets[i] for j in range(len(data_group))]
#     ax.boxplot(data_group, positions=positions, widths=0.6)
#
# # Custom x-axis labels
# tick_positions = [j + offset for offset in offsets for j in range(4)]
# tick_labels = [
#     "ACF Fusion 2.0 (Daymet)", "ACF GHCN", "ACF Daymet", "ACF NLDAS"
# ]
#
# ax.set_xticks(tick_positions)
# ax.set_xticklabels(tick_labels, rotation=45, ha='right')  # rotated for better readability
#
# ax.set_title("Precipitation Statistics")
# plt.tight_layout()
# plt.show()


group1 = [mean_avg_3day, mean_multi_3day, mean_daymet_3day, mean_maurer_3day, mean_nldas_3day]
group2 = [std_avg_3day, std_multi_3day, std_daymet_3day, std_maurer_3day, std_nldas_3day]
# group3 = [acf_ghcn_3day, acf_multi_3day, acf_daymet_3day, acf_maurer_3day, acf_nldas_3day]
group3 = [acf_avg, acf_multi, acf_daymet, acf_maurer, acf_nldas]

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
    "Avg P", "Fused P", "Daymet", "Maurer", "NLDAS"
]

# ax[0].set_xticks(tick_positions)
ax[0].set_xticklabels(tick_labels)  # rotated for better readability

ax[0].set_title("(a)Mean (3-day moving average)", size=12)
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
    "Avg P", "Fused P", "Daymet", "Maurer", "NLDAS"
]

# ax[1].set_xticks(tick_positions)
ax[1].set_xticklabels(tick_labels)  # rotated for better readability
ax[1].set_title("(b)Standard Deviation (3-day moving average)", size=12)

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
    "Avg P", "Fused P", "Daymet", "Maurer", "NLDAS"
]

# ax[2].set_xticks(tick_positions)
ax[2].set_xticklabels(tick_labels)  # rotated for better readability
ax[2].set_title("(c)Autocorrelation (lag=1)", size=12)

# ax.set_title("Precipitation Statistics")
# plt.tight_layout()

# plt.savefig('/home/kas7897/final_plots_fusion_paper/prcp_withGHCN1_box_plots.png',  dpi=300, bbox_inches='tight')
plt.show()
