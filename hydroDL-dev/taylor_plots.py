from skill_metrics import taylor_diagram
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import skill_metrics as sm
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


from taylorDiagram import TaylorDiagram

from mpl_toolkits.axisartist import Subplot



lossfactor = 0
smoothfactor = 0
expno = 2

crd = pd.read_csv("/data/kas7897/dPLHBVrelease/hydroDL-dev/gages_list.csv")

w0 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/daymetwithloss{lossfactor}smooth{smoothfactor}_{4}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts1.csv',
    header=None)
w1 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/maurer_extendedwithloss{lossfactor}smooth{smoothfactor}_{expno}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts1.csv',
    header=None)
w2 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/nldas_extendedwithloss{lossfactor}smooth{smoothfactor}_{expno}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts1.csv',
    header=None)

corr1 = np.zeros(len(crd))
corr2 = np.zeros(len(crd))
rmse1 = np.zeros(len(crd))
rmse2 = np.zeros(len(crd))
mae1 = np.zeros(len(crd))
mae2 = np.zeros(len(crd))

prcp_gagewise_daymet = np.zeros(len(crd))
prcp_gagewise_maurer = np.zeros(len(crd))
prcp_gagewise_nldas = np.zeros(len(crd))

eff_wght_daymet = np.zeros(len(crd))
eff_wght_maurer = np.zeros(len(crd))
eff_wght_nldas = np.zeros(len(crd))

eff_prcp_daymet = np.zeros(len(crd))
eff_prcp_maurer = np.zeros(len(crd))
eff_prcp_nldas = np.zeros(len(crd))

std_daymet = np.zeros(len(crd))
std_nldas = np.zeros(len(crd))
std_eff_daymet = np.zeros(len(crd))
std_eff_nldas = np.zeros(len(crd))


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

    std_daymet[i] = np.std(prcp_daymet)
    std_nldas[i] = np.std(prcp_nldas)


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
    std_eff_daymet[i] = np.std(eff_daymet)
    std_eff_nldas[i] = np.std(eff_nldas)

    corr1[i] = pearsonr(prcp_daymet, prcp_nldas)[0]
    corr2[i] = pearsonr(prcp_daymet, eff_nldas)[0]

    rmse1[i] = np.sqrt(mean_squared_error(prcp_daymet, prcp_nldas))
    rmse2[i] = np.sqrt(mean_squared_error(eff_daymet, eff_nldas))

    mae1[i] = mean_absolute_error(prcp_daymet, prcp_nldas)
    mae2[i] = mean_absolute_error(eff_daymet, eff_nldas)


#reference
squared_std = np.square(std_daymet)
average_squared = np.mean(squared_std)
std_ref = np.sqrt(average_squared)

samples = np.column_stack((std_eff_nldas, corr2)).tolist()

# stdrefs = dict(winter=48.491,
#                spring=44.927,
#                summer=37.664,
#                autumn=41.589)

# Sample std,rho: Be sure to check order and that correct numbers are placed!
# samples = dict(winter=[[17.831, 0.360, "CCSM CRCM"],
#                        [27.062, 0.360, "CCSM MM5"],
#                        [33.125, 0.585, "CCSM WRFG"],
#                        [25.939, 0.385, "CGCM3 CRCM"],
#                        [29.593, 0.509, "CGCM3 RCM3"],
#                        [35.807, 0.609, "CGCM3 WRFG"],
#                        [38.449, 0.342, "GFDL ECP2"],
#                        [29.593, 0.509, "GFDL RCM3"],
#                        [71.215, 0.473, "HADCM3 HRM3"]],
#                spring=[[32.174, -0.262, "CCSM CRCM"],
#                        [24.042, -0.055, "CCSM MM5"],
#                        [29.647, -0.040, "CCSM WRFG"],
#                        [22.820, 0.222, "CGCM3 CRCM"],
#                        [20.505, 0.445, "CGCM3 RCM3"],
#                        [26.917, 0.332, "CGCM3 WRFG"],
#                        [25.776, 0.366, "GFDL ECP2"],
#                        [18.018, 0.452, "GFDL RCM3"],
#                        [79.875, 0.447, "HADCM3 HRM3"]],
#                summer=[[35.863, 0.096, "CCSM CRCM"],
#                        [43.771, 0.367, "CCSM MM5"],
#                        [35.890, 0.267, "CCSM WRFG"],
#                        [49.658, 0.134, "CGCM3 CRCM"],
#                        [28.972, 0.027, "CGCM3 RCM3"],
#                        [60.396, 0.191, "CGCM3 WRFG"],
#                        [46.529, 0.258, "GFDL ECP2"],
#                        [35.230, -0.014, "GFDL RCM3"],
#                        [87.562, 0.503, "HADCM3 HRM3"]],
#                autumn=[[27.374, 0.150, "CCSM CRCM"],
#                        [20.270, 0.451, "CCSM MM5"],
#                        [21.070, 0.505, "CCSM WRFG"],
#                        [25.666, 0.517, "CGCM3 CRCM"],
#                        [35.073, 0.205, "CGCM3 RCM3"],
#                        [25.666, 0.517, "CGCM3 WRFG"],
#                        [23.409, 0.353, "GFDL ECP2"],
#                        [29.367, 0.235, "GFDL RCM3"],
#                        [70.065, 0.444, "HADCM3 HRM3"]])

# Colormap (see http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps)
# colors = plt.matplotlib.cm.Set1(np.linspace(0,1,len(samples['winter'])))
colors = plt.matplotlib.cm.Set1(np.linspace(0,1,len(samples)))

# Here set placement of the points marking 95th and 99th significance
# levels. For more than 102 samples (degrees freedom > 100), critical
# correlation levels are 0.195 and 0.254 for 95th and 99th
# significance levels respectively. Set these by eyeball using the
# standard deviation x and y axis.

#x95 = [0.01, 0.68] # For Tair, this is for 95th level (r = 0.195)
#y95 = [0.0, 3.45]
#x99 = [0.01, 0.95] # For Tair, this is for 99th level (r = 0.254)
#y99 = [0.0, 3.45]

x95 = [0.05, 13.9] # For Prcp, this is for 95th level (r = 0.195)
y95 = [0.0, 71.0]
x99 = [0.05, 19.0] # For Prcp, this is for 99th level (r = 0.254)
y99 = [0.0, 70.0]

rects = dict(winter=221,
             spring=222,
             summer=223,
             autumn=224)

fig = plt.figure(figsize=(11,18))
fig.suptitle("Evaluating weighted-NLDAS2 against Observed-Daymet", size='x-large')

# for season in ['winter','spring','summer','autumn']:

# dia = TaylorDiagram(stdrefs[season], fig=fig, rect=rects[season],
#                     label='Reference')
dia = TaylorDiagram(std_ref, fig=fig)

# dia.ax.plot(x95,y95,color='k')
# dia.ax.plot(x99,y99,color='k')

# Add samples to Taylor diagram
# for i,(stddev,corrcoef,name) in enumerate(samples[season]):
#     dia.add_sample(stddev, corrcoef,
#                    marker='$%d$' % (i+1), ms=10, ls='',
#                    #mfc='k', mec='k', # B&W
#                    mfc=colors[i], mec=colors[i], # Colors
#                    label=name)

for i, (stddev, corrcoef) in enumerate(samples):
    dia.add_sample(stddev, corrcoef,
                   marker='o', ms=10, ls='',
                   # mfc='k', mec='k', # B&W
                   mfc=colors[i], mec=colors[i])

# Add RMS contours, and label them
contours = dia.add_contours(levels=5, colors='0.5') # 5 levels
dia.ax.clabel(contours, inline=1, fontsize=10, fmt='%.1f')
# Tricky: ax is the polar ax (used for plots), _ax is the
# container (used for layout)
# dia._ax.set_title(season.capitalize())

# Add a figure legend and title. For loc option, place x,y tuple inside [ ].
# Can also use special options here:
# http://matplotlib.sourceforge.net/users/legend_guide.html

fig.legend(dia.samplePoints,
           [ p.get_label() for p in dia.samplePoints ],
           numpoints=1, prop=dict(size='small'), loc='center')

# fig.tight_layout()

# PLT.savefig('test_taylor_4panel.png')
plt.show()
