import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import random
import json
import datetime as dt
from datetime import datetime
from hydroDL.post import plot, stat

path= []
path.append('/data/kas7897/dPLHBVrelease/output/ensemble1/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss60/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/pred50.npy')
path.append('/data/kas7897/dPLHBVrelease/output/ensemble2/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss60/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/pred50.npy')
path.append('/data/kas7897/dPLHBVrelease/output/ensemble3/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss60/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/pred50.npy')
path.append('/data/kas7897/dPLHBVrelease/output/ensemble4/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss60/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/pred50.npy')
path.append('/data/kas7897/dPLHBVrelease/output/ensemble5/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss60/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/pred50.npy')
path.append('/data/kas7897/dPLHBVrelease/output/ensemble6/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss60/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/pred50.npy')
path.append('/data/kas7897/dPLHBVrelease/output/ensemble7/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss60/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/pred50.npy')
path.append('/data/kas7897/dPLHBVrelease/output/ensemble8/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss60/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/pred50.npy')
path.append('/data/kas7897/dPLHBVrelease/output/ensemble9/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss60/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/pred50.npy')
path.append('/data/kas7897/dPLHBVrelease/output/ensemble10/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss60/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/pred50.npy')
path.append('/data/kas7897/dPLHBVrelease/output/ensemble1_alpha/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss60/BuffOpt0/RMSE_para0.0/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/pred50.npy')
path.append('/data/kas7897/dPLHBVrelease/output/ensemble2_alpha/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss60/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/pred50.npy')
path.append('/data/kas7897/dPLHBVrelease/output/ensemble3_alpha/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss60/BuffOpt0/RMSE_para0.5/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/pred50.npy')
path.append('/data/kas7897/dPLHBVrelease/output/ensemble4_alpha/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss60/BuffOpt0/RMSE_para0.75/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/pred50.npy')
path.append('/data/kas7897/dPLHBVrelease/output/ensemble5_alpha/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss60/BuffOpt0/RMSE_para1.0/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/pred50.npy')
# print(path)
Ttest = [19951001, 20051001]
dateTest1 = datetime.strptime(str(Ttest[0]), '%Y%m%d')
dateTest2 = datetime.strptime(str(Ttest[1]), '%Y%m%d')
delta_test = dateTest2 - dateTest1
num_days_test = delta_test.days

predtestALL = np.full([671, num_days_test, 5, len(path)], np.nan)

for i in range(len(path)):
    pred = np.load(path[i])
    predtestALL[:,:,:,i] = pred
obstestALL = np.load(
    '/data/kas7897/dPLHBVrelease/output/ensemble5_alpha/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss60/BuffOpt0/RMSE_para1.0/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/obs.npy')

predtestALL_allEn = np.mean(predtestALL, axis=3) #average all the ensemble predictions
l=1
evaDict = [stat.statError(predtestALL_allEn[:, :, 0], obstestALL.squeeze())]  # Q0: the streamflow
#
# if forType == ['daymet', 'maurer_extended', 'nldas_extended']:
#     testsave_path = 'All_ensembles_alpha_attri' + '/CAMELSDemo/dPLHBV/' + puN + '/TDTestforc/' + 'TD' + "_".join(
#         tdRepS) + '/all_extended_withloss' + str(prcp_loss_factor) + '/BuffOpt' + str(
#         buffOptOri) + '/RMSE_para' + '/' + str(testseed)
# else:
#     testsave_path = 'all_4ensemble_alpha' + '/CAMELSDemo/dPLHBV/' + puN + '/TDTestforc/' + 'TD' + "_".join(
#         tdRepS) + '/' + forType + \
#                     '/BuffOpt' + str(buffOptOri) + '/RMSE_para' + '/' + str(testseed)
#
#
#
#
# # save evaluation results
# # Note: for PUB/PUR testing, this path location saves results covering all folds. If you are only testing one fold,
# # consider add fold specification in "seStr" to prevent results overwritten from different folds.
# seStr = 'Train' + str(Ttrain[0]) + '_' + str(Ttrain[1]) + 'Test' + str(Ttest[0]) + '_' + str(
#     Ttest[1]) + 'Buff' + str(TestBuff) + 'Staind' + str(testmodel.staind)
#
# outpath = os.path.join(rootOut, testsave_path, seStr)
# if not os.path.isdir(outpath):
#     os.makedirs(outpath)
#
# EvaFile = os.path.join(outpath, 'Eva' + str(testepoch) + '.npy')
# np.save(EvaFile, evaDict)
#
# obsFile = os.path.join(outpath, 'obs.npy')
# np.save(obsFile, obstestALL)
#
# predFile = os.path.join(outpath, 'pred' + str(testepoch) + '.npy')
# np.save(predFile, predtestALL_allEn)
#
# # calculate metrics for the widely used CAMELS subset with 531 basins
# # we report on this 531 subset in Feng et al., 2022 HESSD
# subsetPath = 'Sub531ID.txt'
# with open(subsetPath, 'r') as fp:
#     sub531IDLst = json.load(fp)  # Subset 531 ID List
# get the evaluation metrics on 531 subset
# [C, ind1, SubInd] = np.intersect1d(sub531IDLst, logtestIDLst, return_indices=True)
evaframe = pd.DataFrame(evaDict[0])
# evaframeSub = evaframe.loc[SubInd, list(evaframe.keys())]
# evaS531Dict = [{col: evaframeSub[col].values for col in evaframeSub}]  # 531 subset evaDict

# print NSE median value of testing basins
# print('Testing finished! Evaluation results saved in\n', outpath)
print('For basins of whole CAMELS, NSE median:', np.nanmedian(evaDict[0]['NSE']))
# print('For basins of 531 subset, NSE median:', np.nanmedian(evaS531Dict[0]['NSE']))

## Show boxplots of the results
# evaDictLst = evaDict + evaS531Dict
evaDictLst = evaDict
plt.rcParams['font.size'] = 14
plt.rcParams["legend.columnspacing"] = 0.1
plt.rcParams["legend.handletextpad"] = 0.2
keyLst = ['NSE', 'KGE', 'lowRMSE', 'highRMSE']
dataBox = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(evaDictLst)):
        data = evaDictLst[k][statStr]
        data = data[~np.isnan(data)]
        temp.append(data)
    dataBox.append(temp)

print("NSE,KGE,lowRMSE, highRMSE, mean lowRMSE, and highRMSE of all basins: ", np.nanmedian(dataBox[0][0]),
      np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]),
      np.nanmean(dataBox[2][0]), np.nanmean(dataBox[3][0]))

labelname = ['dPL+HBV_Multi', 'dPL+HBV_Multi Sub531']
xlabel = keyLst
fig = plot.plotBoxFig(dataBox, xlabel, labelname, sharey=False, figsize=(6, 5))
fig.patch.set_facecolor('white')
fig.show()
# plt.savefig(os.path.join(outpath, 'Metric_BoxPlot.png'), format='png')