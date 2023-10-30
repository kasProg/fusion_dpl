import sys
sys.path.append('../')
from hydroDL import master, utils
from hydroDL.master import default, loadModel
from hydroDL.post import plot, stat
import matplotlib.pyplot as plt
from hydroDL.data import camels
from hydroDL.model import rnn, crit, train

import numpy as np
import pandas as pd
import os
import torch
import random
import datetime as dt
import json

# Options for different interface
interfaceOpt = 1
# ==1 default, the recommended and more interpretable version with clear data and training flow. We improved the
# original one to explicitly load and process data, set up model and loss, and train the model.
# ==0, the original "pro" version to train jobs based on the defined configuration dictionary.
# Results are very similar for two options.

# Options for training and testing
# 0: train base model without DI
# 1: train DI model
# 0,1: do both base and DI model
# 2: test trained models
Action = [0]

gpuid = 4
torch.cuda.set_device(gpuid)
# Set hyperparameters
EPOCH = 300
BATCH_SIZE = 100
RHO = 365
HIDDENSIZE = 256
saveEPOCH = 20 # save model for every "saveEPOCH" epochs
Ttrain = [19801001, 19951001]  # Training period
# Ttrain = [19981001, 20081001]
forType = 'daymet'
trainBuff = 365
loadTrain = True  # load training data

# Fix random seed
seedid = 111111
random.seed(seedid)
torch.manual_seed(seedid)
np.random.seed(seedid)
torch.cuda.manual_seed(seedid)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Change the seed to have different runnings.
# We use the mean discharge of 6 runnings with different seeds to account for randomness and report results

# Define root directory of database and output
# Modify this based on your own location of CAMELS dataset.
# Following the data download instruction in README file, you should organize the folders like
# 'your/path/to/Camels/basin_timeseries_v1p2_metForcing_obsFlow' and 'your/path/to/Camels/camels_attributes_v2.0'
# Then 'rootDatabase' here should be 'your/path/to/Camels'
# You can also define the database directory in hydroDL/__init__.py by modifying pathCamels['DB'] variable

# rootDatabase = os.path.join(os.path.sep, 'scratch', 'Camels')  # CAMELS dataset root directory: /scratch/Camels
# camels.initcamels(rootDatabase)  # initialize three camels module-scope variables in camels.py: dirDB, gageDict, statDict
#
# rootDatabase = "/data/yxs275/DPL_HBV/"
# camels.initcamels(rootDatabase)
# #rootOut = os.path.join(os.path.sep, 'data', 'rnnStreamflow')  # Root directory to save training results: /data/rnnStreamflow
# rootOut = rootDatabase + "output/LSTM/"
rootDatabase = os.path.join(os.path.sep, 'scratch', 'Camels')  # CAMELS dataset root directory: /scratch/Camels
camels.initcamels(rootDatabase)  # initialize three camels module-scope variables in camels.py: dirDB, gageDict, statDict

rootOut = os.path.join(os.path.sep, 'data', 'kas7897', 'dPLHBVrelease', 'output', 'rnnStreamflow')
# Define all the configurations into dictionary variables
# three purposes using these dictionaries. 1. saved as configuration logging file. 2. for future testing. 3. can also
# be used to directly train the model when interfaceOpt == 0

# define dataset
# default module stores default configurations, using update to change the config
# if forType is 'daymet':
#     varF = ['dayl', 'prcp', 'srad', 'tmean', 'vp']
# elif forType is 'fused_prcp':
#     varF = ['dayl', 'prcp', 'srad', 'tmax', 'vp']
# else:
#     varF = ['dayl', 'prcp', 'srad', 'tmax', 'vp'] # tmax is tmean here
varF = ['prcp', 'srad', 'tmax','tmin', 'vp']
#varF = [ 'prcp', 'tmax']
# varF = ['dayl', 'prcp', 'srad', 'tmax', 'tmin', 'vp'] # tmax is tmean here
# attrnewLst = camels.attrLstSel + ['p_mean','pet_mean','p_seasonality','frac_snow','aridity','high_prec_freq',
#                                         'high_prec_dur','low_prec_freq','low_prec_dur']
# attrLst = [ 'p_mean','pet_mean','p_seasonality','frac_snow','aridity','high_prec_freq','high_prec_dur',
#                'low_prec_freq','low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
#                'lai_diff', 'gvf_max', 'gvf_diff', 'dom_land_cover_frac', 'dom_land_cover', 'root_depth_50',
#                'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
#                'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'geol_1st_class', 'glim_1st_class_frac',
#                'geol_2nd_class', 'glim_2nd_class_frac', 'carbonate_rocks_frac', 'geol_porostiy', 'geol_permeability']
attrLst = ['elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max', 'lai_diff', 'gvf_max', 'gvf_diff',
           'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
           'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'carbonate_rocks_frac', 'geol_permeability',
           'p_mean', 'pet_mean', 'aridity', 'frac_snow', 'high_prec_freq', 'high_prec_dur', 'low_prec_freq', 'low_prec_dur']
optData = default.optDataCamels
optData = default.update(optData, varT=varF, varC=attrLst, tRange=Ttrain, forType=forType)  # Update the training period
optData['varT'] = varF
if (interfaceOpt == 1) and (loadTrain is True):
    # load training data explicitly for the interpretable interface. Notice: if you want to apply our codes to your own
    # dataset, here is the place you can replace data.
    # read data from original CAMELS dataset
    # df: CAMELS dataframe; x: forcings[nb,nt,nx]; y: streamflow obs[nb,nt,ny]; c:attributes[nb,nc]
    # nb: number of basins, nt: number of time steps (in Ttrain), nx: number of time-dependent forcing variables
    # ny: number of target variables, nc: number of constant attributes
    df = camels.DataframeCamels(
        subset=optData['subset'], tRange=optData['tRange'], forType=forType)
    x = df.getDataTs(
        varLst=optData['varT'],
        doNorm=False,
        rmNan=False)
    y = df.getDataObs(
        doNorm=False,
        rmNan=False,
        basinnorm=False)
    # transform discharge from ft3/s to mm/day and then divided by mean precip to be dimensionless.
    # output = discharge/(area*mean_precip)
    # this can also be done by setting the above option "basinnorm = True" for df.getDataObs()
    y_temp = camels.basinNorm(y, optData['subset'], toNorm=True)
    c = df.getDataConst(
        varLst=optData['varC'],
        doNorm=False,
        rmNan=False)

    # process, do normalization and remove nan
    series_data = np.concatenate([x, y_temp], axis=2)
    seriesvarLst = varF + ['runoff']

    # seriesvarLst = ['dayl', 'prcp', 'srad', 'tmax', 'vp'] + ['runoff'] # Test the case without gamma transform
    # seriesvarLst = ['dayl', 'prcp', 'srad', 'tmax', 'tmin', 'vp'] + ['runoff'] # Test the case without gamma transform

    # calculate statistics for norm and saved to a dictionary
    statDict = camels.getStatDic(attrLst=attrLst, attrdata=c, seriesLst=seriesvarLst, seriesdata=series_data)
    # normalize
    attr_norm = camels.transNormbyDic(c, attrLst, statDict, toNorm=True)
    attr_norm[np.isnan(attr_norm)] = 0.0
    series_norm = camels.transNormbyDic(series_data, seriesvarLst, statDict, toNorm=True)

    # prepare the inputs
    xTrain = series_norm[:, :, :-1]  # forcing, not include obs
    xTrain[np.isnan(xTrain)] = 0.0
    yTrain = np.expand_dims(series_norm[:, :, -1], 2)
    attrs = attr_norm


# define model and update configure
if torch.cuda.is_available():
    optModel = default.optLstm
else:
    optModel = default.update(
        default.optLstm,
        name='hydroDL.model.rnn.CpuLstmModel')
optModel = default.update(default.optLstm, hiddenSize=HIDDENSIZE)
# define loss function
optLoss = default.optLossRMSE
# optLoss = default.optLossNSE
# define training options
optTrain = default.update(default.optTrainCamels, miniBatch=[BATCH_SIZE, RHO], nEpoch=EPOCH, saveEpoch=saveEPOCH,
                          seed=seedid, trainBuff=trainBuff)

# define output folder for model results
# exp_name = 'CAMELSDemo'
# exp_disp = 'LSTMCAMELS/'+forType+'/'+str(seedid)
# save_path = os.path.join(exp_name, exp_disp, \
#             'epochs{}_batch{}_rho{}_hiddensize{}_Tstart{}_Tend{}_Buff{}'.format(optTrain['nEpoch'], optTrain['miniBatch'][0],
#                                                                           optTrain['miniBatch'][1],
#                                                                           optModel['hiddenSize'],
#                                                                           optData['tRange'][0], optData['tRange'][1],
#                                                                           optTrain['trainBuff']))
exp_name = 'CAMELSDemo-daymet-yalan_test1'
exp_disp = 'TestRun'
save_path = os.path.join(exp_name, exp_disp, \
            'epochs{}_batch{}_rho{}_hiddensize{}_Tstart{}_Tend{}'.format(optTrain['nEpoch'], optTrain['miniBatch'][0],
                                                                          optTrain['miniBatch'][1],
                                                                          optModel['hiddenSize'],
                                                                          optData['tRange'][0], optData['tRange'][1]))

# Train the base model without data integration
if 0 in Action:
    out = os.path.join(rootOut, save_path)  # output folder to save results
    # Wrap up all the training configurations to one dictionary in order to save into "out" folder
    masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
    if interfaceOpt == 1:  # use the more interpretable version interface
        nx = xTrain.shape[-1] + attrs.shape[-1]  # update nx, nx = nx + nc
        ny = yTrain.shape[-1]
        # load model for training
        if torch.cuda.is_available():
            model = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)
        else:
            model = rnn.CpuLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)
        optModel = default.update(optModel, nx=nx, ny=ny)
        # the loaded model should be consistent with the 'name' in optModel Dict above for logging purpose
        lossFun = crit.RmseLoss()
        # lossFun = crit.NSELossBatch(np.nanstd(yTrain, axis=1))

        # lossFun = crit.RmseLossRunoff()

        # alpha = 0.8
        # optLoss = default.update(default.optLossComb, name='hydroDL.model.crit.RmseLossComb', weight=alpha)
        # lossFun = crit.RmseLossComb(alpha=alpha)

        # the loaded loss should be consistent with the 'name' in optLoss Dict above for logging purpose
        # update and write the dictionary variable to out folder for logging and future testing
        masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
        master.writeMasterFile(masterDict)
        # log statistics
        statFile = os.path.join(out, 'statDict.json')
        with open(statFile, 'w') as fp:
            json.dump(statDict, fp, indent=4)
        # train model
        model = train.trainModel(
            model,
            xTrain,
            yTrain,
            attrs,
            lossFun,
            nEpoch=EPOCH,
            miniBatch=[BATCH_SIZE, RHO],
            saveEpoch=saveEPOCH,
            saveFolder=out,
            bufftime=trainBuff)
    elif interfaceOpt==0: # directly train the model using dictionary variable
        master.train(masterDict)


# Train DI model
if 1 in Action:
    nDayLst = [1]
    for nDay in nDayLst:
        # nDay: previous Nth day observation to integrate
        # update parameter "daObs" for data dictionary variable
        optData = default.update(default.optDataCamels, daObs=nDay)
        # define output folder for DI models
        out = os.path.join(rootOut, save_path, 'All-80-95-DI' + str(nDay))
        masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
        if interfaceOpt==1:
            # optData['daObs'] != 0, load previous observation data to integrate
            sd = utils.time.t2dt(
                optData['tRange'][0]) - dt.timedelta(days=nDay)
            ed = utils.time.t2dt(
                optData['tRange'][1]) - dt.timedelta(days=nDay)
            dfdi = camels.DataframeCamels(
                subset=optData['subset'], tRange=[sd, ed])
            datatemp = dfdi.getDataObs(
                doNorm=False, rmNan=False, basinnorm=True) # 'basinnorm=True': output = discharge/(area*mean_precip)
            # normalize data
            dadata = camels.transNormbyDic(datatemp, 'runoff', statDict, toNorm=True)
            dadata[np.where(np.isnan(dadata))] = 0.0

            xIn = np.concatenate([xTrain, dadata], axis=2)
            nx = xIn.shape[-1] + attrs.shape[-1]  # update nx, nx = nx + nc
            ny = yTrain.shape[-1]
            # load model for training
            if torch.cuda.is_available():
                model = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)
            else:
                model = rnn.CpuLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)
            optModel = default.update(optModel, nx=nx, ny=ny)
            lossFun = crit.RmseLoss()
            # update and write dictionary variable to out folder for logging and future testing
            masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
            master.writeMasterFile(masterDict)
            # log statistics
            statFile = os.path.join(out, 'statDict.json')
            with open(statFile, 'w') as fp:
                json.dump(statDict, fp, indent=4)
            # train model
            model = train.trainModel(
                model,
                xIn,
                yTrain,
                attrs,
                lossFun,
                nEpoch=EPOCH,
                miniBatch=[BATCH_SIZE, RHO],
                saveEpoch=saveEPOCH,
                saveFolder=out)
        elif interfaceOpt==0:
            master.train(masterDict)


# Test models
if 2 in Action:
    testTrainBuff = True
    loadtrainBuff = 0  # the length of extra data loaded in training
    tRange = [19951001, 20051001]  # Testing period
    tRangeOri = [19851001, 20051001]
    # TestBuff = len(utils.time.tRange2Array(tRange)) - len(utils.time.tRange2Array(tRangeOri))
    # TestBuff = 365
    TestBuff = xTrain.shape[1] - loadtrainBuff
    TestEPOCH = 300 # choose the model to test after trained "TestEPOCH" epoches
    caseLst = ['All-85-95']
    # nDayLst = [1, 3]  # which DI models to test: DI(1), DI(3)
    nDayLst = []  # which DI models to test: DI(1), DI(3)
    for nDay in nDayLst:
        caseLst.append('All-85-95-DI' + str(nDay))
    # outLst = [os.path.join(rootOut, save_path, x) for x in caseLst]
    # # generate a folder name list containing all the tested model output folders
    # caseLst = ['All-85-95']
    # nDayLst = [1]  # which DI models to test: DI(1), DI(3)
    # nDay = nDayLst[0]
    #optData = default.update(default.optDataCamels, daObs=nDay)
    # for nDay in nDayLst:
    #     caseLst.append('All-85-95-DI' + str(nDay))
    # outLst = [os.path.join(rootOut, save_path, x) for x in caseLst]  # outLst includes all the directories to test
    outLst = [os.path.join(rootOut, save_path)]
    #outLst = [os.path.join(rootOut, save_path, 'All-80-95-DI' + str(nDay))]
    subset = 'All'  # 'All': use all the CAMELS gages to test; Or pass the gage list
    testBatch = 100 # do batch forward to save GPU memory
    predLst = list()
    for out in outLst:
        if interfaceOpt == 1:  # use the more interpretable version interface
            # load testing data
            mDict = master.readMasterFile(out)
            optData = mDict['data']
            optData['varT'] = varF
            df = camels.DataframeCamels(
                subset=subset, tRange=tRange, forType=optData['forType'])
            x = df.getDataTs(
                varLst=optData['varT'],
                doNorm=False,
                rmNan=False)
            obs = df.getDataObs(
                doNorm=False,
                rmNan=False,
                basinnorm=False)
            c = df.getDataConst(
                varLst=optData['varC'],
                doNorm=False,
                rmNan=False)

            # do normalization and remove nan
            # load the saved statDict to make sure using the same statistics as training data
            statFile = os.path.join(out, 'statDict.json')
            with open(statFile, 'r') as fp:
                statDict = json.load(fp)
            seriesvarLst = optData['varT']
            # seriesvarLst = ['dayl', 'prcp', 'srad', 'tmax', 'vp']
            attrLst = optData['varC']
            attr_norm = camels.transNormbyDic(c, attrLst, statDict, toNorm=True)
            attr_norm[np.isnan(attr_norm)] = 0.0
            xTest = camels.transNormbyDic(x, seriesvarLst, statDict, toNorm=True)
            xTest[np.isnan(xTest)] = 0.0
            attrs = attr_norm

            if testTrainBuff is True:
                xTestBuff = xTrain[:, -TestBuff:, :]
                xTest = np.concatenate([xTestBuff, xTest], axis=1)

            if optData['daObs'] > 0:
                # optData['daObs'] != 0, load previous observation data to integrate
                nDay = optData['daObs']
                sd = utils.time.t2dt(
                    tRange[0]) - dt.timedelta(days=nDay)
                ed = utils.time.t2dt(
                    tRange[1]) - dt.timedelta(days=nDay)
                dfdi = camels.DataframeCamels(
                    subset=subset, tRange=[sd, ed])
                datatemp = dfdi.getDataObs(
                    doNorm=False, rmNan=False, basinnorm=True) # 'basinnorm=True': output = discharge/(area*mean_precip)
                # normalize data
                dadata = camels.transNormbyDic(datatemp, 'runoff', statDict, toNorm=True)
                dadata[np.where(np.isnan(dadata))] = 0.0
                xIn = np.concatenate([xTest, dadata], axis=2)

            else:
                xIn = xTest

            # load and forward the model for testing
            testmodel = loadModel(out, epoch=TestEPOCH)
            filePathLst = master.master.namePred(
                out, tRange, 'All', epoch=TestEPOCH)  # prepare the name of csv files to save testing results
            train.testModel(
                testmodel, xIn, c=attrs, batchSize=testBatch, filePathLst=filePathLst)
            # read out predictions
            dataPred = np.ndarray([xTest.shape[0], xTest.shape[1], len(filePathLst)])
            for k in range(len(filePathLst)):
                filePath = filePathLst[k]
                dataPred[:, :, k] = pd.read_csv(
                    filePath, dtype=np.float, header=None).values
            # transform back to the original observation
            temppred = camels.transNormbyDic(dataPred, 'runoff', statDict, toNorm=False)
            pred = camels.basinNorm(temppred, subset, toNorm=False)

        elif interfaceOpt == 0: # only for models trained by the pro interface
            df, pred, obs = master.test(out, tRange=tRange, subset=subset, batchSize=testBatch, basinnorm=True,
                                        epoch=TestEPOCH, reTest=True)

        # change the units ft3/s to m3/s
        if testTrainBuff is True:
            obs = obs[:, 0:, :] * 0.0283168
        else:
            obs = obs[:, TestBuff:, :] * 0.0283168

        pred = pred[:, TestBuff:, :] * 0.0283168
        # pred[pred<0] = 0.0
         # the prediction list for all the models

        obs = camels.basinTrans(obs,subset)
        pred = camels.basinTrans(pred, subset)
        predLst.append(pred)



    predFile = os.path.join(out, 'pred' + str(TestEPOCH) + '.npy')
    np.save(predFile, pred)
    # calculate statistic metrics
    evaDict = [stat.statError(x.squeeze(), obs.squeeze()) for x in predLst]


    # outpath = os.path.join(rootOut, testsave_path, seStr)
    # if not os.path.isdir(out[0]):
    #     os.makedirs(outpath)ou
    save = os.path.join(rootOut, save_path)
    EvaFile = os.path.join(save, 'Eva' + str(TestEPOCH) + '.npy')
    np.save(EvaFile, evaDict)

    gageinfo = camels.gageDict
    hucinfo = gageinfo['huc']
    gageid = gageinfo['id']
    gageidLst = gageid.tolist()


    # calculate metrics for the widely used CAMELS subset with 531 basins
    # we report on this 531 subset in Feng et al., 2022 HESSD
    subsetPath = "/data/kas7897/dPLHBVrelease/hydroDL-dev/example/dPLHBV/Sub531ID.txt"
    with open(subsetPath, 'r') as fp:
        sub531IDLst = json.load(fp)  # Subset 531 ID List
    # get the evaluation metrics on 531 subset
    [C, ind1, SubInd] = np.intersect1d(sub531IDLst, gageidLst, return_indices=True)
    evaframe = pd.DataFrame(evaDict[0])
    evaframeSub = evaframe.loc[SubInd, list(evaframe.keys())]
    evaS531Dict = [{col: evaframeSub[col].values for col in evaframeSub}]  # 531 subset evaDict

    # print NSE median value of testing basins
    # print('Testing finished! Evaluation results saved in\n', outpath)
    print('For basins of whole CAMELS, NSE median:', np.nanmedian(evaDict[0]['NSE']))
    print('For basins of 531 subset, NSE median:', np.nanmedian(evaS531Dict[0]['NSE']))
    print('For basins of 531 subset, KGE median:', np.nanmedian(evaS531Dict[0]['KGE']))

    # Show boxplots of the results
    statDictLst = evaDict
    plt.rcParams['font.size'] = 14
    keyLst = ['NSE', 'KGE','AFLV','AFHV','PBiasother','lowRMSE','highRMSE','midRMSE']
    dataBox = list()
    for iS in range(len(keyLst)):
        statStr = keyLst[iS]
        temp = list()
        for k in range(len(statDictLst)):
            data = statDictLst[k][statStr]
            data = data[~np.isnan(data)]
            temp.append(data)
        dataBox.append(temp)

    print("NSE,KGE,AFLV, AFHV, lowRMSE, highRMSE, mean lowRMSE, highRMSE of all basins: ", np.nanmedian(dataBox[0][0]),
          np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]), np.nanmedian(dataBox[5][0]), np.nanmedian(dataBox[6][0]), np.nanmean(dataBox[5][0]), np.nanmean(dataBox[6][0]))

    # # labelname = ['NLDAS', 'Maurer']
    # labelname = ['Daymet']
    # # labelname = ['LogSqrt', 'Sqrt', 'NoTrans-V0','NoTrans', 'NoTrans-V',]
    # # for nDay in nDayLst:
    # #     labelname.append('DI(' + str(nDay) + ')')
    # # xlabel = ['Bias ($\mathregular{m^3}$/s)', 'NSE', 'FLV(%)', 'FHV(%)']
    # xlabel = keyLst
    # fig = plot.plotBoxFig(dataBox, xlabel, labelname, sharey=False, figsize=(12, 5))
    # fig.patch.set_facecolor('white')
    # fig.show()
    # plt.savefig(os.path.join(rootOut, save_path, "Boxplot.png"), dpi=500)
