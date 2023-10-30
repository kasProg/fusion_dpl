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

from datetime import datetime

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
Action = [2]
gpuid = 5
torch.cuda.set_device(gpuid)

# Set hyperparameters
EPOCH = 300
BATCH_SIZE = 100
RHO = 365
HIDDENSIZE = 256
saveEPOCH = 10 # save model for every "saveEPOCH" epochs
Ttrain = [19801001, 19951001]  # Training period
# "/data/kas7897/dPLHBVrelease/output/rnnStreamflow/CAMELSDemo-Multiforcing_NSE/TestRun/epochs300_batch100_rho365_hiddensize256_Tstart19801001_Tend19951001/All-85-95/master.json"

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
rootDatabase = os.path.join(os.path.sep, 'scratch', 'Camels')  # CAMELS dataset root directory: /scratch/Camels
camels.initcamels(rootDatabase)  # initialize three camels module-scope variables in camels.py: dirDB, gageDict, statDict

rootOut = os.path.join(os.path.sep, 'data', 'kas7897', 'dPLHBVrelease', 'output', 'rnnStreamflow')  # Root directory to save training results: /data/rnnStreamflow
gageinfo = camels.gageDict
hucinfo = gageinfo['huc']
gageid = gageinfo['id']
gageidLst = gageid.tolist()
dateTrain1 = datetime.strptime(str(Ttrain[0]), '%Y%m%d')
dateTrain2 = datetime.strptime(str(Ttrain[1]), '%Y%m%d')
delta_train = dateTrain2 - dateTrain1
num_days_train = delta_train.days

# Define all the configurations into dictionary variables
# three purposes using these dictionaries. 1. saved as configuration logging file. 2. for future testing. 3. can also
# be used to directly train the model when interfaceOpt == 0
multiforcing =True # set True if you want to use multiple forcings
if multiforcing == False:
    forType = 'maurer_extended'
    # for Type defines which forcing in CAMELS to use: 'daymet', 'nldas', 'maurer'
else:
    # forType = ['daymet']
    forType = ['daymet', 'maurer_extended', 'nldas_extended']
# define dataset
# default module stores default configurations, using update to change the config
optData = default.optDataCamels
# print(camels.attrLstSel)
kratzert_attr = ['elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max', 'lai_diff', 'gvf_max', 'gvf_diff', 'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity', 'max_water_content',
                 'sand_frac', 'silt_frac', 'clay_frac', 'carbonate_rocks_frac', 'geol_permeability', 'p_mean', 'pet_mean', 'aridity', 'frac_snow', 'high_prec_freq', 'high_prec_dur', 'low_prec_freq', 'low_prec_dur']
# optData = default.update(optData, varT=camels.forcingLst, varC=camels.attrLstSel, tRange=Ttrain, forType='daymet')
if type(forType) == list:
    if 2 in Action:
        # seriesvarLst = ['dayl', 'prcp_daymet', 'prcp_maurer', 'prcp_nldas', 'srad', 'tmax', 'tmin', 'vp', 'runoff']
        seriesvarLst = ['dayl_daymet', 'dayl_maurer', 'dayl_nldas', 'prcp_daymet', 'prcp_maurer', 'prcp_nldas',
                        'srad_daymet', 'srad_maurer', 'srad_nldas', 'tmax_daymet', 'tmax_maurer', 'tmax_nldas',
                        'tmin_daymet', 'tmin_maurer', 'tmin_nldas', 'vp_daymet', 'vp_maurer', 'vp_nldas', 'runoff']
        optData = default.update(optData, varT=seriesvarLst, varC=camels.attrLstSel, tRange=Ttrain)
        # optData = default.update(optData, varT=seriesvarLst, varC=kratzert_attr, tRange=Ttrain)

    if (interfaceOpt == 1) and (2 not in Action):
        # forcUN = np.empty([len(gageidLst), num_days_train, len(camels.forcingLst) + len(forType) - 1])
        forcUN = np.empty([len(gageidLst), num_days_train, len(camels.forcingLst)*len(forType)])
        # prcp_forc = np.empty([len(gageidLst), num_days_train, len(forType)])

        for i in range(len(forType)):
            if 'daymet' in forType:
                # optData = default.update(optData, varT=camels.forcingLst, varC=camels.attrLstSel, tRange=Ttrain, forType = 'daymet')
                optData = default.update(optData, varT=camels.forcingLst, varC=kratzert_attr, tRange=Ttrain, forType = 'daymet')
            elif 'nldas' in forType:
                # optData = default.update(optData, tRange=Ttrain, varT=camels.forcingLst, varC=camels.attrLstSel, forType='nldas')
                optData = default.update(optData, tRange=Ttrain, varT=camels.forcingLst, varC=kratzert_attr, forType='nldas')
            elif 'nldas_extended' in forType:
                # optData = default.update(optData, tRange=Ttrain, varT=camels.forcingLst, varC=camels.attrLstSel, forType='nldas_extended')
                optData = default.update(optData, tRange=Ttrain, varT=camels.forcingLst, varC=kratzert_attr, forType='nldas_extended')
            else:
                # optData = default.update(optData, tRange=Ttrain, varT=camels.forcingLst, varC=camels.attrLstSel, forType=forType[0])
                optData = default.update(optData, tRange=Ttrain, varT=camels.forcingLst, varC=kratzert_attr, forType=forType[0])


            dfTrain = camels.DataframeCamels(subset=optData['subset'], tRange=optData['tRange'], forType=forType[i])
            forcUN_type = dfTrain.getDataTs(varLst=optData['varT'],doNorm=False,rmNan=False)

            # prcp_forc[:,:,i] = forcUN_type[:, :, 0]
            forcUN[:,:,i] = forcUN_type[:, :, 0]
            forcUN[:,:,i+3] = forcUN_type[:, :, 1]
            forcUN[:,:,i+6] = forcUN_type[:, :, 2]
            forcUN[:,:,i+9] = forcUN_type[:, :, 3]
            forcUN[:,:,i+12] = forcUN_type[:, :, 4]
            forcUN[:,:,i+15] = forcUN_type[:, :, 5]

            # if forType[i] == 'daymet':
            #     daymet_other_forc = forcUN_type[:, :, np.arange(forcUN_type.shape[2]) !=1]
            # if forType[i] == 'nldas' or forType[i] == 'nldas_extended':
            #     nldas_other_forc = forcUN_type[:, :, np.arange(forcUN_type.shape[2]) != 1]
            # if forType[i] == 'maurer' or forType[i] == 'maurer_extended':
            #     maurer_other_forc = forcUN_type[:, :, np.arange(forcUN_type.shape[2]) != 1]

        obsUN = dfTrain.getDataObs(doNorm=False, rmNan=False, basinnorm=False)
        attrsUN = dfTrain.getDataConst(varLst=optData['varC'], doNorm=False, rmNan=False)

        # forcUN[:, :, 1:len(forType)+1] = prcp_forc
        # if 'daymet' in forType:
        #     forcUN[:, :, 0] = daymet_other_forc[:,:,0]
        #     forcUN[:, :, len(forType)+1:] = daymet_other_forc[:,:,1:]
        # elif 'nldas' in forType or 'nldas_extended' in forType:
        #     forcUN[:, :, 0] = nldas_other_forc[:, :, 0]
        #     forcUN[:, :, len(forType) + 1:] = nldas_other_forc[:, :, 1:]
        # elif 'maurer' in forType or 'maurer_extended' in forType:
        #     forcUN[:, :, 0] = maurer_other_forc[:, :, 0]
        #     forcUN[:, :, len(forType) + 1:] = maurer_other_forc[:, :, 1:]

        y_temp = camels.basinNorm(obsUN, optData['subset'], toNorm=True)
        series_data = np.concatenate([forcUN, y_temp], axis=2)
        # seriesvarLst = camels.forcingLst + ['runoff']
        # seriesvarLst = ['dayl', 'prcp_daymet', 'prcp_maurer', 'prcp_nldas', 'srad', 'tmax', 'tmin', 'vp', 'runoff']
        seriesvarLst = ['dayl_daymet','dayl_maurer', 'dayl_nldas', 'prcp_daymet', 'prcp_maurer', 'prcp_nldas',
                        'srad_daymet', 'srad_maurer', 'srad_nldas', 'tmax_daymet', 'tmax_maurer', 'tmax_nldas',
                        'tmin_daymet','tmin_maurer', 'tmin_nldas',  'vp_daymet', 'vp_maurer', 'vp_nldas', 'runoff']
        # statDict = camels.getStatDic(attrLst=camels.attrLstSel, attrdata=attrsUN, seriesLst=seriesvarLst, seriesdata=series_data)
        statDict = camels.getStatDic(attrLst=kratzert_attr, attrdata=attrsUN, seriesLst=seriesvarLst, seriesdata=series_data)
        # normalize
        # attr_norm = camels.transNormbyDic(attrsUN, camels.attrLstSel, statDict, toNorm=True)
        attr_norm = camels.transNormbyDic(attrsUN, kratzert_attr, statDict, toNorm=True)
        attr_norm[np.isnan(attr_norm)] = 0.0
        series_norm = camels.transNormbyDic(series_data, seriesvarLst, statDict, toNorm=True)

        # prepare the inputs
        xTrain = series_norm[:, :, :-1]  # forcing, not include obs
        xTrain[np.isnan(xTrain)] = 0.0
        yTrain = np.expand_dims(series_norm[:, :, -1], 2)
        attrs = attr_norm
else:

    optData = default.update(optData, varT=camels.forcingLst, varC=camels.attrLstSel, tRange=Ttrain, forType=forType)  # Update the training period

    if (interfaceOpt == 1) and (2 not in Action):
        # load training data explicitly for the interpretable interface. Notice: if you want to apply our codes to your own
        # dataset, here is the place you can replace data.
        # read data from original CAMELS dataset
        # df: CAMELS dataframe; x: forcings[nb,nt,nx]; y: streamflow obs[nb,nt,ny]; c:attributes[nb,nc]
        # nb: number of basins, nt: number of time steps (in Ttrain), nx: number of time-dependent forcing variables
        # ny: number of target variables, nc: number of constant attributes
        df = camels.DataframeCamels(
            subset=optData['subset'], tRange=optData['tRange'])
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
        seriesvarLst = camels.forcingLst + ['runoff']
        # calculate statistics for norm and saved to a dictionary
        statDict = camels.getStatDic(attrLst=camels.attrLstSel, attrdata=c, seriesLst=seriesvarLst, seriesdata=series_data)
        # normalize
        attr_norm = camels.transNormbyDic(c, camels.attrLstSel, statDict, toNorm=True)
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
optTrain = default.update(default.optTrainCamels, miniBatch=[BATCH_SIZE, RHO], nEpoch=EPOCH, saveEpoch=saveEPOCH, seed=seedid)

# define output folder for model results
if multiforcing is True:
    exp_name = 'CAMELSDemo-Multiforcing-All'
    exp_disp = 'TestRun'
    save_path = os.path.join(exp_name, exp_disp, \
                             'epochs{}_batch{}_rho{}_hiddensize{}_Tstart{}_Tend{}'.format(optTrain['nEpoch'],
                                                                                          optTrain['miniBatch'][0],
                                                                                          optTrain['miniBatch'][1],
                                                                                          optModel['hiddenSize'],
                                                                                          Ttrain[0],
                                                                                          Ttrain[1]))
else:
    exp_name = 'CAMELSDemo-maurer'
    exp_disp = 'TestRun'
    save_path = os.path.join(exp_name, exp_disp, \
                'epochs{}_batch{}_rho{}_hiddensize{}_Tstart{}_Tend{}'.format(optTrain['nEpoch'], optTrain['miniBatch'][0],
                                                                              optTrain['miniBatch'][1],
                                                                              optModel['hiddenSize'],
                                                                              optData['tRange'][0], optData['tRange'][1]))

# Train the base model without data integration
if 0 in Action:
    out = os.path.join(rootOut, save_path, 'All-85-95')  # output folder to save results
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
        # lossFun = crit.NSELosstest()
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
            saveFolder=out)
    elif interfaceOpt==0: # directly train the model using dictionary variable
        master.train(masterDict)


# Train DI model
if 1 in Action:
    nDayLst = [1,3]
    for nDay in nDayLst:
        # nDay: previous Nth day observation to integrate
        # update parameter "daObs" for data dictionary variable
        optData = default.update(default.optDataCamels, daObs=nDay)
        # define output folder for DI models
        out = os.path.join(rootOut, save_path, 'All-85-95-DI' + str(nDay))
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
            # lossFun = crit.RmseLoss()
            lossFun = crit.NSELosstest()
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
    TestEPOCH = 300 # choose the model to test after trained "TestEPOCH" epoches
    # generate a folder name list containing all the tested model output folders
    caseLst = ['All-85-95']
    # nDayLst = [1, 3]  # which DI models to test: DI(1), DI(3)
    nDayLst = []  # which DI models to test: DI(1), DI(3)
    for nDay in nDayLst:
        caseLst.append('All-85-95-DI' + str(nDay))
    outLst = [os.path.join(rootOut, save_path, x) for x in caseLst]  # outLst includes all the directories to test
    subset = 'All'  # 'All': use all the CAMELS gages to test; Or pass the gage list
    tRange = [19951001, 20081001]  # Testing period
    dateTest1 = datetime.strptime(str(tRange[0]), '%Y%m%d')
    dateTest2 = datetime.strptime(str(tRange[1]), '%Y%m%d')
    delta_test = dateTest2 - dateTest1
    num_days_test = delta_test.days

    testBatch = 100 # do batch forward to save GPU memory
    predLst = list()
    for out in outLst:
        if interfaceOpt == 1:  # use the more interpretable version interface
            # load testing data
            mDict = master.readMasterFile(out)
            optData = mDict['data']
            if type(forType) == list:
                # forcUN = np.empty([len(gageidLst), num_days_test, len(camels.forcingLst) + len(forType) - 1])
                forcUN = np.empty([len(gageidLst), num_days_test, len(camels.forcingLst)*len(forType)])
                prcp_forc = np.empty([len(gageidLst), num_days_test, len(forType)])

                for i in range(len(forType)):
                    if 'daymet' in forType:
                        optData = default.update(optData, varT=camels.forcingLst, varC=camels.attrLstSel, tRange=tRange,
                                                 forType='daymet')
                        # optData = default.update(optData, varT=camels.forcingLst, varC=kratzert_attr, tRange=tRange,
                        #                          forType='daymet')
                    elif 'nldas' in forType:
                        optData = default.update(optData, tRange=tRange, varT=camels.forcingLst, varC=camels.attrLstSel,
                                                 forType='nldas')
                        # optData = default.update(optData, tRange=tRange, varT=camels.forcingLst, varC=kratzert_attr,
                        #                          forType='nldas')
                    elif 'nldas_extended' in forType:
                        optData = default.update(optData, tRange=tRange, varT=camels.forcingLst, varC=camels.attrLstSel,
                                                 forType='nldas_extended')
                        # optData = default.update(optData, tRange=tRange, varT=camels.forcingLst, varC=kratzert_attr,
                        #                          forType='nldas_extended')
                    else:
                        optData = default.update(optData, tRange=tRange, varT=camels.forcingLst, varC=camels.attrLstSel,
                                                 forType=forType[0])
                        # optData = default.update(optData, tRange=tRange, varT=camels.forcingLst, varC=kratzert_attr,
                        #                          forType=forType[0])

                    dfTrain = camels.DataframeCamels(subset=optData['subset'], tRange=tRange,
                                                     forType=forType[i])
                    forcUN_type = dfTrain.getDataTs(varLst=optData['varT'], doNorm=False, rmNan=False)

                    # prcp_forc[:, :, i] = forcUN_type[:, :, 1]
                    forcUN[:, :, i] = forcUN_type[:, :, 0]
                    forcUN[:, :, i + 3] = forcUN_type[:, :, 1]
                    forcUN[:, :, i + 6] = forcUN_type[:, :, 2]
                    forcUN[:, :, i + 9] = forcUN_type[:, :, 3]
                    forcUN[:, :, i + 12] = forcUN_type[:, :, 4]
                    forcUN[:, :, i + 15] = forcUN_type[:, :, 5]
                    # if forType[i] == 'daymet':
                    #     daymet_other_forc = forcUN_type[:, :, np.arange(forcUN_type.shape[2]) != 1]
                    # if forType[i] == 'nldas' or forType[i] == 'nldas_extended':
                    #     nldas_other_forc = forcUN_type[:, :, np.arange(forcUN_type.shape[2]) != 1]
                    # if forType[i] == 'maurer' or forType[i] == 'maurer_extended':
                    #     maurer_other_forc = forcUN_type[:, :, np.arange(forcUN_type.shape[2]) != 1]

                obsUN = dfTrain.getDataObs(doNorm=False, rmNan=False, basinnorm=False)
                attrsUN = dfTrain.getDataConst(varLst=optData['varC'], doNorm=False, rmNan=False)

                # forcUN[:, :, 1:len(forType) + 1] = prcp_forc
                # if 'daymet' in forType:
                #     forcUN[:, :, 0] = daymet_other_forc[:, :, 0]
                #     forcUN[:, :, len(forType) + 1:] = daymet_other_forc[:, :, 1:]
                # elif 'nldas' in forType or 'nldas_extended' in forType:
                #     forcUN[:, :, 0] = nldas_other_forc[:, :, 0]
                #     forcUN[:, :, len(forType) + 1:] = nldas_other_forc[:, :, 1:]
                # elif 'maurer' in forType or 'maurer_extended' in forType:
                #     forcUN[:, :, 0] = maurer_other_forc[:, :, 0]
                #     forcUN[:, :, len(forType) + 1:] = maurer_other_forc[:, :, 1:]

                x = forcUN
                obs = obsUN
                c = attrsUN
            else:
                df = camels.DataframeCamels(
                    subset=subset, tRange=tRange)
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
            # seriesvarLst = optData['varT']
            # seriesvarLst = ['dayl', 'prcp_daymet', 'prcp_maurer', 'prcp_nldas', 'srad', 'tmax', 'tmin', 'vp']
            # seriesvarLst = ['dayl', 'prcp_daymet', 'prcp_maurer', 'prcp_nldas', 'srad', 'tmax', 'tmin', 'vp']
            seriesvarLst = ['dayl_daymet', 'dayl_maurer', 'dayl_nldas', 'prcp_daymet', 'prcp_maurer', 'prcp_nldas',
                            'srad_daymet', 'srad_maurer', 'srad_nldas', 'tmax_daymet', 'tmax_maurer', 'tmax_nldas',
                            'tmin_daymet', 'tmin_maurer', 'tmin_nldas', 'vp_daymet', 'vp_maurer', 'vp_nldas']
            attrLst = optData['varC']
            attr_norm = camels.transNormbyDic(c, attrLst, statDict, toNorm=True)
            attr_norm[np.isnan(attr_norm)] = 0.0
            xTest = camels.transNormbyDic(x, seriesvarLst, statDict, toNorm=True)
            xTest[np.isnan(xTest)] = 0.0
            attrs = attr_norm

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
            dataPred = np.ndarray([obs.shape[0], obs.shape[1], len(filePathLst)])
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
        # obs = obs * 0.0283168
        # pred = pred * 0.0283168
        # changing the units ft3/s to mm/day
        areas = gageinfo['area']  # unit km2
        temparea = np.tile(areas[:, None, None], (1, obs.shape[1], 1))
        obs = (obs * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10 ** 3
        pred = (pred * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10 ** 3
        predLst.append(pred) # the prediction list for all the models

    # calculate statistic metrics
    statDictLst = [stat.statError(x.squeeze(), obs.squeeze()) for x in predLst]
    outpath = os.path.join(rootOut, save_path)
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    obsFile = os.path.join(outpath, 'obs.npy')
    np.save(obsFile, obs)

    predFile = os.path.join(outpath, 'pred.npy')
    np.save(predFile, predLst)
    # Show boxplots of the results
    plt.rcParams['font.size'] = 14
    # keyLst = ['Bias', 'NSE', 'FLV', 'FHV']
    keyLst = ['NSE', 'KGE', 'lowRMSE', 'highRMSE', 'FLV', 'FHV', 'RMSE', 'Corr']

    dataBox = list()
    for iS in range(len(keyLst)):
        statStr = keyLst[iS]
        temp = list()
        for k in range(len(statDictLst)):
            data = statDictLst[k][statStr]
            data = data[~np.isnan(data)]
            temp.append(data)
        dataBox.append(temp)

    print("without-DI: NSE,KGE,lowRMSE, highRMSE, mean lowRMSE, highRMSE, FLV, FHV, RMSE, and Corr of all basins: ", np.nanmedian(dataBox[0][0]),
              np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]),
              np.nanmean(dataBox[2][0]), np.nanmean(dataBox[3][0]), np.nanmedian(dataBox[4][0]), np.nanmedian(dataBox[5][0]),
          np.nanmedian(dataBox[6][0]), np.nanmedian(dataBox[7][0]))

    # print("DI1: NSE,KGE,lowRMSE, highRMSE, mean lowRMSE, and highRMSE of all basins: ", np.nanmedian(dataBox[0][1]),
    #           np.nanmedian(dataBox[1][1]), np.nanmedian(dataBox[2][1]), np.nanmedian(dataBox[3][0]),
    #           np.nanmean(dataBox[2][1]), np.nanmean(dataBox[3][1]))
    # print("DI3: NSE,KGE,lowRMSE, highRMSE, mean lowRMSE, and highRMSE of all basins: ", np.nanmedian(dataBox[0][2]),
    #       np.nanmedian(dataBox[1][2]), np.nanmedian(dataBox[2][2]), np.nanmedian(dataBox[3][2]),
    #       np.nanmean(dataBox[2][2]), np.nanmean(dataBox[3][2]))
    labelname = ['LSTM']
    for nDay in nDayLst:
        labelname.append('DI(' + str(nDay) + ')')
    xlabel = ['Bias ($\mathregular{m^3}$/s)', 'NSE', 'FLV(%)', 'FHV(%)']
    fig = plot.plotBoxFig(dataBox, xlabel, labelname, sharey=False, figsize=(12, 5))
    fig.patch.set_facecolor('white')
    fig.show()
    # plt.savefig(os.path.join(rootOut, save_path, "Boxplot.png"), dpi=500)

    # Plot timeseries and locations
    plt.rcParams['font.size'] = 12
    # get Camels gages info
    gageinfo = camels.gageDict
    gagelat = gageinfo['lat']
    gagelon = gageinfo['lon']
    # randomly select 7 gages to plot
    gageindex = np.random.randint(0, 671, size=7).tolist()
    plat = gagelat[gageindex]
    plon = gagelon[gageindex]
    t = utils.time.tRange2Array(tRange)
    fig, axes = plt.subplots(4,2, figsize=(12,10), constrained_layout=True)
    axes = axes.flat
    npred = 2  # plot the first two prediction: Base LSTM and DI(1)
    subtitle = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(k)', '(l)']
    txt = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'k']
    # ylabel = 'Flow rate ($\mathregular{m^3}$/s)'
    # for k in range(len(gageindex)):
    #     iGrid = gageindex[k]
    #     yPlot = [obs[iGrid, :]]
    #     for y in predLst[0:npred]:
    #         yPlot.append(y[iGrid, :])
    #     # get the NSE value of LSTM and DI(1) model
    #     NSE_LSTM = str(round(statDictLst[0]['NSE'][iGrid], 2))
    #     # NSE_DI1 = str(round(statDictLst[1]['NSE'][iGrid], 2))
    #     # plot time series
    #     plot.plotTS(
    #         t,
    #         yPlot,
    #         ax=axes[k],
    #         cLst='kbrmg',
    #         markerLst='---',
    #         # legLst=['USGS', 'LSTM: '+NSE_LSTM, 'DI(1): '+NSE_DI1], title=subtitle[k], linespec=['-',':',':'], ylabel=ylabel)
    #         legLst=['USGS', 'LSTM: '+NSE_LSTM], title=subtitle[k], linespec=['-',':',':'], ylabel=ylabel)
    # # plot gage location
    # plot.plotlocmap(plat, plon, ax=axes[-1], baclat=gagelat, baclon=gagelon, title=subtitle[-1], txtlabel=txt)
    # fig.patch.set_facecolor('white')
    # fig.show()
    # plt.savefig(os.path.join(rootOut, save_path, "/Timeseries.png"), dpi=500)

    # # Plot NSE spatial patterns
    # gageinfo = camels.gageDict
    # gagelat = gageinfo['lat']
    # gagelon = gageinfo['lon']
    # nDayLst = [1, 3]
    # fig, axs = plt.subplots(3,1, figsize=(8,8), constrained_layout=True)
    # axs = axs.flat
    # data = statDictLst[0]['NSE']
    # plot.plotMap(data, ax=axs[0], lat=gagelat, lon=gagelon, title='(a) LSTM', cRange=[0.0, 1.0], shape=None)
    # data = statDictLst[1]['NSE']
    # plot.plotMap(data, ax=axs[1], lat=gagelat, lon=gagelon, title='(b) DI(1)', cRange=[0.0, 1.0], shape=None)
    # deltaNSE = statDictLst[1]['NSE'] - statDictLst[0]['NSE']
    # plot.plotMap(deltaNSE, ax=axs[2], lat=gagelat, lon=gagelon, title='(c) Delta NSE', shape=None)
    # fig.show()
    # plt.savefig(os.path.join(rootOut, save_path, "/NSEPattern.png"), dpi=500)
