import sys
sys.path.append('../../')
from hydroDL import master, utils
from hydroDL.data import camels
from hydroDL.master import loadModel
from hydroDL.model import train
from hydroDL.post import plot, stat

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import random
import json
import datetime as dt

from datetime import datetime



## fix the random seeds
randomseed = 111111
random.seed(randomseed)
torch.manual_seed(randomseed)
np.random.seed(randomseed)
torch.cuda.manual_seed(randomseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

## GPU setting
testgpuid = 6
torch.cuda.set_device(testgpuid)

## setting options, keep the same as your training
PUOpt = 0  # 0 for All; 1 for PUB; 2 for PUR;
buffOptOri = 0  # original buffOpt, Same as what you set for training
buffOpt = 0  # control load training data 0: do nothing; 1: repeat first year; 2: load one more year
multiforcing = True # set True if you want to use multiple forcings
if multiforcing == False:
    forType = 'maurer'
    # for Type defines which forcing in CAMELS to use: 'daymet', 'nldas', 'maurer'
else:
    # forType = ['nldas_extended', 'maurer_extended']
    forType = ['daymet', 'maurer_extended', 'nldas_extended']
    # forType = ['nldas_extended']
    # forType = ['maurer_extended']

smooth_loss_factor = 0
prcp_loss_factor = 23 #used only when multiforcing is True; else does not matter


## Hyperparameters, keep the same as your training setup
BATCH_SIZE = 100
RHO = 365
HIDDENSIZE = 256
Ttrain = [19801001, 19951001]  # Training period
# Ttrain = [19891001, 19991001]  # PUB/PUR period
Tinv = [19801001, 19951001] # dPL Inversion period
# Tinv = [19891001, 19991001]  # PUB/PUR period
Nfea = 13 # number of HBV parameters, 13 includes the added one for ET eq
BUFFTIME = 365
routing = True
Nmul = 16
comprout = False
compwts = False
pcorr = None

alpha = 0.25


tdRep = [1, 13] # index of dynamic parameters
tdRepS = [str(ix) for ix in tdRep]
dydrop = 0.0 # the possibility to make dynamic become static; 0.0, all dynamic; 1.0, all static
staind = -1

## Testing parameters
Ttest = [19951001, 20051001]  # testing period
# Ttest = [19891001, 19991001]  # PUB/PUR period
TtestLst = utils.time.tRange2Array(Ttest)
TtestLoad = [19951001, 20051001]  # could potentially use this to load more forcings before testing period as warm-up
# TtestLoad = [19801001, 19991001]  # PUB/PUR period
testbatch = 15  # forward number of "testbatch" basins each time to save GPU memory. You can set this even smaller to save more.
testepoch = 50

testseed = 111111

# Define root directory of database and saved output dir
# Modify this based on your own location of CAMELS dataset and saved models
rootDatabase = os.path.join(os.path.sep, 'scratch', 'Camels')  # CAMELS dataset root directory
camels.initcamels(rootDatabase)  # initialize three camels module-scope variables in camels.py: dirDB, gageDict, statDict

rootOut = os.path.join(os.path.sep, 'data', 'kas7897', 'dPLHBVrelease', 'output')  # Model output root directory

# Convert the date strings to datetime objects
dateTrain1 = datetime.strptime(str(Ttrain[0]), '%Y%m%d')
dateTrain2 = datetime.strptime(str(Ttrain[1]), '%Y%m%d')
delta_train = dateTrain2 - dateTrain1
num_days_train = delta_train.days

dateTest1 = datetime.strptime(str(Ttest[0]), '%Y%m%d')
dateTest2 = datetime.strptime(str(Ttest[1]), '%Y%m%d')
delta_test = dateTest2 - dateTest1
num_days_test = delta_test.days

# CAMLES basin info
gageinfo = camels.gageDict
hucinfo = gageinfo['huc']
gageid = gageinfo['id']
gageidLst = gageid.tolist()

# same as training, load data based on ALL, PUB, PUR scenarios
if PUOpt == 0: # for All the basins
    puN = 'ALL'
    tarIDLst = [gageidLst]

elif PUOpt == 1: # for PUB
    puN = 'PUB'
    # load the subset ID
    # splitPath saves the basin ID of random groups
    splitPath = 'PUBsplitLst.txt'
    with open(splitPath, 'r') as fp:
        testIDLst=json.load(fp)
    tarIDLst = testIDLst

elif PUOpt == 2: # for PUR
    puN = 'PUR'
    # Divide CAMELS dataset into 7 PUR regions
    # get the id list of each region
    regionID = list()
    regionNum = list()
    regionDivide = [ [1,2], [3,6], [4,5,7], [9,10], [8,11,12,13], [14,15,16,18], [17] ] # seven regions
    for ii in range(len(regionDivide)):
        tempcomb = regionDivide[ii]
        tempregid = list()
        for ih in tempcomb:
            tempid = gageid[hucinfo==ih].tolist()
            tempregid = tempregid + tempid
        regionID.append(tempregid)
        regionNum.append(len(tempregid))
    tarIDLst = regionID

# define the matrix to save results
predtestALL = np.full([len(gageid), len(TtestLst), 5], np.nan)
obstestALL = np.full([len(gageid), len(TtestLst), 1], np.nan)

# this testsave_path should be consistent with where you save your model
if forType == ['daymet', 'maurer_extended', 'nldas_extended']:
    testsave_path = 'CAMELSDemo/LSTM-dPLHBV/' + puN + '/TDTestforc/' + 'TD' + "_".join(tdRepS) + '/allprcp_7withloss'+str(prcp_loss_factor)+'smooth'+str(smooth_loss_factor)+'/BuffOpt' + str(buffOptOri) + '/RMSE_para'+ str(alpha) + '/' + str(testseed)
elif forType == ['daymet', 'maurer', 'nldas']:
    testsave_path = 'CAMELSDemo/dPLHBV/' + puN + '/TDTestforc/' + 'TD' + "_".join(tdRepS) + '/all_withloss'+str(prcp_loss_factor)+'smooth'+str(smooth_loss_factor)+'/BuffOpt' + str(buffOptOri) + '/RMSE_para'+ str(alpha) + '/' + str(testseed)
elif type(forType) == list:
    forType_string = '|'.join(forType)
    testsave_path = 'CAMELSDemo/dPLHBV/' + puN + '/TDTestforc/' + 'TD' + "_".join(tdRepS) +'/'+ forType_string + 'withloss'+str(prcp_loss_factor)+'smooth'+str(smooth_loss_factor)+'/BuffOpt' + str(buffOptOri) + '/RMSE_para'+ str(alpha) + '/' + str(testseed)
else:
    testsave_path = 'CAMELSDemo/dPLHBV/' + puN + '/TDTestforc/' + 'TD'+"_".join(tdRepS)+'/' + forType + \
                    '/BuffOpt' + str(buffOptOri) + '/RMSE_para'+str(alpha)+'/'+str(testseed)
## load data and test the model
nstart = 0
logtestIDLst = []
# loop to test all the folds for PUB and PUR. The default is you need to have run all folds, but if
# you only run one fold for PUB or PUR and just want to test that fold (i.e. fold X), you may set this as:
# for ifold in range(X, X+1):
for ifold in range(1, len(tarIDLst)+1):
    testfold = ifold
    TestLS = tarIDLst[testfold - 1]
    TestInd = [gageidLst.index(j) for j in TestLS]
    if PUOpt == 0:  # Train and test on ALL basins
        TrainLS = gageidLst
        TrainInd = [gageidLst.index(j) for j in TrainLS]
    else:
        TrainLS = list(set(gageid.tolist()) - set(TestLS))
        TrainInd = [gageidLst.index(j) for j in TrainLS]
    gageDic = {'TrainID':TrainLS, 'TestID':TestLS}

    nbasin = len(TestLS) # number of basins for testing

    # get the dir path of the saved model for testing
    foldstr = 'Fold' + str(testfold)
    exp_info = 'T_'+str(Ttrain[0])+'_'+str(Ttrain[1])+'_BS_'+str(BATCH_SIZE)+'_HS_'+str(HIDDENSIZE)\
               +'_RHO_'+str(RHO)+'_NF_'+str(Nfea)+'_Buff_'+str(BUFFTIME)+'_Mul_'+str(Nmul)
    # the final path to test with the trained model saved in
    testout = os.path.join(rootOut, testsave_path, foldstr, exp_info)
    testmodel = loadModel(testout, epoch=testepoch)

    # apply buffOpt for loading the training data
    if buffOpt == 2: # load more "BUFFTIME" forcing before the training period
        sd = utils.time.t2dt(Ttrain[0]) - dt.timedelta(days=BUFFTIME)
        sdint = int(sd.strftime("%Y%m%d"))
        TtrainLoad = [sdint, Ttrain[1]]
        TinvLoad = [sdint, Ttrain[1]]
    else:
        TtrainLoad = Ttrain
        TinvLoad = Tinv

    # prepare input data
    # load camels dataset
    if forType is 'daymet' or forType==['daymet', 'maurer_extended', 'nldas_extended']:
        varF = ['prcp', 'tmean']
        varFInv = ['prcp', 'tmean']
    else:
        varF = ['prcp', 'tmax']  # tmax is tmean here for the original CAMELS maurer and nldas forcing
        varFInv = ['prcp', 'tmax']

    attrnewLst = [ 'p_mean','pet_mean','p_seasonality','frac_snow','aridity','high_prec_freq','high_prec_dur',
                   'low_prec_freq','low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
                   'lai_diff', 'gvf_max', 'gvf_diff', 'dom_land_cover_frac', 'dom_land_cover', 'root_depth_50',
                   'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
                   'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'geol_1st_class', 'glim_1st_class_frac',
                   'geol_2nd_class', 'glim_2nd_class_frac', 'carbonate_rocks_frac', 'geol_porostiy', 'geol_permeability']
    # attrWghts = ['p_mean', 'pet_mean', 'p_seasonality', 'frac_snow', 'aridity', 'high_prec_freq', 'high_prec_dur',
    #               'low_prec_freq', 'low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
    #               'lai_diff', 'gvf_max', 'gvf_diff', 'dom_land_cover_frac', 'dom_land_cover', 'root_depth_50',
    #               'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
    #               'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'geol_1st_class', 'glim_1st_class_frac',
    #               'geol_2nd_class', 'glim_2nd_class_frac', 'carbonate_rocks_frac', 'geol_porostiy', 'geol_permeability']
    attrWghts = ['p_mean', 'pet_mean', 'p_seasonality', 'frac_snow', 'aridity', 'high_prec_freq', 'high_prec_dur',
                 'low_prec_freq', 'low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
                 'lai_diff', 'gvf_max', 'gvf_diff']

    # attrWghts = ['frac_snow', 'aridity', 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
    #              'lai_diff', 'gvf_max', 'gvf_diff']
    if type(forType) == list:
        # forcUN = np.empty([len(gageidLst), num_days_train, 1 + len(forType)])
        forcUN = np.empty([len(gageidLst), num_days_train, len(forType)*2])
        # forcInvUN = np.empty([len(gageidLst), num_days_train, 1 + len(forType)])
        forcInvUN = np.empty([len(gageidLst), num_days_train, len(forType)*2])
        # forcTestUN = np.empty([len(gageidLst), num_days_test, 1 + len(forType)])
        forcTestUN = np.empty([len(gageidLst), num_days_test, len(forType)*2])

        for i in range(len(forType)):
            if forType[i] == 'daymet':
                varF = ['prcp', 'tmean']
                varFInv = ['prcp', 'tmean']
            else:
                varF = ['prcp', 'tmax']  # For CAMELS maurer and nldas forcings, tmax is actually tmean
                varFInv = ['prcp', 'tmax']

            dfTrain = camels.DataframeCamels(tRange=TtrainLoad, subset=TrainLS, forType=forType[i])
            forcUN_type = dfTrain.getDataTs(varLst=varF, doNorm=False, rmNan=False)

            dfInv = camels.DataframeCamels(tRange=TinvLoad, subset=TrainLS, forType=forType[i])
            forcInvUN_type = dfInv.getDataTs(varLst=varFInv, doNorm=False, rmNan=False)

            dfTest = camels.DataframeCamels(tRange=TtestLoad, subset=TestLS, forType=forType[i])
            forcTestUN_type = dfTest.getDataTs(varLst=varF, doNorm=False, rmNan=False)

            # forcUN[:, :, i] = forcUN_type[:, :, 0]
            # forcInvUN[:, :, i] = forcInvUN_type[:, :, 0]
            # forcTestUN[:, :, i] = forcTestUN_type[:, :, 0]
            # forcUN[:, :, -1] = forcUN_type[:, :, 1]
            # forcInvUN[:, :, -1] = forcInvUN_type[:, :, 1]
            # forcTestUN[:, :, -1] = forcTestUN_type[:, :, 1]

            forcUN[:, :, i] = forcUN_type[:, :, 0]
            forcUN[:, :, i + 3] = forcUN_type[:, :, 1]
            forcInvUN[:, :, i] = forcInvUN_type[:, :, 0]
            forcInvUN[:, :, i + 3] = forcInvUN_type[:, :, 1]
            forcTestUN[:, :, i] = forcTestUN_type[:, :, 0]
            forcTestUN[:, :, i + 3] = forcTestUN_type[:, :, 1]


            # if forType[i] == 'daymet':
            #     daymet_temp = forcUN_type[:, :, 1]
            #     daymetInV_temp = forcInvUN_type[:, :, 1]
            #     daymetTest_temp = forcTestUN_type[:, :, 1]
            # if forType[i] == 'nldas' or forType[i] == 'nldas_extended':
            #     nldas_temp = forcUN_type[:, :, 1]
            #     nldasInV_temp = forcInvUN_type[:, :, 1]
            #     nldasTest_temp = forcTestUN_type[:, :, 1]

        attrsUN = dfInv.getDataConst(varLst=attrnewLst, doNorm=False, rmNan=False)
        obsTestUN = dfTest.getDataObs(doNorm=False, rmNan=False, basinnorm=False)
        attrsTestUN = dfTest.getDataConst(varLst=attrnewLst, doNorm=False, rmNan=False)
        attrs_wghtsUN = dfInv.getDataConst(varLst=attrWghts, doNorm=False, rmNan=False)
        attrsTest_wghtsUN = dfTest.getDataConst(varLst=attrWghts, doNorm=False, rmNan=False)

        # if 'daymet' in forType:
        #     forcUN[:, :, -1] = daymet_temp
        #     forcInvUN[:, :, -1] = daymetInV_temp
        #     forcTestUN[:, :, -1] = daymetTest_temp
        # elif 'nldas' in forType or 'nldas_extended' in forType:
        #     forcUN[:, :, -1] = nldas_temp
        #     forcInvUN[:, :, -1] = nldasInV_temp
        #     forcTestUN[:, :, -1] = nldasTest_temp


    else:
        if forType == 'daymet':
            varF = ['prcp', 'tmean']
            varFInv = ['prcp', 'tmean']
        else:
            varF = ['prcp', 'tmax']  # For CAMELS maurer and nldas forcings, tmax is actually tmean
            varFInv = ['prcp', 'tmax']

        dfTrain = camels.DataframeCamels(tRange=TtrainLoad, subset=TrainLS, forType=forType)
        forcUN = dfTrain.getDataTs(varLst=varF, doNorm=False, rmNan=False)

        dfInv = camels.DataframeCamels(tRange=TinvLoad, subset=TrainLS, forType=forType)
        forcInvUN = dfInv.getDataTs(varLst=varFInv, doNorm=False, rmNan=False)
        attrsUN = dfInv.getDataConst(varLst=attrnewLst, doNorm=False, rmNan=False)

        dfTest = camels.DataframeCamels(tRange=TtestLoad, subset=TestLS, forType=forType)
        forcTestUN = dfTest.getDataTs(varLst=varF, doNorm=False, rmNan=False)
        obsTestUN = dfTest.getDataObs(doNorm=False, rmNan=False, basinnorm=False)
        attrsTestUN = dfTest.getDataConst(varLst=attrnewLst, doNorm=False, rmNan=False)

    # for HBV training inputs
    # dfTrain = camels.DataframeCamels(tRange=TtrainLoad, subset=TrainLS, forType=forType)
    # dfTrain = camels.DataframeCamels(tRange=TtrainLoad, subset=TrainLS, forType=forType)
    # forcUN = dfTrain.getDataTs(varLst=varF, doNorm=False, rmNan=False)
    # # obsUN = dfTrain.getDataObs(doNorm=False, rmNan=False, basinnorm=False)  # useless for testing
    #
    # # for dPL inversion training data
    # dfInv = camels.DataframeCamels(tRange=TinvLoad, subset=TrainLS, forType=forType)
    # forcInvUN = dfInv.getDataTs(varLst=varFInv, doNorm=False, rmNan=False)
    # attrsUN = dfInv.getDataConst(varLst=attrnewLst, doNorm=False, rmNan=False)

    # for HBV testing input
    # dfTest = camels.DataframeCamels(tRange=TtestLoad, subset=TestLS, forType=forType)
    # forcTestUN = dfTest.getDataTs(varLst=varF, doNorm=False, rmNan=False)
    # obsTestUN = dfTest.getDataObs(doNorm=False, rmNan=False, basinnorm=False)
    # attrsTestUN = dfTest.getDataConst(varLst=attrnewLst, doNorm=False, rmNan=False)

    # Transform obs from ft3/s to mm/day
    # areas = gageinfo['area'][TrainInd] # unit km2
    # temparea = np.tile(areas[:, None, None], (1, obsUN.shape[1],1))
    # obsUN = (obsUN * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10**3  # useless for testing

    areas = gageinfo['area'][TestInd] # unit km2
    temparea = np.tile(areas[:, None, None], (1, obsTestUN.shape[1],1))
    obsTestUN = (obsTestUN * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10**3 # transform to mm/day

    # load potential ET calculated by hargreaves method
    varLstNL = ['PEVAP']
    usgsIdLst = gageid

    # for multiple PETs
    PETUN = np.empty([len(usgsIdLst), num_days_train, len(forType)])
    PETInvUN = np.empty([len(usgsIdLst), num_days_train, len(forType)])
    PETTestUN = np.empty([len(usgsIdLst), num_days_test, len(forType)])
    if type(forType) == list:
        for i in range(len(forType)):
            if forType[i] == 'nldas_extended' or forType[i] == 'nldas':
                PETDir = rootDatabase + '/pet_harg/' + 'nldas' + '/'
                tPETRange = [19800101, 20150101]
                tPETLst = utils.time.tRange2Array(tPETRange)
            if forType[i] == 'maurer_extended' or forType[i] == 'maurer':
                PETDir = rootDatabase + '/pet_harg/' + 'maurer' + '/'
                tPETRange = [19800101, 20090101]
                tPETLst = utils.time.tRange2Array(tPETRange)
            if forType[i] == 'daymet':
                PETDir = rootDatabase + '/pet_harg/' + 'daymet' + '/'
                tPETRange = [19800101, 20150101]
                tPETLst = utils.time.tRange2Array(tPETRange)
            ntime = len(tPETLst)
            PETfull = np.empty([len(usgsIdLst), ntime, len(varLstNL)])
            for k in range(len(usgsIdLst)):
                dataTemp = camels.readcsvGage(PETDir, usgsIdLst[k], varLstNL, ntime)
                PETfull[k, :, :] = dataTemp
            TtrainLst = utils.time.tRange2Array(TtrainLoad)
            TinvLst = utils.time.tRange2Array(TinvLoad)
            TtestLoadLst = utils.time.tRange2Array(TtestLoad)
            C, ind1, ind2 = np.intersect1d(TtrainLst, tPETLst, return_indices=True)
            PETUN_type = PETfull[:, ind2, :]
            PETUN_type = PETUN_type[TrainInd, :, :]  # select basins
            PETUN[:, :, i] = PETUN_type[:, :, 0]
            C, ind1, ind2inv = np.intersect1d(TinvLst, tPETLst, return_indices=True)
            PETInvUN_type = PETfull[:, ind2inv, :]
            PETInvUN_type = PETInvUN_type[TrainInd, :, :]
            PETInvUN[:, :, i] = PETInvUN_type[:, :, 0]
            C, ind1, ind2test = np.intersect1d(TtestLoadLst, tPETLst, return_indices=True)
            PETTestUN_type = PETfull[:, ind2test, :]
            PETTestUN_type = PETTestUN_type[TestInd, :, :]
            PETTestUN[:,:,i] = PETTestUN_type[:, :, 0]


    # if forType == 'maurer' or forType == 'maurer_extended' or forType==['maurer_extended'] or forType==['maurer']:
    #     tPETRange = [19800101, 20090101]
    # else:
    #     tPETRange = [19800101, 20150101]
    # tPETLst = utils.time.tRange2Array(tPETRange)
    #
    # # if forType == ['daymet', 'maurer_extended', 'nldas_extended']:
    # #     PETDir = rootDatabase + '/pet_harg/' + 'daymet' + '/'
    # # else:
    # #     PETDir = rootDatabase + '/pet_harg/' + forType + '/'
    # if type(forType) == list:
    #     if forType[0] == 'nldas_extended':
    #         PETDir = rootDatabase + '/pet_harg/' + 'nldas' + '/'
    #     elif forType[0] == 'maurer_extended':
    #         PETDir = rootDatabase + '/pet_harg/' + 'maurer' + '/'
    #     else:
    #         PETDir = rootDatabase + '/pet_harg/' + forType[0] + '/'
    # else:
    #     PETDir = rootDatabase + '/pet_harg/' + forType + '/'
    #
    # ntime = len(tPETLst)
    # PETfull = np.empty([len(usgsIdLst), ntime, len(varLstNL)])
    # for k in range(len(usgsIdLst)):
    #     dataTemp = camels.readcsvGage(PETDir, usgsIdLst[k], varLstNL, ntime)
    #     PETfull[k, :, :] = dataTemp
    #
    # TtrainLst = utils.time.tRange2Array(TtrainLoad)
    # TinvLst = utils.time.tRange2Array(TinvLoad)
    # TtestLoadLst = utils.time.tRange2Array(TtestLoad)
    # C, ind1, ind2 = np.intersect1d(TtrainLst, tPETLst, return_indices=True)
    # PETUN = PETfull[:, ind2, :]
    # PETUN = PETUN[TrainInd, :, :] # select basins
    # C, ind1, ind2inv = np.intersect1d(TinvLst, tPETLst, return_indices=True)
    # PETInvUN = PETfull[:, ind2inv, :]
    # PETInvUN = PETInvUN[TrainInd, :, :]
    # C, ind1, ind2test = np.intersect1d(TtestLoadLst, tPETLst, return_indices=True)
    # PETTestUN = PETfull[:, ind2test, :]
    # PETTestUN = PETTestUN[TestInd, :, :]

    # process data, do normalization and remove nan
    series_inv = np.concatenate([forcInvUN, PETInvUN], axis=2)
    # series_inv_hbv = series_inv[:, :, (0, -2, -1)]
    # for multiforcing
    series_inv_hbv = series_inv[:, :, (0, 3, 6)]

    seriesvarLst = varFInv + ['pet']
    # load the saved statistics
    statFile_wghts = os.path.join(testout, 'statDict_wghts.json')
    with open(statFile_wghts, 'r') as fp:
        statDict_wghts = json.load(fp)
    statFile_hbv = os.path.join(testout, 'statDict_hbv.json')
    with open(statFile_hbv, 'r') as fp:
        statDict_hbv = json.load(fp)

    # normalize
    attr_hbv_norm = camels.transNormbyDic(attrsUN, attrnewLst, statDict_hbv, toNorm=True)
    attrWghts_norm = camels.transNormbyDic(attrs_wghtsUN, attrWghts, statDict_wghts, toNorm=True)
    attr_hbv_norm[np.isnan(attr_hbv_norm)] = 0.0
    attrWghts_norm[np.isnan(attrWghts_norm)] = 0.0
    series_norm_hbv = camels.transNormbyDic(series_inv_hbv, seriesvarLst, statDict_hbv, toNorm=True)
    series_Wghts_norm = camels.transNormbyDic(series_inv,['prcp_daymet', 'prcp_maurer', 'prcp_nldas','tmax_daymet', 'tmax_maurer', 'tmax_nldas',
                                                                                         'pet_daymet', 'pet_maurer', 'pet_nldas'], statDict_wghts, toNorm=True)
    series_norm_hbv[np.isnan(series_norm_hbv)] = 0.0
    series_Wghts_norm[np.isnan(series_Wghts_norm)] = 0.0

    attrtest_hbv_norm = camels.transNormbyDic(attrsTestUN, attrnewLst, statDict_hbv, toNorm=True)
    attrtest_Wghts_norm = camels.transNormbyDic(attrsTest_wghtsUN, attrWghts, statDict_wghts, toNorm=True)
    attrtest_hbv_norm[np.isnan(attrtest_hbv_norm)] = 0.0
    attrtest_Wghts_norm[np.isnan(attrtest_Wghts_norm)] = 0.0
    seriestest_inv = np.concatenate([forcTestUN, PETTestUN], axis=2)


    # seriestest_inv_hbv = seriestest_inv[:, :, (0, -2, -1)]
    #for multiforcing
    seriestest_inv_hbv = seriestest_inv[:, :, (0, 3, 6)]


    seriestest_Wghts_norm = camels.transNormbyDic(seriestest_inv, ['prcp_daymet', 'prcp_maurer', 'prcp_nldas',
                                                                                         'tmax_daymet', 'tmax_maurer', 'tmax_nldas',
                                                                                         'pet_daymet', 'pet_maurer', 'pet_nldas'], statDict_wghts, toNorm=True)
    seriestest_norm_hbv = camels.transNormbyDic(seriestest_inv_hbv, seriesvarLst, statDict_hbv, toNorm=True)
    seriestest_Wghts_norm[np.isnan(seriestest_Wghts_norm)] = 0.0
    seriestest_norm_hbv[np.isnan(seriestest_norm_hbv)] = 0.0

    # prepare the inputs
    zTrain_hbv = series_norm_hbv  # used as the inputs for dPL inversion gA along with attributes
    zTrain_wghts = series_Wghts_norm  # used as the inputs for dPL inversion gA along with attributes
    xTrain_wghts = np.concatenate([forcUN, PETUN], axis=2)  # used as HBV forcing
    xTrain_hbv = xTrain_wghts[:, :, (0, -2, -1)]  # used as HBV forcing
    xTrain_wghts[np.isnan(xTrain_wghts)] = 0.0
    xTrain_hbv[np.isnan(xTrain_hbv)] = 0.0

    if buffOpt == 1: # repeat the first year for buff
        zTrainIn_wghts = np.concatenate([zTrain_wghts[:,0:BUFFTIME,:], zTrain_wghts], axis=1)
        zTrainIn_hbv = np.concatenate([zTrain_hbv[:,0:BUFFTIME,:], zTrain_hbv], axis=1)
        xTrainIn_wghts = np.concatenate([xTrain_wghts[:,0:BUFFTIME,:], xTrain_wghts], axis=1) # Bufftime for the first year
        xTrainIn_hbv = np.concatenate([xTrain_hbv[:,0:BUFFTIME,:], xTrain_hbv], axis=1) # Bufftime for the first year
        # yTrainIn = np.concatenate([obsUN[:,0:BUFFTIME,:], obsUN], axis=1)
    else: # no repeat, original data
        zTrainIn_wghts = zTrain_wghts
        zTrainIn_hbv = zTrain_hbv
        xTrainIn_wghts = xTrain_wghts
        xTrainIn_hbv = xTrain_hbv
        # yTrainIn = obsUN

    forcTuple_hbv = (xTrainIn_hbv, zTrainIn_hbv)
    forcTuple_wghts = (xTrainIn_wghts, zTrainIn_wghts)
    attrs_hbv = attr_hbv_norm
    attrsWghts = attrWghts_norm

    ## Prepare the testing data and forward the trained model for testing
    runBUFF = 0

    # TestBuff = 365 # Use 365 days forcing to warm up the model for testing
    TestBuff = xTrain_hbv.shape[1]  # Use the whole training period to warm up the model for testing
    # TestBuff_wghts = xTrain_wghts.shape[1]  # Use the whole training period to warm up the model for testing
    # TestBuff = len(TtestLoadLst) - len(TtestLst)  # use the redundantly loaded data to warm up

    testmodel.inittime = runBUFF
    testmodel.dydrop = 0.0

    # prepare file name to save the testing predictions
    filePathLst = master.master.namePred(
          testout, Ttest, 'All_Buff'+str(TestBuff), epoch=testepoch, targLst=['Qr', 'Q0', 'Q1', 'Q2', 'ET'])

    # prepare the inputs for TESTING
    if PUOpt == 0: # for ALL basins, temporal generalization test
        testmodel.staind = TestBuff-1
        zTest_wghts = np.concatenate([series_Wghts_norm[:, -TestBuff:, :], seriestest_Wghts_norm], axis=1)
        zTest_hbv = np.concatenate([series_norm_hbv[:, -TestBuff:, :], seriestest_norm_hbv], axis=1)
        xTest_wghts = np.concatenate([forcTestUN, PETTestUN], axis=2)  # HBV forcing
        # xTest_hbv = xTest_wghts[:,:,(0,-2,-1)]
        xTest_hbv = xTest_wghts[:,:,(0,3,6)]
        # forcings to warm up the model. Here use the forcing of training period to warm up
        xTestBuff_wghts = xTrain_wghts[:, -TestBuff:, :]
        xTestBuff_hbv = xTrain_hbv[:, -TestBuff:, :]
        xTest_wghts = np.concatenate([xTestBuff_wghts, xTest_wghts], axis=1)
        xTest_hbv = np.concatenate([xTestBuff_hbv, xTest_hbv], axis=1)
        obs = obsTestUN[:, 0:, :]  # starts with 0 when not loading more data before testing period

    else:  # for PUB/PUR cases, different testing basins. Load more forcings to warm up.
        # testmodel.staind = -1
        testmodel.staind = TestBuff-1
        zTest_wghts = seriestest_Wghts_norm
        zTest_hbv = seriestest_norm_hbv
        xTest_wghts = np.concatenate([forcTestUN, PETTestUN], axis=2)  # HBV forcing
        xTest_hbv =  xTest_wghts[:,:,(0,-2,-1)]
        obs = obsTestUN[:, TestBuff:, :]  # exclude loaded obs in warming up period (first TestBuff days) for evaluation

    # Final inputs to the test model
    xTest_wghts[np.isnan(xTest_wghts)] = 0.0
    xTest_hbv[np.isnan(xTest_hbv)] = 0.0
    attrtest_hbv = attrtest_hbv_norm
    attrtest_wghts = attrtest_Wghts_norm
    cTemp_hbv = np.repeat(
        np.reshape(attrtest_hbv, [attrtest_hbv.shape[0], 1, attrtest_hbv.shape[-1]]), zTest_hbv.shape[1], axis=1)
    cTemp_wghts = np.repeat(
        np.reshape(attrtest_wghts, [attrtest_wghts.shape[0], 1, attrtest_wghts.shape[-1]]), zTest_wghts.shape[1], axis=1)
    zTest_hbv = np.concatenate([zTest_hbv, cTemp_hbv], 2) # Add attributes to historical forcings as the inversion part
    zTest_wghts = np.concatenate([zTest_wghts, cTemp_wghts], 2) # Add attributes to historical forcings as the inversion part
    testTuple_wghts = (xTest_wghts, zTest_wghts) # xTest: input forcings to HBV; zTest: inputs to gA LSTM to learn parameters
    testTuple_hbv = (xTest_hbv, zTest_hbv) # xTest: input forcings to HBV; zTest: inputs to gA LSTM to learn parameters
    hbv_path = "/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL" \
              "/TDTestforc/TD1_13/daymet/BuffOpt0/RMSE_para0.25/111111/Fold1" \
              "/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/model_Ep50.pt"
    # hbv_path = "/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/" \
    #            "ALL/TDTestforc/TD1_13/allprcp_5withloss200smooth0/BuffOpt0/RMSE_para0.25/" \
    #            "111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/model_hbv_Ep50.pt"

    loaded_hbv = torch.load(hbv_path)
    loaded_hbv.inittime = runBUFF
    loaded_hbv.dydrop = 0.0
    loaded_hbv.staind = TestBuff - 1
    # forward the model and save results
    train.test2Model(
        testmodel, loaded_hbv, testTuple_wghts, testTuple_hbv, c=None, batchSize=testbatch,
        filePathLst=filePathLst, prcp_loss_factor=prcp_loss_factor, smooth_loss_factor=smooth_loss_factor, prcp_datatypes = len(forType))

    # read out the saved forward predictions
    dataPred = np.ndarray([obs.shape[0], obs.shape[1]+TestBuff-runBUFF, len(filePathLst)])
    for k in range(len(filePathLst)):
        filePath = filePathLst[k]
        dataPred[:, :, k] = pd.read_csv(
            filePath, dtype=np.float, header=None).values
    # save the predictions to the big matrix
    predtestALL[nstart:nstart+nbasin, :, :] = dataPred[:, TestBuff-runBUFF:, :]
    obstestALL[nstart:nstart+nbasin, :, :] = obs
    nstart = nstart + nbasin
    logtestIDLst = logtestIDLst + TestLS

## post processing
# calculate evaluation metrics
evaDict = [stat.statError(predtestALL[:,:,0], obstestALL.squeeze())]  # Q0: the streamflow

# save evaluation results
# Note: for PUB/PUR testing, this path location saves results covering all folds. If you are only testing one fold,
# consider add fold specification in "seStr" to prevent results overwritten from different folds.
seStr = 'Train'+str(Ttrain[0])+'_'+str(Ttrain[1])+'Test'+str(Ttest[0])+'_'+str(Ttest[1])+'Buff'+str(TestBuff)+'Staind'+str(testmodel.staind)
outpath = os.path.join(rootOut, testsave_path, seStr)
if not os.path.isdir(outpath):
    os.makedirs(outpath)

EvaFile = os.path.join(outpath, 'Eva'+str(testepoch)+'.npy')
np.save(EvaFile, evaDict)

obsFile = os.path.join(outpath, 'obs.npy')
np.save(obsFile, obstestALL)

predFile = os.path.join(outpath, 'pred'+str(testepoch)+'.npy')
np.save(predFile, predtestALL)

# calculate metrics for the widely used CAMELS subset with 531 basins
# we report on this 531 subset in Feng et al., 2022 HESSD
subsetPath = 'Sub531ID.txt'
with open(subsetPath, 'r') as fp:
    sub531IDLst = json.load(fp)  # Subset 531 ID List
# get the evaluation metrics on 531 subset
[C, ind1, SubInd] = np.intersect1d(sub531IDLst, logtestIDLst, return_indices=True)
evaframe = pd.DataFrame(evaDict[0])
evaframeSub = evaframe.loc[SubInd, list(evaframe.keys())]
evaS531Dict = [{col:evaframeSub[col].values for col in evaframeSub}] # 531 subset evaDict

# print NSE median value of testing basins
print('Testing finished! Evaluation results saved in\n', outpath)
print('For basins of whole CAMELS, NSE median:', np.nanmedian(evaDict[0]['NSE']))
print('For basins of 531 subset, NSE median:', np.nanmedian(evaS531Dict[0]['NSE']))
print('For basins of 531 subset, KGE median:', np.nanmedian(evaS531Dict[0]['KGE']))


## Show boxplots of the results
evaDictLst = evaDict + evaS531Dict
plt.rcParams['font.size'] = 14
plt.rcParams["legend.columnspacing"] = 0.1
plt.rcParams["legend.handletextpad"] = 0.2
keyLst = ['NSE', 'KGE', 'lowRMSE', 'highRMSE', 'midRMSE','AFHV', 'AFLV', 'AFMV']
dataBox = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(evaDictLst)):
        data = evaDictLst[k][statStr]
        data = data[~np.isnan(data)]
        temp.append(data)
    dataBox.append(temp)

print("NSE,KGE,lowRMSE, highRMSE, mean lowRMSE, mean highRMSE, mean midRMSE, abs FHV, abs FLV, abs FMV of all basins: ", np.nanmedian(dataBox[0][0]),
          np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]),
          np.nanmean(dataBox[2][0]), np.nanmean(dataBox[3][0]), np.nanmean(dataBox[4][0]),
      np.nanmedian(dataBox[5][0]), np.nanmedian(dataBox[6][0]), np.nanmedian(dataBox[7][0]))

labelname = ['dPL+HBV_Multi', 'dPL+HBV_Multi Sub531']
xlabel = keyLst
fig = plot.plotBoxFig(dataBox, xlabel, labelname, sharey=False, figsize=(6, 5))
fig.patch.set_facecolor('white')
fig.show()
plt.savefig(os.path.join(outpath, 'Metric_BoxPlot.png'), format='png')