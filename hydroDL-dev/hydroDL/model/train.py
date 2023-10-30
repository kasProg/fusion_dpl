import numpy as np
import torch
import time
import os
import hydroDL
from hydroDL.model import rnn, rnn_new, cnn, crit
import pandas as pd
import torch.autograd as autograd


def trainModel(model,
               x,
               y,
               c,
               lossFun,
               *,
               nEpoch=500,
               miniBatch=[100, 30],
               saveEpoch=100,
               saveFolder=None,
               mode='seq2seq',
               bufftime=0,
               prcp_loss_factor = 15,
               smooth_loss_factor = 1,
               multiforcing=False):
    batchSize, rho = miniBatch
    # x- input; z - additional input; y - target; c - constant input
    if type(x) is tuple or type(x) is list:
        x, z = x
    ngrid, nt, nx = x.shape
    if c is not None:
        nx = nx + c.shape[-1]
    if batchSize >= ngrid:
        # batchsize larger than total grids
        batchSize = ngrid

    nIterEp = int(
        np.ceil(np.log(0.01) / np.log(1 - batchSize * rho / ngrid / (nt-bufftime))))
    if hasattr(model, 'ctRm'):
        if model.ctRm is True:
            nIterEp = int(
                np.ceil(
                    np.log(0.01) / np.log(1 - batchSize *
                                          (rho - model.ct) / ngrid / (nt-bufftime))))

    if torch.cuda.is_available():
        lossFun = lossFun.cuda()
        model = model.cuda()

    optim = torch.optim.Adadelta(model.parameters())
    model.zero_grad()
    if saveFolder is not None:
        runFile = os.path.join(saveFolder, 'run.csv')
        rf = open(runFile, 'w+')
    for iEpoch in range(1, nEpoch + 1):
        lossEp = 0
        if multiforcing is True:
            loss_prcp_Ep = 0
            loss_pet_Ep = 0
            loss_temp_Ep = 0
            loss_sf_Ep = 0
            loss_smooth_Ep = 0
        t0 = time.time()
        for iIter in range(0, nIterEp):
            # training iterations
            if type(model) in [rnn.CudnnLstmModel, rnn.AnnModel, rnn.CpuLstmModel]:
                iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho], bufftime=bufftime)
                xTrain = selectSubset(x, iGrid, iT, rho, c=c, bufftime=bufftime)
                # xTrain = rho/time * Batchsize * Ninput_var
                yTrain = selectSubset(y, iGrid, iT, rho)
                # yTrain = rho/time * Batchsize * Ntraget_var
                yP = model(xTrain)[bufftime:, :, :]
            if type(model) in [rnn.CudnnLstmModel_R2P]:
                # yP = rho/time * Batchsize * Ntraget_var
                iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho])
                xTrain = selectSubset(x, iGrid, iT, rho, c=c, tupleOut=True)
                yTrain = selectSubset(y, iGrid, iT, rho)
                yP, Param_R2P = model(xTrain)
            if type(model) in [rnn.LstmCloseModel, rnn.AnnCloseModel, rnn.CNN1dLSTMmodel, rnn.CNN1dLSTMInmodel,
                               rnn.CNN1dLCmodel, rnn.CNN1dLCInmodel, rnn.CudnnInvLstmModel,
                               rnn.MultiInv_HBVModel, rnn.MultiInv_HBVTDModel, rnn.MultiInv_HBVTDModel_1]:
                iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho], bufftime=bufftime)
                if type(model) in [rnn.MultiInv_HBVModel, rnn.MultiInv_HBVTDModel, rnn.MultiInv_HBVTDModel_1]:
                    xTrain = selectSubset(x, iGrid, iT, rho, bufftime=bufftime)
                else:
                    xTrain = selectSubset(x, iGrid, iT, rho, c=c)
                yTrain = selectSubset(y, iGrid, iT, rho)
                if type(model) in [rnn.CNN1dLCmodel, rnn.CNN1dLCInmodel]:
                    zTrain = selectSubset(z, iGrid, iT=None, rho=None, LCopt=True)
                elif type(model) in [rnn.CudnnInvLstmModel]: # For smap inv LSTM, HBV Inv
                    # zTrain = selectSubset(z, iGrid, iT=None, rho=None, LCopt=False)
                    zTrain = selectSubset(z, iGrid, iT=None, rho=None, LCopt=False, c=c) # Add the attributes to inv
                elif type(model) in [rnn.MultiInv_HBVModel]:
                    zTrain = selectSubset(z, iGrid, iT, rho, c=c)
                elif type(model) in [rnn.MultiInv_HBVTDModel, rnn.MultiInv_HBVTDModel_1]:
                    zTrain = selectSubset(z, iGrid, iT, rho, c=c, bufftime=bufftime)
                else:
                    zTrain = selectSubset(z, iGrid, iT, rho)

                if multiforcing==True:
                    yP, prcp_loss, pet_loss, temp_loss, prcp_pet_wghtsm, smooth_loss = model(xTrain, zTrain, prcp_loss_factor, smooth_loss_factor, multiforcing)
                else:
                    yP = model(xTrain, zTrain, prcp_loss_factor, smooth_loss_factor, multiforcing)
                # yP = model(xTrain, zTrain)
            if type(model) in [cnn.LstmCnn1d]:
                iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho])
                xTrain = selectSubset(x, iGrid, iT, rho, c=c)
                # xTrain = rho/time * Batchsize * Ninput_var
                xTrain = xTrain.permute(1, 2, 0)
                yTrain = selectSubset(y, iGrid, iT, rho)
                # yTrain = rho/time * Batchsize * Ntraget_var
                yTrain = yTrain.permute(1, 2, 0)[:, :, int(rho/2):]
                yP = model(xTrain)
            # if type(model) in [hydroDL.model.rnn.LstmCnnCond]:
            #     iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho])
            #     xTrain = selectSubset(x, iGrid, iT, rho)
            #     yTrain = selectSubset(y, iGrid, iT, rho)
            #     zTrain = selectSubset(z, iGrid, None, None)
            #     yP = model(xTrain, zTrain)
            # if type(model) in [hydroDL.model.rnn.LstmCnnForcast]:
            #     iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho])
            #     xTrain = selectSubset(x, iGrid, iT, rho)
            #     yTrain = selectSubset(y, iGrid, iT + model.ct, rho - model.ct)
            #     zTrain = selectSubset(z, iGrid, iT, rho)
            #     yP = model(xTrain, zTrain)
            else:
                Exception('unknown model')
            # # consider the buff time for initialization
            # if bufftime > 0:
            #     yP = yP[bufftime:,:,:]
            ## temporary test for NSE loss
            if type(lossFun) in [crit.NSELossBatch, crit.NSESqrtLossBatch]:
                if multiforcing==True:
                    loss_sf = lossFun(yP, yTrain, iGrid)
                    loss =  loss_sf + prcp_loss + smooth_loss + pet_loss + temp_loss
                else:
                    loss = lossFun(yP, yTrain, iGrid)
            else:
                if multiforcing==True:
                    loss_sf = lossFun(yP, yTrain)
                    loss = loss_sf + prcp_loss + smooth_loss + pet_loss + temp_loss
                else:
                    loss = lossFun(yP, yTrain)
            loss.backward()
            optim.step()
            model.zero_grad()
            lossEp = lossEp + loss.item()
            if multiforcing==True:
                loss_prcp_Ep = loss_prcp_Ep + prcp_loss.item()
                # loss_pet_Ep = loss_pet_Ep + pet_loss.item()
                # loss_temp_Ep = loss_temp_Ep + temp_loss.item()
                loss_temp_Ep = 0
                loss_pet_Ep = 0
                loss_sf_Ep = loss_sf_Ep + loss_sf.item()
                loss_smooth_Ep = loss_smooth_Ep + smooth_loss.item()
            # print(iIter, '  ', loss.item())
            # if iIter == 223:
            #     print('This is the error point')
            #     print('Debug start')

            if iIter % 100 == 0:
                print('Iter {} of {}: Loss {:.3f}'.format(iIter, nIterEp, loss.item()))
        # print loss
        lossEp = lossEp / nIterEp
        if multiforcing==True:
            loss_sf_Ep = loss_sf_Ep / nIterEp
            loss_prcp_Ep = loss_prcp_Ep / nIterEp
            loss_pet_Ep = loss_pet_Ep / nIterEp
            loss_temp_Ep = loss_temp_Ep / nIterEp
            logStr = 'Epoch {} Loss {:.3f}, Streamflow Loss {:.3f}, Precipitation Loss {:.3f}, PET Loss {:.3f}, Temperature Loss {:.3f}, Weights Smoothing Loss {:.3f}, time {:.2f}'.format(
                iEpoch, lossEp, loss_sf_Ep, loss_prcp_Ep, loss_pet_Ep, loss_temp_Ep, loss_smooth_Ep,
                time.time() - t0)
        else:
            logStr = 'Epoch {} Loss {:.3f}, time {:.2f}'.format(
                iEpoch, lossEp, time.time() - t0)
        print(logStr)
        # save model and loss
        if saveFolder is not None:
            rf.write(logStr + '\n')
            if iEpoch % saveEpoch == 0:
                # save model
                modelFile = os.path.join(saveFolder,
                                         'model_Ep' + str(iEpoch) + '.pt')
                torch.save(model, modelFile)
    if saveFolder is not None:
        rf.close()
    return model


def train2Model(model,
                loaded_hbv,
               x,
                x2,
               y,
               c,
                c2,
               lossFun,
               *,
               nEpoch=500,
               miniBatch=[100, 30],
               saveEpoch=100,
               saveFolder=None,
               mode='seq2seq',
               bufftime=0,
               prcp_loss_factor = 15,
               smooth_loss_factor = 0,
               multiforcing=False):
    batchSize, rho = miniBatch
    # x- input; z - additional input; y - target; c - constant input
    if type(x) is tuple or type(x) is list:
        x, z = x
        x2, z2 = x2
    ngrid, nt, nx = x.shape
    ngrid, nt, nx2 = x2.shape
    if c is not None:
        nx = nx + c.shape[-1]
        nx2 = nx2 + c2.shape[-1]
    if batchSize >= ngrid:
        # batchsize larger than total grids
        batchSize = ngrid

    nIterEp = int(
        np.ceil(np.log(0.01) / np.log(1 - batchSize * rho / ngrid / (nt-bufftime))))
    if hasattr(model, 'ctRm'):
        if model.ctRm is True:
            nIterEp = int(
                np.ceil(
                    np.log(0.01) / np.log(1 - batchSize *
                                          (rho - model.ct) / ngrid / (nt-bufftime))))

    if torch.cuda.is_available():
        lossFun = lossFun.cuda()
        model = model.cuda()
        # loaded_hbv = loaded_hbv.cuda()

    # optim = torch.optim.Adadelta(list(model.parameters()) + list(loaded_hbv.parameters()))
    optim = torch.optim.Adadelta(list(model.parameters()))
    # loaded_hbv.zero_grad()
    model.zero_grad()
    if saveFolder is not None:
        runFile = os.path.join(saveFolder, 'run.csv')
        rf = open(runFile, 'w+')
    for iEpoch in range(1, nEpoch + 1):
        lossEp = 0
        if multiforcing is True:
            loss_prcp_Ep = 0
            # loss_pet_Ep = 0
            # loss_temp_Ep = 0
            loss_sf_Ep = 0
            # loss_smooth_Ep = 0
        t0 = time.time()
        for iIter in range(0, nIterEp):
            # training iterations
            if type(model) in [rnn.LstmCloseModel, rnn.AnnCloseModel, rnn.CNN1dLSTMmodel, rnn.CNN1dLSTMInmodel,
                               rnn.CNN1dLCmodel, rnn.CNN1dLCInmodel, rnn.CudnnInvLstmModel,
                               rnn.MultiInv_HBVModel, rnn.MultiInv_HBVTDModel, rnn.prcp_weights]:
                iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho], bufftime=bufftime)
                if type(model) in [rnn.MultiInv_HBVModel, rnn.MultiInv_HBVTDModel, rnn.prcp_weights]:
                    xTrain = selectSubset(x, iGrid, iT, rho, bufftime=bufftime)
                    xTrain_hbv = selectSubset(x2, iGrid, iT, rho, bufftime=bufftime)
                else:
                    xTrain = selectSubset(x, iGrid, iT, rho, c=c)
                    xTrain_hbv = selectSubset(x2, iGrid, iT, rho, c=c2)
                yTrain = selectSubset(y, iGrid, iT, rho)
                if type(model) in [rnn.CNN1dLCmodel, rnn.CNN1dLCInmodel]:
                    zTrain = selectSubset(z, iGrid, iT=None, rho=None, LCopt=True)
                    zTrain_hbv = selectSubset(z2, iGrid, iT=None, rho=None, LCopt=True)
                elif type(model) in [rnn.CudnnInvLstmModel]: # For smap inv LSTM, HBV Inv
                    # zTrain = selectSubset(z, iGrid, iT=None, rho=None, LCopt=False)
                    zTrain = selectSubset(z, iGrid, iT=None, rho=None, LCopt=False, c=c) # Add the attributes to inv
                    zTrain_hbv = selectSubset(z2, iGrid, iT=None, rho=None, LCopt=False, c=c2) # Add the attributes to inv
                elif type(model) in [rnn.MultiInv_HBVModel]:
                    zTrain = selectSubset(z, iGrid, iT, rho, c=c)
                    zTrain_hbv = selectSubset(z2, iGrid, iT, rho, c=c2)
                elif type(model) in [rnn.MultiInv_HBVTDModel, rnn.prcp_weights]:
                    zTrain = selectSubset(z, iGrid, iT, rho, c=c, bufftime=bufftime)
                    zTrain_hbv = selectSubset(z2, iGrid, iT, rho, c=c2, bufftime=bufftime)
                else:
                    zTrain = selectSubset(z, iGrid, iT, rho)
                    zTrain_hbv = selectSubset(z2, iGrid, iT, rho)
                # loaded_hbv.train(mode=False)
                xP, prcp_loss, prcp_wghts = model(xTrain, zTrain, prcp_loss_factor)
                yP = loaded_hbv(xP, zTrain_hbv, prcp_loss_factor =0, smooth_loss_factor=0, multiforcing=False)
                # yP = model(xTrain, zTrain, prcp_loss_factor, smooth_loss_factor, multiforcing)
                # yP = model(xTrain, zTrain)
            if type(model) in [cnn.LstmCnn1d]:
                iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho])
                xTrain = selectSubset(x, iGrid, iT, rho, c=c)
                # xTrain = rho/time * Batchsize * Ninput_var
                xTrain = xTrain.permute(1, 2, 0)
                yTrain = selectSubset(y, iGrid, iT, rho)
                # yTrain = rho/time * Batchsize * Ntraget_var
                yTrain = yTrain.permute(1, 2, 0)[:, :, int(rho/2):]
                yP = model(xTrain)
            # if type(model) in [hydroDL.model.rnn.LstmCnnCond]:
            #     iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho])
            #     xTrain = selectSubset(x, iGrid, iT, rho)
            #     yTrain = selectSubset(y, iGrid, iT, rho)
            #     zTrain = selectSubset(z, iGrid, None, None)
            #     yP = model(xTrain, zTrain)
            # if type(model) in [hydroDL.model.rnn.LstmCnnForcast]:
            #     iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho])
            #     xTrain = selectSubset(x, iGrid, iT, rho)
            #     yTrain = selectSubset(y, iGrid, iT + model.ct, rho - model.ct)
            #     zTrain = selectSubset(z, iGrid, iT, rho)
            #     yP = model(xTrain, zTrain)
            else:
                Exception('unknown model')
            # # consider the buff time for initialization
            # if bufftime > 0:
            #     yP = yP[bufftime:,:,:]
            ## temporary test for NSE loss
            if type(lossFun) in [crit.NSELossBatch, crit.NSESqrtLossBatch]:
                # if multiforcing==True:
                loss_sf = lossFun(yP, yTrain, iGrid)
                loss =  loss_sf + prcp_loss
                # else:
                #     loss = lossFun(yP, yTrain, iGrid)
            else:
                # if multiforcing==True:
                loss_sf = lossFun(yP, yTrain)
                loss = loss_sf + prcp_loss
                # else:
                #     loss = lossFun(yP, yTrain)
            loss.backward()
            optim.step()
            # model.zero_grad()
            optim.zero_grad()
            lossEp = lossEp + loss.item()
            try:
                loss_prcp_Ep = loss_prcp_Ep + prcp_loss.item()
            except:
                pass
            # loss_pet_Ep = loss_pet_Ep + pet_loss.item()
            # loss_temp_Ep = loss_temp_Ep + temp_loss.item()
            # loss_temp_Ep = 0
            # loss_pet_Ep = 0
            loss_sf_Ep = loss_sf_Ep + loss_sf.item()
                # loss_smooth_Ep = loss_smooth_Ep + smooth_loss.item()
            # print(iIter, '  ', loss.item())
            # if iIter == 223:
            #     print('This is the error point')
            #     print('Debug start')

            if iIter % 100 == 0:
                print('Iter {} of {}: Loss {:.3f}'.format(iIter, nIterEp, loss.item()))
        # print loss
        lossEp = lossEp / nIterEp
        # if multiforcing==True:
        loss_sf_Ep = loss_sf_Ep / nIterEp
        loss_prcp_Ep = loss_prcp_Ep / nIterEp
        # loss_pet_Ep = loss_pet_Ep / nIterEp
        # loss_temp_Ep = loss_temp_Ep / nIterEp
        logStr = 'Epoch {} Loss {:.3f}, Streamflow Loss {:.3f}, Precipitation Loss {:.3f}, time {:.2f}'.format(
            iEpoch, lossEp, loss_sf_Ep, loss_prcp_Ep,
            time.time() - t0)
        # else:
        #     logStr = 'Epoch {} Loss {:.3f}, time {:.2f}'.format(
        #         iEpoch, lossEp, time.time() - t0)
        print(logStr)
        # save model and loss
        if saveFolder is not None:
            rf.write(logStr + '\n')
            if iEpoch % saveEpoch == 0:
                # save model
                modelFile = os.path.join(saveFolder,
                                         'model_Ep' + str(iEpoch) + '.pt')
                torch.save(model, modelFile)
                # modelFile_hbv = os.path.join(saveFolder,
                #                          'model_hbv_Ep' + str(iEpoch) + '.pt')
                # torch.save(loaded_hbv, modelFile_hbv)
    if saveFolder is not None:
        rf.close()
    return model

def saveModel(outFolder, model, epoch, modelName='model'):
    modelFile = os.path.join(outFolder, modelName + '_Ep' + str(epoch) + '.pt')
    torch.save(model, modelFile)


def loadModel(outFolder, epoch, modelName='model'):
    modelFile = os.path.join(outFolder, modelName + '_Ep' + str(epoch) + '.pt')
    model = torch.load(modelFile)
    return model


def testModel(model, x, c, *, batchSize=None, filePathLst=None, doMC=False, outModel=None, savePath=None, prcp_loss_factor=15, multiforcing=False, smooth_loss_factor=1, prcp_datatypes=1):
# def testModel(model, x, c, *, batchSize=None, filePathLst=None, doMC=False, outModel=None, savePath=None, prcp_loss_factor=15, multiforcing=False, smooth_loss_factor=1):
    # outModel, savePath: only for R2P-hymod model, for other models always set None
    if type(x) is tuple or type(x) is list:
        x, z = x
        if type(model) is rnn.CudnnLstmModel:
            # For Cudnn, only one input. First concat inputs and obs
            x = np.concatenate([x, z], axis=2)
            z = None
    else:
        z = None
    ngrid, nt, nx = x.shape
    if c is not None:
        nc = c.shape[-1]
    if type(model) in [rnn.MultiInv_HBVModel, rnn.MultiInv_HBVTDModel, rnn.MultiInv_HBVTDModel_1]:
        ny=5 # streamflow
    else:
        ny = model.ny
    if batchSize is None:
        batchSize = ngrid
    if torch.cuda.is_available():
        model = model.cuda()

    model.train(mode=False)
    if hasattr(model, 'ctRm'):
        if model.ctRm is True:
            nt = nt - model.ct
    # yP = torch.zeros([nt, ngrid, ny])
    iS = np.arange(0, ngrid, batchSize)
    iE = np.append(iS[1:], ngrid)

    # deal with file name to save
    if filePathLst is None:
        filePathLst = ['out' + str(x) for x in range(ny)]
    fLst = list()
    for filePath in filePathLst:
        if os.path.exists(filePath):
            os.remove(filePath)
        f = open(filePath, 'a')
        fLst.append(f)

    # forward for each batch
    for i in range(0, len(iS)):
        print('batch {}'.format(i))
        xTemp = x[iS[i]:iE[i], :, :]
        if c is not None:
            cTemp = np.repeat(
                np.reshape(c[iS[i]:iE[i], :], [iE[i] - iS[i], 1, nc]), nt, axis=1)
            xTest = torch.from_numpy(
                np.swapaxes(np.concatenate([xTemp, cTemp], 2), 1, 0)).float()
        else:
            xTest = torch.from_numpy(
                np.swapaxes(xTemp, 1, 0)).float()
        if torch.cuda.is_available():
            xTest = xTest.cuda()
        if z is not None:
            if type(model) in [rnn.CNN1dLCmodel, rnn.CNN1dLCInmodel]:
                if len(z.shape) == 2:
                    # Used for local calibration kernel as FDC
                    # x = Ngrid * Ntime
                    zTest = torch.from_numpy(z[iS[i]:iE[i], :]).float()
                elif len(z.shape) == 3:
                    # used for LC-SMAP x=Ngrid*Ntime*Nvar
                    zTest = torch.from_numpy(np.swapaxes(z[iS[i]:iE[i], :, :], 1, 2)).float()
            else:
                zTemp = z[iS[i]:iE[i], :, :]
                # if type(model) in [rnn.CudnnInvLstmModel]: # Test SMAP Inv with attributes
                #     cInv = np.repeat(
                #         np.reshape(c[iS[i]:iE[i], :], [iE[i] - iS[i], 1, nc]), zTemp.shape[1], axis=1)
                #     zTemp = np.concatenate([zTemp, cInv], 2)
                zTest = torch.from_numpy(np.swapaxes(zTemp, 1, 0)).float()
            if torch.cuda.is_available():
                zTest = zTest.cuda()
        if type(model) in [rnn.CudnnLstmModel, rnn.AnnModel, rnn.CpuLstmModel]:
            # if z is not None:
            #     xTest = torch.cat((xTest, zTest), dim=2)
            yP = model(xTest)
            if doMC is not False:
                ySS = np.zeros(yP.shape)
                yPnp=yP.detach().cpu().numpy()
                for k in range(doMC):
                    # print(k)
                    yMC = model(xTest, doDropMC=True).detach().cpu().numpy()
                    ySS = ySS+np.square(yMC-yPnp)
                ySS = np.sqrt(ySS)/doMC
        if type(model) in [rnn.LstmCloseModel, rnn.AnnCloseModel, rnn.CNN1dLSTMmodel, rnn.CNN1dLSTMInmodel,
                           rnn.CNN1dLCmodel, rnn.CNN1dLCInmodel, rnn.CudnnInvLstmModel,
                           rnn.MultiInv_HBVModel, rnn.MultiInv_HBVTDModel, rnn.MultiInv_HBVTDModel_1]:
            if multiforcing is True:
                # model.train(mode=True)
                # yP, prcp_loss, prcp_wghts= model(xTest, zTest, prcp_loss_factor, multiforcing)

                # zTest.requires_grad_(True)
                # torch.set_grad_enabled(True)

                # yP, prcp_loss, pet_loss, temp_loss, prcp_pet_wghts, smooth_loss, slope_grad_daymet, slope_grad_maurer, slope_grad_nldas \
                #     = model(xTest, zTest, prcp_loss_factor, smooth_loss_factor, multiforcing)
                yP, prcp_loss, pet_loss, temp_loss, prcp_pet_wghts, smooth_loss = model(xTest, zTest, prcp_loss_factor, smooth_loss_factor, multiforcing)
            else:
                yP = model(xTest, zTest, prcp_loss_factor, smooth_loss_factor, multiforcing)
        if type(model) in [hydroDL.model.rnn.LstmCnnForcast]:
            yP = model(xTest, zTest)
        if type(model) in [cnn.LstmCnn1d]:
            xTest = xTest.permute(1, 2, 0)
            yP = model(xTest)
            yP = yP.permute(2, 0, 1)

        # CP-- marks the beginning of problematic merge
        # prcp_pet_wghts.requires_grad_(True)
        #
        # if multiforcing is True:
        #         if i == 0:
        #             slope_grad_ar_daymet = slope_grad_daymet.detach().cpu().numpy()
        #             slope_grad_ar_maurer = slope_grad_maurer.detach().cpu().numpy()
        #             slope_grad_ar_nldas = slope_grad_nldas.detach().cpu().numpy()
        #         else:
        #             slope_grad_daymet_tmp = slope_grad_daymet.detach().cpu().numpy()
        #             slope_grad_nldas_tmp = slope_grad_nldas.detach().cpu().numpy()
        #             slope_grad_maurer_tmp = slope_grad_maurer.detach().cpu().numpy()
        #             slope_grad_ar_daymet= np.concatenate([slope_grad_ar_daymet, slope_grad_daymet_tmp], axis=1)
        #             slope_grad_ar_maurer= np.concatenate([slope_grad_ar_maurer, slope_grad_maurer_tmp], axis=1)
        #             slope_grad_ar_nldas= np.concatenate([slope_grad_ar_nldas, slope_grad_nldas_tmp], axis=1)
        # model.train(mode=False)


        yOut = yP.detach().cpu().numpy().swapaxes(0, 1)
        if doMC is not False:
            yOutMC = ySS.swapaxes(0, 1)

        # save output
        for k in range(ny):
            f = fLst[k]
            pd.DataFrame(yOut[:, :, k]).to_csv(f, header=False, index=False)
        if doMC is not False:
            for k in range(ny):
                f = fLst[ny+k]
                pd.DataFrame(yOutMC[:, :, k]).to_csv(
                    f, header=False, index=False)

        if multiforcing is True:
            if i == 0:
                prcp_pet_wghts_ar = prcp_pet_wghts.detach().cpu().numpy()
            else:
                prcp_pet_wghts_tmp = prcp_pet_wghts.detach().cpu().numpy()
                prcp_pet_wghts_ar = np.concatenate([prcp_pet_wghts_ar, prcp_pet_wghts_tmp], axis=1)


                # torch.set_grad_enabled(True)

        model.zero_grad()
        torch.cuda.empty_cache()
    if multiforcing is True:
        # np.savetxt('/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss'+str(prcp_loss_factor)+'/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts1.csv', prcp_wghts_ar[:,:,0], delimiter=',')
        # for i in range(40):
        #     np.savetxt(filePathLst[0][:-43] + f'/{i}_grad_daymet.csv', slope_grad_ar_daymet[:,:,i], delimiter=',')
        #     np.savetxt(filePathLst[0][:-43] + f'/{i}_grad_maurer.csv', slope_grad_ar_maurer[:,:,i], delimiter=',')
        #     np.savetxt(filePathLst[0][:-43] + f'/{i}_grad_nldas.csv', slope_grad_ar_nldas[:,:,i], delimiter=',')
        for types in range(prcp_datatypes):
        # for types in range(3):
        #     np.savetxt(filePathLst[0][:-43] + f'/prcp_wghts{types + 1}.csv', prcp_pet_wghts_ar[:, :, types], delimiter=',')
            np.savetxt(filePathLst[0][:-43] + f'/prcp_wghts{types + 1}.csv', prcp_pet_wghts_ar[:, :, types], delimiter=',')
            # np.savetxt(filePathLst[0][:-43] + f'/pet_wghts{types + 1}.csv', prcp_pet_wghts_ar[:, :, types+prcp_datatypes+prcp_datatypes], delimiter=',')
            # np.savetxt(filePathLst[0][:-43] + f'/temp_wghts{types + 1}.csv', prcp_pet_wghts_ar[:, :, types+prcp_datatypes], delimiter=',')
        # np.savetxt(filePathLst[0][:-43] + f'/pet_wghts{types + 1}.csv', prcp_pet_wghts_ar[:, :, -2], delimiter=',')
        # np.savetxt(filePathLst[0][:-43] + f'/temp_wghts{types+1}.csv', prcp_pet_wghts_ar[:, :, -1], delimiter=',')
        # np.savetxt(filePathLst[0][:-42] + f'/pet_wghts{1}.csv', prcp_pet_wghts_ar[:, :, -1], delimiter=',')
            # np.savetxt('/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/daymetwithloss0smooth0.005/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts_revised.csv', prcp_wghts_ar[:, :, types], delimiter=',')
        # np.savetxt(filePathLst[0][:-43]+'/prcp_wghts1.csv', prcp_wghts_ar[:,:,0], delimiter=',')
        # # np.savetxt('/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss'+str(prcp_loss_factor)+'/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/prcp_wghts2.csv', prcp_wghts_ar[:,:,1], delimiter=',')
        # np.savetxt(filePathLst[0][:-43]+'/prcp_wghts2.csv', prcp_wghts_ar[:,:,1], delimiter=',')
        # np.savetxt(filePathLst[0][:-43]+'/prcp_wghts3.csv', prcp_wghts_ar[:,:,2], delimiter=',')
    for f in fLst:
        f.close()

    if batchSize == ngrid:
        # For Wenping's work to calculate loss of testing data
        # Only valid for testing without using minibatches
        yOut = torch.from_numpy(yOut)
        if type(model) in [rnn.CudnnLstmModel_R2P]:
            Parameters_R2P = torch.from_numpy(Parameters_R2P)
            if outModel is None:
                return yOut, Parameters_R2P
            else:
                return q, evap, Parameters_R2P
        else:
            return yOut


def test2Model(model, loaded_hbv, x1, x2, c, *, batchSize=None, filePathLst=None, doMC=False, outModel=None, savePath=None, prcp_loss_factor=15, multiforcing=False, smooth_loss_factor=0, prcp_datatypes=1):
# def testModel(model, x, c, *, batchSize=None, filePathLst=None, doMC=False, outModel=None, savePath=None, prcp_loss_factor=15, multiforcing=False, smooth_loss_factor=1):
    # outModel, savePath: only for R2P-hymod model, for other models always set None
    if type(x2) is tuple or type(x2) is list:
        x2, z2 = x2
        x1, z1 = x1
    else:
        z2 = None
        z1 = None
    ngrid, nt, nx = x1.shape
    if c is not None:
        nc = c.shape[-1]
    if type(model) in [rnn.MultiInv_HBVModel, rnn.MultiInv_HBVTDModel, rnn.MultiInv_HBVTDModel_1, rnn.prcp_weights]:
        ny=5 # streamflow
    else:
        ny = model.ny
    if batchSize is None:
        batchSize = ngrid
    if torch.cuda.is_available():
        model = model.cuda()

    model.train(mode=True)
    loaded_hbv.train(mode=False)
    if hasattr(model, 'ctRm'):
        if model.ctRm is True:
            nt = nt - model.ct
    # yP = torch.zeros([nt, ngrid, ny])
    iS = np.arange(0, ngrid, batchSize)
    iE = np.append(iS[1:], ngrid)

    # deal with file name to save
    if filePathLst is None:
        filePathLst = ['out' + str(x) for x in range(ny)]
    fLst = list()
    for filePath in filePathLst:
        if os.path.exists(filePath):
            os.remove(filePath)
        f = open(filePath, 'a')
        fLst.append(f)

    # forward for each batch
    for i in range(0, len(iS)):
        print('batch {}'.format(i))
        xTemp1 = x1[iS[i]:iE[i], :, :]
        xTemp2 = x2[iS[i]:iE[i], :, :]
        if c is not None:
            cTemp = np.repeat(
                np.reshape(c[iS[i]:iE[i], :], [iE[i] - iS[i], 1, nc]), nt, axis=1)
            xTest1 = torch.from_numpy(
                np.swapaxes(np.concatenate([xTemp1, cTemp], 2), 1, 0)).float()
            xTest2 = torch.from_numpy(
                np.swapaxes(np.concatenate([xTemp2, cTemp], 2), 1, 0)).float()
        else:
            xTest1 = torch.from_numpy(
                np.swapaxes(xTemp1, 1, 0)).float()
            xTest2 = torch.from_numpy(
                np.swapaxes(xTemp2, 1, 0)).float()
        if torch.cuda.is_available():
            xTest1 = xTest1.cuda()
            xTest2 = xTest2.cuda()
        if z1 is not None:

            zTemp1 = z1[iS[i]:iE[i], :, :]
            zTemp2 = z2[iS[i]:iE[i], :, :]

            zTest1 = torch.from_numpy(np.swapaxes(zTemp1, 1, 0)).float()
            zTest2 = torch.from_numpy(np.swapaxes(zTemp2, 1, 0)).float()
            if torch.cuda.is_available():
                zTest1 = zTest1.cuda()
                zTest2 = zTest2.cuda()

        if type(model) in [rnn.LstmCloseModel, rnn.AnnCloseModel, rnn.CNN1dLSTMmodel, rnn.CNN1dLSTMInmodel,
                           rnn.CNN1dLCmodel, rnn.CNN1dLCInmodel, rnn.CudnnInvLstmModel,
                           rnn.MultiInv_HBVModel, rnn.MultiInv_HBVTDModel, rnn.MultiInv_HBVTDModel_1, rnn.prcp_weights]:

            xP, prcp_loss, prcp_pet_wghts, grad_daymet, grad_maurer, grad_nldas = model(xTest1, zTest1, prcp_loss_factor)
            yP = loaded_hbv(xP, zTest2, prcp_loss_factor=0, smooth_loss_factor=0, multiforcing=False)




        # if i == 0:
        #     grad_ar_daymet = grad_daymet.detach().cpu().numpy()
        #     grad_ar_maurer = grad_maurer.detach().cpu().numpy()
        #     grad_ar_nldas = grad_nldas.detach().cpu().numpy()
        # else:
        #     grad_daymet_tmp = grad_daymet.detach().cpu().numpy()
        #     grad_nldas_tmp = grad_nldas.detach().cpu().numpy()
        #     grad_maurer_tmp = grad_maurer.detach().cpu().numpy()
        #     grad_ar_daymet= np.concatenate([grad_ar_daymet, grad_daymet_tmp], axis=1)
        #     grad_ar_maurer= np.concatenate([grad_ar_maurer, grad_maurer_tmp], axis=1)
        #     grad_ar_nldas= np.concatenate([grad_ar_nldas, grad_nldas_tmp], axis=1)




        yOut = yP.detach().cpu().numpy().swapaxes(0, 1)
        if doMC is not False:
            yOutMC = ySS.swapaxes(0, 1)

        # save output
        for k in range(ny):
            f = fLst[k]
            pd.DataFrame(yOut[:, :, k]).to_csv(f, header=False, index=False)
        if doMC is not False:
            for k in range(ny):
                f = fLst[ny+k]
                pd.DataFrame(yOutMC[:, :, k]).to_csv(
                    f, header=False, index=False)
        if i == 0:
            prcp_pet_wghts_ar = prcp_pet_wghts.detach().cpu().numpy()
        else:
            prcp_pet_wghts_tmp = prcp_pet_wghts.detach().cpu().numpy()
            prcp_pet_wghts_ar = np.concatenate([prcp_pet_wghts_ar, prcp_pet_wghts_tmp], axis=1)



        model.zero_grad()
        torch.cuda.empty_cache()
    for i in range(z1.shape[2]):
        k=1
        # np.savetxt(filePathLst[0][:-43] + f'/{i}_grad_daymet.csv', grad_ar_daymet[:,:,i], delimiter=',')
        # np.savetxt(filePathLst[0][:-43] + f'/{i}_grad_maurer.csv', grad_ar_maurer[:,:,i], delimiter=',')
        # np.savetxt(filePathLst[0][:-43] + f'/{i}_grad_nldas.csv', grad_ar_nldas[:,:,i], delimiter=',')
    for types in range(prcp_pet_wghts_ar.shape[2]):
        np.savetxt(filePathLst[0][:-43] + f'/prcp_wghts{types + 1}.csv', prcp_pet_wghts_ar[:, :, types], delimiter=',')

    for f in fLst:
        f.close()

    if batchSize == ngrid:
        # For Wenping's work to calculate loss of testing data
        # Only valid for testing without using minibatches
        yOut = torch.from_numpy(yOut)
        if type(model) in [rnn.CudnnLstmModel_R2P]:
            Parameters_R2P = torch.from_numpy(Parameters_R2P)
            if outModel is None:
                return yOut, Parameters_R2P
            else:
                return q, evap, Parameters_R2P
        else:
            return yOut


def testModelCnnCond(model, x, y, *, batchSize=None):
    ngrid, nt, nx = x.shape
    ct = model.ct
    ny = model.ny
    if batchSize is None:
        batchSize = ngrid
    xTest = torch.from_numpy(np.swapaxes(x, 1, 0)).float()
    # cTest = torch.from_numpy(np.swapaxes(y[:, 0:ct, :], 1, 0)).float()
    cTest = torch.zeros([ct, ngrid, y.shape[-1]], requires_grad=False)
    for k in range(ngrid):
        ctemp = y[k, 0:ct, 0]
        i0 = np.where(np.isnan(ctemp))[0]
        i1 = np.where(~np.isnan(ctemp))[0]
        if len(i1) > 0:
            ctemp[i0] = np.interp(i0, i1, ctemp[i1])
            cTest[:, k, 0] = torch.from_numpy(ctemp)

    if torch.cuda.is_available():
        xTest = xTest.cuda()
        cTest = cTest.cuda()
        model = model.cuda()

    model.train(mode=False)

    yP = torch.zeros([nt - ct, ngrid, ny])
    iS = np.arange(0, ngrid, batchSize)
    iE = np.append(iS[1:], ngrid)
    for i in range(0, len(iS)):
        xTemp = xTest[:, iS[i]:iE[i], :]
        cTemp = cTest[:, iS[i]:iE[i], :]
        yP[:, iS[i]:iE[i], :] = model(xTemp, cTemp)
    yOut = yP.detach().cpu().numpy().swapaxes(0, 1)
    return yOut


def randomSubset(x, y, dimSubset):
    ngrid, nt, nx = x.shape
    batchSize, rho = dimSubset
    xTensor = torch.zeros([rho, batchSize, x.shape[-1]], requires_grad=False)
    yTensor = torch.zeros([rho, batchSize, y.shape[-1]], requires_grad=False)
    iGrid = np.random.randint(0, ngrid, [batchSize])
    iT = np.random.randint(0, nt - rho, [batchSize])
    for k in range(batchSize):
        temp = x[iGrid[k]:iGrid[k] + 1, np.arange(iT[k], iT[k] + rho), :]
        xTensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
        temp = y[iGrid[k]:iGrid[k] + 1, np.arange(iT[k], iT[k] + rho), :]
        yTensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
    if torch.cuda.is_available():
        xTensor = xTensor.cuda()
        yTensor = yTensor.cuda()
    return xTensor, yTensor


def randomIndex(ngrid, nt, dimSubset, bufftime=0):
    batchSize, rho = dimSubset
    iGrid = np.random.randint(0, ngrid, [batchSize])
    iT = np.random.randint(0+bufftime, nt - rho, [batchSize])
    return iGrid, iT


def selectSubset(x, iGrid, iT, rho, *, c=None, tupleOut=False, LCopt=False, bufftime=0):
    nx = x.shape[-1]
    nt = x.shape[1]
    if x.shape[0] == len(iGrid):   #hack
        iGrid = np.arange(0,len(iGrid))  # hack
    if nt <= rho:
        iT.fill(0)

    batchSize = iGrid.shape[0]
    if iT is not None:
        # batchSize = iGrid.shape[0]
        xTensor = torch.zeros([rho+bufftime, batchSize, nx], requires_grad=False)
        for k in range(batchSize):
            temp = x[iGrid[k]:iGrid[k] + 1, np.arange(iT[k]-bufftime, iT[k] + rho), :]
            xTensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
    else:
        if LCopt is True:
            # used for local calibration kernel: FDC, SMAP...
            if len(x.shape) == 2:
                # Used for local calibration kernel as FDC
                # x = Ngrid * Ntime
                xTensor = torch.from_numpy(x[iGrid, :]).float()
            elif len(x.shape) == 3:
                # used for LC-SMAP x=Ngrid*Ntime*Nvar
                xTensor = torch.from_numpy(np.swapaxes(x[iGrid, :, :], 1, 2)).float()
        else:
            # Used for rho equal to the whole length of time series
            xTensor = torch.from_numpy(np.swapaxes(x[iGrid, :, :], 1, 0)).float()
            rho = xTensor.shape[0]
    if c is not None:
        nc = c.shape[-1]
        temp = np.repeat(
            np.reshape(c[iGrid, :], [batchSize, 1, nc]), rho+bufftime, axis=1)
        cTensor = torch.from_numpy(np.swapaxes(temp, 1, 0)).float()

        if (tupleOut):
            if torch.cuda.is_available():
                xTensor = xTensor.cuda()
                cTensor = cTensor.cuda()
            out = (xTensor, cTensor)
        else:
            out = torch.cat((xTensor, cTensor), 2)
    else:
        out = xTensor

    if torch.cuda.is_available() and type(out) is not tuple:
        out = out.cuda()
    return out
