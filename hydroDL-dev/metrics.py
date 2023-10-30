from hydroDL.post import plot, stat
import pandas as pd
import numpy as np
# pred = np.load('/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss23smooth0/'
#                'BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/pred50.npy', allow_pickle=True)
# pred = np.load("/data/kas7897/dPLHBVrelease/output/rnnStreamflow/CAMELSDemo-daymet/TestRun/epochs300_batch100_rho365_hiddensize256_Tstart19801001_Tend19951001/pred.npy", allow_pickle=True)
# pred = np.load("/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/maurer/BuffOpt0/RMSE_para0.25/111111/"
#                "Train19801001_19951001Test19951001_20051001Buff5478Staind5477/pred50.npy", allow_pickle=True)
pred = np.load("/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/maurer_extendedwithloss0smooth0_11/"
               "BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/pred50.npy", allow_pickle=True)
obs = np.load('/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss23smooth0/'
              'BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20051001Buff5478Staind5477/obs.npy', allow_pickle=True)

# statDict = stat.statError(pred[:,:,0], obs.squeeze())
# lstm_daymet = np.load("/data/kas7897/dPLHBVrelease/output/rnnStreamflow/CAMELSDemo-daymet/TestRun/epochs300_batch100_rho365_hiddensize256_Tstart19801001_Tend19951001/Eva300.npy", allow_pickle=True)

statDict = stat.statError(pred[:,:,0], obs.squeeze())

k=1