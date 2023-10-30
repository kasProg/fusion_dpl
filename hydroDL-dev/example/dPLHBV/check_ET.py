import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.stats

lossfactor= 200
smoothfactor=0
crd = pd.read_csv("/data/kas7897/dPLHBVrelease/hydroDL-dev/gages_list.csv")

# et_model = pd.read_csv(
#     f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/allprcp_withloss200smooth0/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/All_Buff5478_19951001_20051001_ep50_ET.csv',
#     header=None)
et_model = pd.read_csv("/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/"
                       "allprcp_withloss23smooth0/BuffOpt0/RMSE_para0.25/111111/Fold1/"
                       "T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/All_Buff5478_19951001_20051001_ep50_ET.csv", header=None)
# et_model = pd.read_csv(
#     f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/daymet/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/All_Buff5478_19951001_20051001_ep50_ET.csv',
#     header=None)
# et_model = pd.read_csv(
#     f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/nldas/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/All_Buff5478_19951001_20051001_ep50_ET.csv',
#     header=None)
# "/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV/ALL/TDTestforc/TD1_13/all_extended_withloss200smooth0_old/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_13_Buff_365_Mul_16/All_Buff5478_19951001_20051001_ep50_ET.csv"

dateTrain1 = datetime.strptime(str(20010101), '%Y%m%d')
dateTrain2 = datetime.strptime(str(20110101), '%Y%m%d')
delta_train = dateTrain2 - dateTrain1
num_days_train = delta_train.days
et_obs = np.zeros((671,num_days_train))

k = 0
for i in range(2001, 2011):
    et = pd.read_csv(f"/data/shared_data/CAMELSET/quick_prj/CSV_8day/{i}/ET.csv")
    et[['temp1', 'temp2', 'gage']] = et['Unnamed: 0'].str.split('_', expand=True)
    et.drop(columns=['Unnamed: 0', 'temp1', 'temp2'], inplace=True)
    num_cols = et.shape[1]
    for j in range(len(crd)):
        gage = crd['gage'][j]
        et_gage = et[et['gage']==str(gage)]
        # et_gage_copy = et_gage.copy()
        # et_gage.drop(columns = ['gage'], inplace=True)
        # et_gage_values = et_gage.values.reshape(-1, 1)
        et_obs[j][k:k+num_cols-1] = et_gage.loc[:, et_gage.columns != 'gage']

        # et_obs = np.append(et_obs, np.expand_dims(et_gage, axis=1), axis=1)
        # et_obs = np.append(et_obs[:, np.newaxis], et_gage[:, np.newaxis], axis=1)
    k = k + num_cols - 1

# et_obs = np.where(et_obs == -9999.00, np.nan, et_obs)
et_obs = np.where(et_obs < 0, np.nan, et_obs)

et_model_range = np.array(et_model.loc[:, 7397:]) #from 2001
# et_model_range = np.array(et_model.loc[:, 7390:]) #from 2001
# et_obs_range = et_obs[:, :1734] #till 2005-09-30
# et_obs_range = et_obs[:, :1734] #till 2005-09-30
et_obs_range = et_obs[:, :1727] #till 2005-09-22



# nse = 1 - np.sum((et_model_range - et_obs_range) ** 2, axis=1) / np.sum((et_obs_range - np.mean(et_obs_range, axis=1, keepdims=True)) ** 2, axis=1)

# Calculate correlations
# correlations = np.nanmean(np.multiply(et_model_range - np.nanmean(et_model_range, axis=1, keepdims=True), et_obs_range - np.nanmean(et_obs_range, axis=1, keepdims=True)), axis=1) / (np.nanstd(et_model_range, axis=1) * np.nanstd(et_obs_range, axis=1))

nan_columns = np.all(np.isnan(et_obs_range), axis=0)
et_obs_range_filtered = et_obs_range[:, ~nan_columns]



et8_model = np.zeros_like(et_obs_range_filtered)
et8_model[:,0] = np.sum(et_model_range[:, :8], axis = 1)
# et8_model[:,0] = np.sum(et_model_range[:, :8], axis = 1)
counter = 1
l=0
# for i in range(1, et_obs_range.shape[1]):
for i in range(8, et_obs_range.shape[1]):
    if np.isnan(et_obs_range[:, i]).all():
        l = l + 1
    else:
        if l!=7:
            # et8_model[:,counter] = (np.sum(et_model_range[:, i-l:i], axis = 1))*8/(l+1)
            et8_model[:,counter] = (np.sum(et_model_range[:, i:i+l+1], axis = 1))*8/(l+1)
        else:
            # et8_model[:, counter] = np.sum(et_model_range[:, i - l:i], axis=1)
            et8_model[:, counter] = np.sum(et_model_range[:, i:i+l+1], axis=1)
        l = 0
        counter = counter + 1


# nse_filtered = 1 - np.nansum((et_model_range_filtered - et_obs_range_filtered ) ** 2, axis=1) / np.nansum((et_obs_range_filtered  - np.nanmean(et_obs_range_filtered , axis=1, keepdims=True)) ** 2, axis=1)
et8_mean_spatial = np.nanmean(et8_model, axis=1)
et_obs_range_filtered = et_obs_range_filtered/10
et8obs_mean_spatial = np.nanmean(et_obs_range_filtered, axis=1)
# correlation = np.corrcoef(et8_model, et_obs_range_filtered)[0, 1]
CORR = []
R2 =[]
NSE =[]
for i in range(671):
    # valid_indices = np.logical_and(~np.isnan(x), ~np.isnan(y))
    ind = np.where(np.logical_and(~np.isnan(et8_model[i]), ~np.isnan(et_obs_range_filtered[i])))[0]
    xx = et8_model[i][ind]
    yy = et_obs_range_filtered[i][ind]
    corr = scipy.stats.pearsonr(xx, yy)[0]
    yymean = yy.mean()
    yystd = np.std(yy)
    xxmean = xx.mean()
    xxstd = np.std(xx)
    SST = np.sum((yy - yymean) ** 2)
    SSReg = np.sum((xx - yymean) ** 2)
    SSRes = np.sum((yy - xx) ** 2)
    R2.append(1 - SSRes / SST)
    NSE.append(1 - SSRes / SST)
    CORR.append(corr)
correlation = np.array(CORR)
nse = np.array(NSE)
r2 = np.array(R2)
print(np.nanmedian(correlation))
print(np.nanmedian(r2))
print(np.nanmedian(NSE))
# plt.plot(et8obs_mean_spatial, et8_mean_spatial, 'o')
spatial_correlation = scipy.stats.pearsonr(et8obs_mean_spatial, et8_mean_spatial)[0]
print(f"spatial corr: {spatial_correlation}")

