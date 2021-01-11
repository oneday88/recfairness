import os
import _pickle as pickle

import numpy as np
import scipy as sp
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

"""
Load the dataset
"""
### Load the data
data_dir_path = '/Users/yitianchen/Work/Recommendation/Dataset/netease'
mlogStats = pd.read_csv(data_dir_path + '/csv_data_fix'+'/mlog_stats.csv',low_memory=False, na_values = ['NA','NULL','NaN','\\N'])
mlogDemo = pd.read_csv(data_dir_path + '/csv_data_fix'+'/mlog_demographics.csv',low_memory=False, na_values = ['NA','NULL','NaN','\\N'])
creatorDemo = pd.read_csv(data_dir_path + '/csv_data_fix'+'/creator_demographics.csv',low_memory=False, na_values = ['NA','NULL','NaN','\\N'])

# The preprocess, get the published time
mlogDemo['dt'] = 30+2-mlogDemo['publishTime']
# Add the information of creators
mlogStats = pd.merge(mlogStats, mlogDemo[['mlogId','creatorId']], on='mlogId', how='left')

##The creator publish stats, per week
creatingStats = mlogDemo.loc[(mlogDemo.dt>1) & (mlogDemo.dt<30),].groupby(["creatorId","dt"]).size().reset_index()
creatingStats.columns = ["creatorId", "dt", "mlogCount"]
creatingStats['nWeek'] = ((creatingStats['dt']-2)/7).astype('int32')+1
creatingStats = creatingStats.groupby(["creatorId","nWeek"]).agg({"mlogCount":sum}).reset_index()

## The creator feedback stats by week
mlogStats = mlogStats.loc[(mlogStats['dt']>1) & (mlogStats['dt']<30), ["creatorId","mlogId","dt","userImprssionCount","userClickCount"]]
mlogStats['nWeek'] = ((mlogStats['dt']-2)/7).astype('int32')+1

creatorFeedbackDt =  mlogStats.groupby(["creatorId","nWeek"]).agg({'userImprssionCount': sum,'userClickCount':sum}).reset_index()

## The combine the dataset
churnDt = pd.merge(creatorFeedbackDt, creatingStats, on=["creatorId","nWeek"], how='outer')
churnDt =  churnDt.loc[(~pd.isna(churnDt.creatorId)) ,]
churnDt = churnDt.fillna(0)

###The churn label
churnDt = churnDt.sort_values(by=["creatorId","nWeek"], ascending=True)
churnDt['nextPMlogCnt'] = churnDt.groupby('creatorId')['mlogCount'].shift(-1)
churnDt['isStay'] = churnDt['nextPMlogCnt'].apply(lambda x: 1 if x >0 else 0)
churnDt = churnDt.fillna(0)

"""
### The average creator click rate: the quality of the creators, we used the week1,2,3's average clicks rate to represent the quality
"""
#churnDt['creatorTotalImpressions']  = churnDt['userImprssionCount']*(churnDt['nWeek']<4)
#churnDt['creatorTotalClicks']  = churnDt['userImprssionCount']*(churnDt['nWeek']<4)

#churnDt['creatorTotalImpressions'] = churnDt.groupby('creatorId')['creatorTotalImpressions'].transform(np.sum)
#churnDt['creatorTotalClicks'] = churnDt.groupby('creatorId')['creatorTotalClicks'].transform(np.sum)

churnDt['creatorAveClickRate'] = (churnDt['userClickCount']+0)/(churnDt['userImprssionCount']+5)

#churnDt.drop(['creatorTotalImpressions','creatorTotalClicks'], axis=1, inplace=True)

### Add creator attributes
churnDt = pd.merge(churnDt, creatorDemo[['creatorId','registeredMonthCnt','followeds','level']], on='creatorId', how='left')
## select those stay in the previous week
churnDt = churnDt[churnDt['mlogCount']>0]

"""
The creator churn model
"""
churnDt['logImpress'] = np.log(churnDt['userImprssionCount']+1)
churnDt['logFans'] = np.log(churnDt['followeds']+1)

import statsmodels.api as sm
features = ['logFans','creatorAveClickRate','logImpress']
# defining the dependent and independent variables
Xtrain = churnDt.loc[churnDt['nWeek']<3, features]
Xtrain =sm.add_constant(Xtrain)
ytrain = churnDt.loc[churnDt['nWeek']<3,['isStay']]

##The testing
Xtest = churnDt.loc[churnDt['nWeek']==3, features]
Xtest =sm.add_constant(Xtest)
ytest = churnDt.loc[churnDt['nWeek']==3,['isStay']]

# building the model and fitting the data
creatorChurnFit = sm.Logit(ytrain, Xtrain).fit()
print(creatorChurnFit.summary())

yTrainPred = creatorChurnFit.predict(Xtrain)
yTestPred = creatorChurnFit.predict(Xtest)

##The evaluation metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
## The auc
print(roc_auc_score(ytrain, yTrainPred))
print(roc_auc_score(ytest, yTestPred))

import matplotlib.pyplot as plt
import scikitplot as skplt
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
skplt.metrics.plot_roc_curve(ytrain['isStay'], np.column_stack([1-yTrainPred.values, yTrainPred.values]), curves='micro', title='Training ROC: ')
skplt.metrics.plot_roc_curve(ytest['isStay'], np.column_stack([1-yTestPred.values, yTestPred.values]), curves='micro')
plt.show()


plotDt1 = pd.DataFrame({'label': ytrain['isStay'], 'pred': yTrainPred})
plotDt1['mark']='training'
plotDt2 = pd.DataFrame({'label': ytest['isStay'], 'pred': yTestPred})
plotDt2['mark']='testing'

plotDt = plotDt1.append(plotDt2)
plotDt.to_csv("plotDt.csv", index=False)



