import os, logging
import _pickle as pickle

import numpy as np
import pandas as pd

from numpy import random

from utils import metrics_at_k, filterTrainDt


### Load the dataset
with open('basicPreprocess.pkl', 'rb') as f:
    [impressions, newItems] = pickle.load(f)
newItems =  newItems.loc[newItems.mlogId.isin(impressions['mlogId'].values)]

logging.basicConfig(level=logging.DEBUG, filename="baselineRandomEDM.log", filemode="w+",format="%(asctime)-15s %(levelname)-8s %(message)s")
periodsIntervals = [2, 9, 16, 23, 30]
"""
The baseline popularity strategies
# (1) first apply the recommendation strategy
# (2) Predict the probability that the user with stay
# (3) Creator the new data
"""

i=0
logging.info("Evaluation of Week {}".format(i+2))

subTrainingStart, subTrainingEnd = periodsIntervals[0], periodsIntervals[i+2]
subTestingStart, subTestingEnd = periodsIntervals[i+2], (periodsIntervals[i+4])

subTrainDt = impressions[(impressions['dt']>=subTrainingStart) & (impressions['dt']<subTrainingEnd)]
subTestDt = impressions[(impressions['dt']>=subTestingStart) & (impressions['dt']<subTestingEnd)]

subTrainDt = filterTrainDt(subTrainDt,0,0)
subTestDt = filterTrainDt(subTestDt, 0,0)

### Filter half of the new videos
#itemCandidates = subTestDt.loc[subTestDt.publishDt<16,'mlogId'].unique()
wilsonScoreThreshold = np.percentile(newItems['mlogWilsonScore'].values, 33)

filterItems = newItems.loc[newItems['mlogWilsonScore']<=wilsonScoreThreshold,]
#subTestDt = subTestDt.loc[subTestDt.mlogId.isin(itemCandidates),]
itemCandidates = impressions.loc[~impressions.mlogId.isin(filterItems),'mlogId'].unique()
### Predict with the most popularity strategies
np.random.seed(54321)
testUsersList = subTestDt['userId'].unique()
predDt = []
for userId in testUsersList:
    subPred = pd.DataFrame({'mlogId':random.choice(itemCandidates, 300),'rating_preds': np.linspace(start=1, stop=0.5, num=300)})
    subPred['userId'] = userId
    predDt.append(subPred)
predDt = pd.concat(predDt, axis=0)

##Filter those already recommended items 
recommendDt  = subTrainDt.loc[subTrainDt.userId.isin(testUsersList), ['mlogId','userId','isClick']]
predDt = pd.merge(predDt, recommendDt, how='left', on=['mlogId','userId'])
predDt = predDt[~(predDt['isClick']>=0)]
predDt.drop(['isClick'], axis=1, inplace=True)
### The metrics evaluation 
metricsDt = []
for K in [20, 50, 100]:
    hitK, precK, recallK, nDCGK = metrics_at_k(subTestDt, predDt, K)
    logging.info("The top{} metrics: hits: {:.4}, precision: {:.4}, recall: {:.4}, nDCG: {:.4}".format(K, hitK, precK, recallK, nDCGK))
    subRow = [K, hitK, precK, recallK, nDCGK]
    metricsDt.append(subRow)
metricsDt = pd.DataFrame(metricsDt, columns=['K', 'hit', 'prec','recall','nDCG'])
metricsDt.to_csv("dataAllEDM.csv", index=False, sep=',',float_format='%.5f')
