import os, logging
import _pickle as pickle

import numpy as np
import pandas as pd

import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn

from utils import filterTrainDt

"""
Load the dataset
"""
with open('basicPreprocess.pkl', 'rb') as f:
    [impressions, label_enc_list,newItems] = pickle.load(f)

newItems = newItems.loc[newItems['mlogId'].isin(impressions['newMlogId'].unique())]

logging.basicConfig(level=logging.DEBUG, filename="CurveExperiment.log", filemode="w",format="%(asctime)-15s %(levelname)-8s %(message)s")
impressions = impressions.fillna(0)

"""
The baseline CF model
"""
embedding_size = 30
data_ctx = mx.cpu()
model_ctx = mx.cpu()

max_user_count = impressions['userId'].max()

max_item_count = impressions['mlogId'].max()
max_creator_count = impressions['creatorId'].max()
max_artist_count = impressions['artistId'].max()
max_song_count = impressions['songId'].max()

max_pos_count = impressions['impressPosition'].max()

from CFModels import baseNCF
from CFTrainer import CFTrainer
pmfModel = baseNCF(embedding_size=embedding_size, user_input_dim=max_user_count, item_input_dim=[max_item_count, max_creator_count,max_artist_count,max_song_count], pos_input_dim=max_pos_count)

file_name = "checkpoints/neuralCF.params"
pmfModel.load_parameters(file_name, ctx=model_ctx)

cf = CFTrainer(pmfModel,  data_ctx, model_ctx)

periodsIntervals = [2, 9, 16, 23, 30]

i=0
logging.info("Evaluation of Week {}".format(i+2))
subTrainingStart, subTrainingEnd = periodsIntervals[0], periodsIntervals[i+2]
subTestingStart, subTestingEnd = periodsIntervals[i+2], (periodsIntervals[i+4])

subTrainDt = impressions[(impressions['dt']>=subTrainingStart) & (impressions['dt']<subTrainingEnd)]
subTestDt = impressions[(impressions['dt']>=subTestingStart) & (impressions['dt']<subTestingEnd)]

subTrainDt2 = filterTrainDt(subTrainDt,0,0)
subTestDt = filterTrainDt(subTestDt, 0,0)
### The popularity strategies
### The popularity strategies
#wilsonScoreThreshold = np.percentile(newItems['mlogWilsonScore'].values, 33)

#filterItems = newItems.loc[newItems['mlogWilsonScore']<=wilsonScoreThreshold,]
#subTestDt = subTestDt.loc[subTestDt.mlogId.isin(itemCandidates),]
#subTestDt = subTestDt.loc[~subTestDt.mlogId.isin(filterItems),]


#testUsersList = subTestDt['userId'].unique()
    
trainX = subTrainDt2[['userId','mlogId','creatorId','artistId','songId','impressPosition','userClickCount','userZanCount']].values
trainY = subTrainDt2['isClick'].values


## The new items

quantileList = list(range(21))
metricsDt = []
for quantilePoint in quantileList:
    ###The new items
    subN = quantilePoint*5
    print(subN)
    filterIndex = int(1.0*newItems.shape[0]*subN/100)
    filterItems = newItems.iloc[0:filterIndex,]['mlogId'].values
    print(filterItems.shape)
    subTestDt = subTestDt.loc[~subTestDt.newMlogId.isin(filterItems),]
    print(subTestDt.shape)
    ##The recomended dataset
    testUsersList = subTestDt['userId'].unique()
    recommendDt  = subTrainDt.loc[subTrainDt.userId.isin(testUsersList), ['mlogId','userId','isClick']]
    
    subHitK, subPrecK, subRecallK, subnDCGK = cf.topK_evaluator(subTestDt, recommendDt,K_list=[20,50,100])
    subRow = [subN]+subHitK+subPrecK+subRecallK+subnDCGK
    metricsDt.append(subRow)

metricsDt = pd.DataFrame(metricsDt, columns=['delta', 'hit20', 'hit50','hit100', 'recall20','recall50','recall100', 'prec20', 'prec50','prec100','nDCG20','nDCG50', 'nDCG100'])
metricsDt.to_csv("metricsDt2.csv", index=False, sep=',')

