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
### Load the dataset
with open('basicPreprocess.pkl', 'rb') as f:
    [impressions, label_enc_list,newItems] = pickle.load(f)

logging.basicConfig(level=logging.DEBUG, filename="EDMbaselineCF.log", filemode="w",format="%(asctime)-15s %(levelname)-8s %(message)s")
impressions = impressions.fillna(0)

"""
The baseline CF model
"""
embedding_size = 30
data_ctx = mx.gpu()
model_ctx = mx.gpu()

max_user_count = impressions['userId'].max()
max_item_count = impressions['mlogId'].max()
max_pos_count = impressions['impressPosition'].max()

from CFModels import baseCF
from CFTrainer import CFTrainer
pmfModel = baseCF(embedding_size=embedding_size, user_input_dim=max_user_count, item_input_dim=max_item_count, pos_input_dim=max_pos_count)

cf = CFTrainer(pmfModel,  data_ctx, model_ctx)
###The parameters for deep learning framework
learning_rate = 0.01
batch_size = 1024
epochs = 50

### Parameters for the model
sample_rate = 0.5
negative_sample_rate = 0.3
CELoss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
huberLoss = gluon.loss.HuberLoss()
#initializer = mx.init.Xavier(magnitude=2.24)
initializer = mx.init.Normal(0.05)
optimizer = 'adam';

trainer_params_list = {'learning_rate': learning_rate, 'sample_rate':sample_rate, 'negative_sample_rate': negative_sample_rate,
                        'batch_size': batch_size, 'epochs': epochs,
                        'loss_func': huberLoss, 'initializer': initializer, 'optimizer':optimizer}

"""
The model training
"""
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
wilsonScoreThreshold = np.percentile(newItems['mlogWilsonScore'].values, 33)

filterItems = newItems.loc[newItems['mlogWilsonScore']<=wilsonScoreThreshold,]
#subTestDt = subTestDt.loc[subTestDt.mlogId.isin(itemCandidates),]
subTestDt = subTestDt.loc[~subTestDt.mlogId.isin(filterItems),]

trainX = subTrainDt2[['userId','mlogId','impressPosition','userClickCount','userZanCount']].values
trainY = subTrainDt2['isClick'].values

testUsersList = subTestDt['userId'].unique()
recommendedDt  = subTrainDt.loc[subTrainDt.userId.isin(testUsersList), ['mlogId','userId','isClick']]
training_log = cf.fit(trainX, trainY, trainer_params_list, subTestDt,recommendedDt)
