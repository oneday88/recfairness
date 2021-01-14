import math
import logging

import numpy as np
import pandas as pd

import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn

from utils import metrics_at_k


"""
The trainer framework for different CF (collaborative filtering) models
"""
class CFTrainer(object):
    def __init__(self, MF_model=None, model_ctx=mx.cpu(), data_ctx=mx.cpu()):
        self.model_ctx = model_ctx
        self.data_ctx = data_ctx
        self.model = MF_model

    def topK_evaluator(self, test_dt,recommendedDt, K_list=[20,50,100,200],user_batch_size=64, eval_count=None):
        test_userId = test_dt['userId'].unique()
        if eval_count is not None:
            test_userId = np.random.choice(test_userId,eval_count)
        test_mlogId = test_dt[['mlogId','creatorId','songId','artistId','userClickCount','userZanCount']].drop_duplicates().values
        
        pred_dt = self.predict(test_userId, test_mlogId,recommendedDt, user_batch_size)
        pred_dt = pd.merge(pred_dt, recommendedDt, how='left', on=['mlogId','userId'])
        pred_dt = pred_dt[~(pred_dt['isClick']>=0)]
        pred_dt.drop(['isClick'], axis=1, inplace=True)

        for K in K_list:
            hitK, precK, recallK, nDCGK = metrics_at_k(test_dt, pred_dt, K)
            logging.info("The top{} metrics: hits: {:.4}, precision: {:.4}, recall: {:.4}, nDCG: {:.4}".format(K, hitK, precK, recallK, nDCGK))

    def basic_evaluator(self, test_x, test_y, loss_func, metric, batch_size=64):
        test_x = nd.array(test_x, ctx=self.data_ctx)
        test_y = nd.array(test_y, ctx=self.data_ctx)
        test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(test_x, test_y), batch_size=batch_size,shuffle=True)
        ### The mae and rmse
        loss_avg = 0.
        for i, (data, rating) in enumerate(test_data):
            data = data.as_in_context(self.model_ctx)
            rating = rating.as_in_context(self.model_ctx)
            predictions = self.model(data)
            loss = loss_func(predictions, rating)
            metric.update(preds=predictions, labels=rating)
            loss_avg = loss_avg * i / (i + 1) + nd.mean(loss).asscalar() / (i + 1)
        return metric.get()[1], loss_avg

    def predict(self, user_id_test, item_id_test, recommendedDt, user_batch_size=252, K=500):
        itemCount,itemDim = item_id_test.shape
        n_batches = int(len(user_id_test)/user_batch_size)+1
        total_pred_dt = []
        for i in range(n_batches):
            sub_beg,sub_end = i*user_batch_size, min((i+1)*user_batch_size,len(user_id_test))
            sub_user_id_test = user_id_test[sub_beg:sub_end]
            sub_pairwise_list = np.column_stack([np.tile(sub_user_id_test, itemCount), np.tile(item_id_test, len(sub_user_id_test)).reshape(-1, itemDim)])
            row_count, col_count = sub_pairwise_list.shape
            pos_constant = np.ones((row_count,col_count+1))
            pos_constant[:,:-1] = sub_pairwise_list
            sub_pairwise_pred = (self.model(nd.array(pos_constant, ctx=self.data_ctx))).asnumpy()
            sub_pred_dt = pd.DataFrame({'userId':sub_pairwise_list[:,0], 'mlogId':sub_pairwise_list[:,1], 'rating_preds': sub_pairwise_pred})
            sub_pred_dt = sub_pred_dt.groupby(['userId']).apply(lambda x: x.nlargest(K, ['rating_preds'])).reset_index(drop=True)
            total_pred_dt.append(sub_pred_dt)
        total_pred_dt = pd.concat(total_pred_dt, axis=0)
        return total_pred_dt

    def fit(self, train_X, train_Y, params_dict, test_data=None, recommendedDt=None):
        ### The parameters
        batch_size = params_dict['batch_size']
        epochs = params_dict['epochs']
        # The sample rate
        sample_rate = params_dict['sample_rate']
        negative_sample_rate = params_dict['negative_sample_rate']

        optimizer = params_dict['optimizer']
        learning_rate = params_dict['learning_rate']
        initializer = params_dict['initializer']
        loss_func = params_dict['loss_func']
        ### The training data
        zeroIndex = np.where(train_Y==0)[0]
        ### The model initialization
        self.model.collect_params().initialize(initializer, ctx=self.model_ctx)
        ### The trainer
        trainer = gluon.Trainer(self.model.collect_params(), optimizer=optimizer, optimizer_params={'learning_rate': learning_rate})

        num_samples = len(train_Y)
        batch_count = int(math.ceil(num_samples / batch_size))

        history = dict()
        loss_train_seq = []
        mae_train_seq = []
        loss_test_seq = []
        mae_test_seq = []

        for e in range(epochs):
            oneIndex = np.where(train_Y==1)[0]
            zeroIndex = np.where(train_Y==0)[0]
            zeroIndex = np.random.choice(zeroIndex, size=int((1-negative_sample_rate)*len(zeroIndex)))
            sampleIndex = np.concatenate([oneIndex, zeroIndex])
            subTrainX = nd.array(train_X[sampleIndex],ctx=self.data_ctx)
            subTrainY = nd.array(train_Y[sampleIndex],ctx=self.data_ctx)
            train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(subTrainX, subTrainY), batch_size=batch_size, shuffle=True)
            cumulative_loss = 0
            for i, (data, rating) in enumerate(train_data):
                data = data.as_in_context(self.model_ctx)
                rating = rating.as_in_context(self.model_ctx)
                with autograd.record():
                    output = self.model(data)
                    #print(output)
                    loss = loss_func(output, rating)
                    #print(loss)
                loss.backward()
                trainer.step(batch_size)
                batch_loss = nd.sum(loss).asscalar()
                batch_avg_loss = batch_loss / data.shape[0]
                cumulative_loss += batch_loss
            #logging.info("Epoch %s / %s, Batch %s / %s. Loss: %s" % (e + 1, epochs, i + 1, batch_count, batch_avg_loss))
            logging.info("Epoch %s / %s. Loss: %s." % (e + 1, epochs, cumulative_loss / num_samples))
            #rmse = mx.metric.RMSE()
            #train_rmse, train_loss = self.basic_evaluator(train_X, train_Y,loss_func,rmse, batch_size=batch_size)
            #mae_train_seq.append(train_rmse)
            if test_data is None:
                logging.info("Epoch %s / %s. Loss: %s." % (e + 1, epochs, cumulative_loss / num_samples))
            elif e%10==0:
                self.topK_evaluator(test_data, recommendedDt)
            loss_train_seq.append(cumulative_loss)

        history['loss_train'] = loss_train_seq
        history['mae_train'] = mae_train_seq
        history['loss_test'] = loss_test_seq
        history['mae_test'] = mae_test_seq

        return history
