import math
import numpy as np
import scipy.stats as st
import pandas as pd
from sklearn import preprocessing


"""
Function for data preprocess
"""
def DLPreprocess(dt, cat_feature_list, numeric_feature_list=None):
    ### label encode of categorical features
    label_enc_list = []
    for category_feature in cat_feature_list:
        print(category_feature)
        label_enc = preprocessing.LabelEncoder()
        label_enc.fit(dt.loc[:, category_feature])
        label_enc_list.append(label_enc)
        dt.loc[:, category_feature] = label_enc.transform(dt.loc[:, category_feature])
    ### numeric feature normalization
    if numeric_feature_list is not None:
        dt[numeric_feature_list] = preprocessing.scale(dt[numeric_feature_list])
    return dt,label_enc_list

"""
To filter the dataset
"""
def getCount(df, id):
    playcountGroupbyid = df[[id,'isClick']].groupby(id, as_index=False)
    count = playcountGroupbyid.size()
    return count


def filterModelDt(dt, minUserThreshold=100, minItemThreshold=0):
    userClickCountDt = getCount(dt, 'userId')
    mlogClickCountDt = getCount(dt, 'mlogId')

    dt = dt[dt['userId'].isin(userClickCountDt.loc[userClickCountDt['size']>=minUserThreshold,'userId'].values)]
    dt = dt[dt['mlogId'].isin(mlogClickCountDt.loc[mlogClickCountDt['size']>=minItemThreshold,'mlogId'].values)]

    userCount, mlogCount = len(dt['userId'].unique()), len(dt['mlogId'].unique())
    return dt, userCount, mlogCount

"""
filter the training data
"""
def filterTrainDt(dt, minUserClicks=0, minItemClicks=0):
    dt['userTotalClicks'] = dt.groupby('userId')['isClick'].transform(np.sum)
    dt = dt[dt['userTotalClicks']>minUserClicks]
    dt['itemTotalClicks'] = dt.groupby('mlogId')['isClick'].transform(np.sum)
    dt = dt[dt['itemTotalClicks']>minItemClicks]
    dt.drop(['userTotalClicks','itemTotalClicks'], axis=1, inplace=True)
    return dt

def wilson_lower_bound(pos, n, confidence=0.95):
    """
        Function to provide lower bound of wilson score
        :param pos:  No of positive ratings
        :param n: Total number of rating
        :param confidence, confidence interval, by default is 95%
        :return: Wilson Lowerbound score
    """
    if n==0: return 0
    z = st.norm.ppf(1-(1-confidence)/2)
    phat = 1.0 * pos /n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

def ndcg_with_pos(rank, hit):
    maxRank = np.array(sorted(rank))
    maxHit = sorted(hit, reverse=True)
    maxWeight = np.where(maxRank>1, 1.0/np.log2(maxRank), 1)
    maxGain = maxHit/maxWeight
    return sum(maxGain)


def metrics_at_k(trueDt, predDt, K=100):
    # truncated by topK prediction
    topKPred = predDt.groupby(['userId']).apply(lambda x: x.nlargest(K, ['rating_preds'])).reset_index(drop=True)
    
    # The evaluation dataset
    evalDt = pd.merge(trueDt, topKPred, how='left', on=['userId','mlogId'])
    evalDt['recall_total'] =  evalDt['isClick'].groupby(evalDt['userId']).transform('sum')
    evalDt['precision_total'] = K
    evalDt['hit'] = np.where((evalDt['isClick']==1).values & (evalDt['rating_preds']>0).values, 1,0)

    # The DCG
    evalDt['rankWeight'] = np.where(evalDt['impressPosition']>1, 1/np.log2(evalDt['impressPosition']), 1)
    evalDt['gain'] = evalDt['hit']*evalDt['rankWeight']
    # The DCG max
    maxGainDt = evalDt[['userId','impressPosition','hit']].groupby(['userId']).apply(lambda x: ndcg_with_pos(x.impressPosition, x.hit))

    topKByuserStats = evalDt.groupby(['userId','recall_total','precision_total']).agg({'hit':sum, 'gain': sum}).reset_index()
    topKByuserStats['maxGain'] = maxGainDt.values
    topKByuserStats['recallK'] = 1.0*topKByuserStats['hit']/(topKByuserStats['recall_total']+1)
    topKByuserStats['precisionK'] = 1.0*topKByuserStats['hit']/(topKByuserStats['precision_total']+1)
    topKByuserStats['nDCG'] = topKByuserStats['gain']/topKByuserStats['maxGain']
    topKByuserStats = topKByuserStats.fillna(0)

    hitK = topKByuserStats['hit'].mean()
    precisionK = topKByuserStats['precisionK'].mean()
    recallK = topKByuserStats['recallK'].mean()
    nDCGK = topKByuserStats['nDCG'].mean()

    return hitK, precisionK, recallK, nDCGK
