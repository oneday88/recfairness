import os
import _pickle as pickle

import numpy as np
import pandas as pd

from utils import wilson_lower_bound,filterModelDt

"""
Load the data
"""
### Load the data
data_dir_path = '/home/workspace/data'
# The impression data for recommendation testing
impressions = pd.read_csv(data_dir_path + '/informs'+'/impression_data.csv',low_memory=False, na_values = ['NA','NULL','NaN','\\N'], nrows=5000000)
impressions.drop(['detailMlogInfoList'],axis=1,inplace=True)

# The mlogs
mlogStats = pd.read_csv(data_dir_path + '/informs'+'/mlog_stats.csv',low_memory=False, na_values = ['NA','NULL','NaN','\\N'])
mlogDemo = pd.read_csv(data_dir_path + '/informs'+'/mlog_demographics.csv',low_memory=False, na_values = ['NA','NULL','NaN','\\N'])
# The creator 
creatorDemo = pd.read_csv(data_dir_path + '/informs'+'/creator_demographics.csv',low_memory=False, na_values = ['NA','NULL','NaN','\\N'])
mlogDemo['publishDt'] = 30+2-mlogDemo['publishTime']

"""
### Filter and select the modeling dataset
"""
minUserThreshold = 50
minMlogThreshold = 50

impressions, userCount, mlogCount = filterModelDt(impressions,minUserThreshold,minMlogThreshold)
sparsityLevel = float(impressions.shape[0])/(userCount*mlogCount)
print("There are {} records constructed by {} users and {} mlogs \n The sparsity level is {:.3f}".format(impressions.shape[0], userCount, mlogCount, sparsityLevel*100))

"""
Sort the dataset
"""
## Add other dataset for modeling
impressions = pd.merge(impressions, mlogDemo[['mlogId','songId','artistId','creatorId','type','publishDt']], on='mlogId', how='left')
mlogStats = pd.merge(mlogStats, mlogDemo[['mlogId','creatorId','publishDt']], on='mlogId', how='left')

## StatsFeatures
statsFeatures = mlogStats.loc[mlogStats['dt']<16, ].groupby(['creatorId']).agg({'userClickCount': sum, 'userZanCount':sum}).reset_index()
impressions = pd.merge(impressions, statsFeatures, on='creatorId', how='left')

impressions = impressions.sort_values(['dt','userId','impressTime','impressPosition'])
impressions = impressions[~pd.isna(impressions.creatorId)]

impressions = impressions.fillna(0)

"""
The new items
"""
newItems = mlogStats.loc[(mlogStats['publishDt']>=-3) & (mlogStats['publishDt'] <16), ['creatorId','mlogId','userImprssionCount','userClickCount']]
newItems = newItems.groupby(['creatorId','mlogId']).agg({'userImprssionCount': sum, 'userClickCount':sum}).reset_index()

newItems['mlogWilsonScore'] = newItems.apply(lambda x: wilson_lower_bound(x.userClickCount, x.userImprssionCount), axis=1)

newItems.sort_values(by=["mlogWilsonScore"], ascending=True, inplace=True)

### Save the dataset
with open('basicPreprocess.pkl', 'wb') as f:
    pickle.dump([impressions, newItems] , f, -1)
