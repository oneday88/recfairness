import math
import numpy as np

import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn

"""
The basic probabilistic matrix factorization
"""
class baseCF(nn.Block):

    def __init__(self, embedding_size, user_input_dim, item_input_dim, pos_input_dim, **kwargs):
        super(baseCF, self).__init__(**kwargs)
        with self.name_scope():
            ### The user and item
            self.user_embedding = nn.Embedding(input_dim=user_input_dim, output_dim=embedding_size,
                                                     weight_initializer=mx.init.Uniform(0.05))
            self.item_embedding = nn.Embedding(input_dim=item_input_dim, output_dim=embedding_size,
                                                     weight_initializer=mx.init.Uniform(0.05))
            self.user_bias = nn.Embedding(user_input_dim, 1)
            self.item_bias = nn.Embedding(item_input_dim, 1)
            self.pos_bias = nn.Embedding(pos_input_dim, 1)
            ### The weight for the stats metrics
            self.W = self.params.get('statsW', shape=(2,), init=mx.init.Uniform(0.05))

    def forward(self, x):
        user_vecs = self.user_embedding(x[:, 0])
        item_vecs = self.item_embedding(x[:, 1])
        dot_product = nd.multiply(user_vecs,item_vecs).sum(axis=1)
        
        b_u = self.user_bias(x[:, 0])[:,0]
        b_i = self.item_bias(x[:, 1])[:,0]
        b_pos = self.pos_bias(x[:, 2])[:,0]
        ## The weight of the stats features
        statsWeight = nd.sum(nd.multiply(x[:, 3:5], self.W.data()), axis=1)
        return nd.sigmoid(dot_product+b_u + b_i+b_pos+statsWeight)
