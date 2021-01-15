import math
import numpy as np

import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn

"""
The neural-based CF
"""
class contentNCF(nn.Block):
    def __init__(self, embedding_size, user_input_dim, item_input_dim, pos_input_dim, **kwargs):
        super(contentNCF, self).__init__(**kwargs)
        with self.name_scope():
            ### The user and item
            self.user_embedding = nn.Embedding(input_dim=user_input_dim, output_dim=embedding_size,
                                                     weight_initializer=mx.init.Uniform(0.05))
         #   self.item_embedding = nn.Embedding(input_dim=item_input_dim[0], output_dim=embedding_size,
         #                                            weight_initializer=mx.init.Uniform(0.05))
            self.pos_bias = nn.Embedding(pos_input_dim, 1)
            self.input_dropout = nn.Dropout(rate=0.5)
            self.creator_embedding = nn.Embedding(input_dim=item_input_dim[1], output_dim=10, weight_initializer=mx.init.Uniform(0.05))
            self.song_embedding = nn.Embedding(input_dim=item_input_dim[2], output_dim=10, weight_initializer=mx.init.Uniform(0.05))
            self.artist_embedding = nn.Embedding(input_dim=item_input_dim[3], output_dim=10, weight_initializer=mx.init.Uniform(0.05))
            self.dense0 = nn.Dense(units=64, activation='relu')
            self.dense1 = nn.Dense(units=1)

    def forward(self, x):
        user_vecs = self.user_embedding(x[:, 0])
        #item_vecs = self.item_embedding(x[:, 1])

        creator_vecs = self.creator_embedding(x[:, 2])
        song_vecs = self.song_embedding(x[:, 3])
        artist_vecs = self.artist_embedding(x[:, 4])
        content_vecs = nd.concat(creator_vecs,song_vecs,artist_vecs)

        b_pos = self.pos_bias(x[:, 5])
        ## Concat all features
        input_vecs = nd.concat(user_vecs, b_pos, creator_vecs,song_vecs,artist_vecs)
        iput_cecs = self.input_dropout(input_vecs)
        dense0_output = self.dense0(input_vecs)
        return self.dense1(dense0_output)[:,0]


class baseNCF(nn.Block):
    def __init__(self, embedding_size, user_input_dim, item_input_dim, pos_input_dim,  **kwargs):
        super(baseNCF, self).__init__(**kwargs)
        with self.name_scope():
            # initializer: random with minval = -0.05, maxval = 0.05
            self.user_embedding = nn.Embedding(input_dim=user_input_dim, output_dim=embedding_size,
                                                     weight_initializer=mx.init.Uniform(0.05))
            self.item_embedding = nn.Embedding(input_dim=item_input_dim[0], output_dim=embedding_size,
                                                     weight_initializer=mx.init.Uniform(0.05))
            self.pos_bias = nn.Embedding(pos_input_dim, 1)
            self.input_dropout = nn.Dropout(rate=0.5)
            self.dense0 = nn.Dense(units=64, activation='relu')
            self.dense1 = nn.Dense(units=1)

    def forward(self, x):
        user_vecs = self.user_embedding(x[:, 0])
        item_vecs = self.item_embedding(x[:, 1])
        b_pos = self.pos_bias(x[:, 5])
        input_vecs = self.input_dropout(nd.concat(user_vecs, item_vecs,b_pos, x[:,6:8]))
        dense0_output = self.dense0(input_vecs)
        return self.dense1(dense0_output)[:,0]
