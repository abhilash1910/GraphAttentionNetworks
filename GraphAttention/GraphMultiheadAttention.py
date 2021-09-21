# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 01:30:37 2021

@author: Abhilash
"""
import tensorflow as tf
from tensorflow.keras.initializers import Identity, glorot_uniform, Zeros
from tensorflow.keras.layers import Dropout, Input, Layer, Embedding, Reshape,LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import networkx as nx
import scipy
from sklearn.preprocessing import LabelEncoder
import logging
import numpy as np
import pandas as pd

class GraphAttention(tf.keras.layers.Layer):
    def __init__(self, units,
                 activation=tf.nn.relu, dropout_rate=0.5,
                 use_bias=True, l2_reg=0, 
                 seed=1024, **kwargs):
        super(GraphAttention, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.activation = tf.keras.layers.Activation(tf.keras.activations.relu)
        self.seed = seed
        
        
    def build(self,input_shapes):
        input_dim = int(input_shapes[0][-1])
        # [units* input_dims]
        self.kernel = self.add_weight(shape=(self.units,input_dim),
                                      initializer=tf.keras.initializers.glorot_uniform(
                                          seed=self.seed),
                                      regularizer=tf.keras.regularizers.l2(self.l2_reg),
                                      name='kernel_W' )
        
        #[units*1] -> self attention [units*1]->cross attention
        self.kernel_attention_self=self.add_weight(shape=(input_dim,self.units),
            trainable=True,
            initializer=tf.keras.initializers.glorot_uniform(seed=self.seed),
            regularizer=tf.keras.regularizers.l2(self.l2_reg),name='kernel_Aself'
             )
        self.kernel_attention_cross=self.add_weight(shape=(input_dim,self.units),
            trainable=True,
            initializer=tf.keras.initializers.glorot_uniform(seed=self.seed),
            regularizer=tf.keras.regularizers.l2(self.l2_reg),name='kernel_Across'
             )
        self.bias = self.add_weight(shape=(self.units,),
                                        initializer=tf.keras.initializers.Zeros(),
                                        name='bias')

        
        self.built=True
        
    def call(self,inputs,**kwargs):
        
        #[X,A]
        features, A = inputs
        A=tf.sparse.to_dense(A)
        #[X*W^{T}]
        #feature_transform = tf.matmul(features,self.kernel,transpose_b=True)
        #[X_f*units*1]
        feature_transform=A
        self_attention=tf.matmul(feature_transform,self.kernel_attention_self)
        #[X_f*units*1]
        cross_attention=tf.matmul(feature_transform,self.kernel_attention_cross)
        #[X_f*units*1]+#[X_f*units*1].T
        attention_combined=(self_attention+ cross_attention)
        #leaky relu -> 0.2 empirical
        attention_scores=tf.nn.leaky_relu(attention_combined,alpha=0.2)
        #print('l',attention_scores.shape)
        #masking
        #mask = -10e9 * (1.0 - A)
        #additive
        #print('h',mask.shape)
        attention_masked=attention_scores
        #softmax
        softmax_attention=tf.nn.softmax(attention_masked,axis=0)
        #dense [X_f*units*1*X_f]->[N*X_f]
        #output=self.flatten(output)
        #output=self.dense(output)
        #output=tf.nn.sigmoid(output)
        #output += self.bias
        output=softmax_attention
        return output
         
class MultiheadAttention(tf.keras.layers.Layer):
    def __init__(self, units,num_heads,aggregation,
                 activation=tf.nn.relu, dropout_rate=0.5,
                 use_bias=True, l2_reg=0, 
                 seed=1024, **kwargs):
        self.units=units
        self.num_heads=num_heads
        self.aggregation=aggregation
        self.activation=activation
        self.attn_layers=[GraphAttention(self.units) for _ in range(self.num_heads)]
        super(MultiheadAttention,self).__init__(**kwargs)
        
    
    def call(self,inputs,**kwargs):
        features, A = inputs
        features_stack=[self.attn_layers[j]([features,A]) for j in range(self.num_heads)]
        #if self.aggregation=='concat':
        #    return self.activation(tf.concat(features_stack,axis=-1))
        return self.activation(tf.reduce_mean(tf.stack(features_stack, axis=-1), axis=-1))

def GAT(adj_dim,feature_dim,n_hidden, num_class, num_layers,num_heads,mode,activation=tf.nn.relu,dropout_rate=0.5, l2_reg=0 ):
    Adj = Input(shape=(feature_dim,), sparse=True,name='first')
    
    X_in = Input(shape=(feature_dim,), sparse=False,name='second')
    emb = Embedding(adj_dim, feature_dim,embeddings_initializer=Identity(1.0), trainable=False)
    X_emb = emb(X_in)
    H=X_emb
    for i in range(num_layers):
        if i == num_layers - 1:
            activation = tf.nn.softmax
            n_hidden = num_class
        h =MultiheadAttention(n_hidden,num_heads,mode, activation=activation, dropout_rate=dropout_rate, l2_reg=l2_reg)([H,Adj])
    output = h
    model = Model(inputs=[X_in,Adj], outputs=output)
    return model
    
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = scipy.sparse.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = scipy.sparse.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + scipy.sparse.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    print('adj',adj.shape)
    return adj

def get_gat_embeddings(hidden_units,train_df_temp,source_label,target_label,epochs,num_layers,num_heads,mode,subset):
    label_set=[]
    if(subset<train_df_temp.index.size):
        train_df=train_df_temp[:subset]
        graph=nx.from_pandas_edgelist(train_df,source=source_label,target=target_label)
        if(graph.number_of_nodes()>subset ):
            label_set=train_df_temp[target_label][:graph.number_of_nodes()].tolist()
        else:
            label_set=train_df[target_label][:graph.number_of_nodes()].tolist()
    else:
        graph=nx.from_pandas_edgelist(train_df_temp[:],source=source_label,target=target_label)
        if(graph.number_of_nodes()>subset ):
            temp_list=train_df_temp[target_label][:].tolist()
            for i in range(graph.number_of_nodes()-subset):
                label_set.append(temp_list[-1])
        else:
            label_set=train_df_temp[target_label][:graph.number_of_nodes()].tolist()
    A=nx.adjacency_matrix(graph,nodelist=range(graph.number_of_nodes()))
    A=preprocess_adj(A)
    #print(f"Created Laplacian {A}")
    label_y= LabelEncoder()
    
    labels=label_y.fit_transform(label_set)
    #y_train=encode_onehot(labels)
    y_train = tf.keras.utils.to_categorical(labels)
    print('shape of target',y_train.shape)
    
    feature_dim = A.shape[-1]
    X = np.arange(A.shape[-1])
    X_n=[]
    for i in range(feature_dim):
        X_n.append(X)
    X=np.asarray(X_n)
    model_input = [X, A]

    model = GAT(A.shape[-1],feature_dim, hidden_units, y_train.shape[-1],num_layers,num_heads,mode,  dropout_rate=0.5, l2_reg=2.5e-4 )
    model.compile(optimizer='adam', loss='categorical_crossentropy',weighted_metrics=[ 'acc'])
    print(model.summary())
    print("Fitting model with {hidden_units} units")
    model.fit([X,A],y_train,epochs=epochs)
    
    embedding_weights = model.predict(model_input)
    print(f"Dimensions of embeddings {embedding_weights.shape}")
    
    print(embedding_weights)
    return embedding_weights,graph