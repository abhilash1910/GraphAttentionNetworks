# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 01:42:49 2021

@author: Abhilash
"""

import pandas as pd
import numpy as np
import GraphAttentionNetworks.GraphMultiheadAttention as gat
import GraphAttentionNetworks.GraphAttentionBase as gat_base

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot,plot
import plotly
import plotly.graph_objs as go
import networkx as nx
from pyvis.network import Network
import os
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
os. environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
init_notebook_mode(connected=True)


def test_gat_embeddings():
    print("Testing for GAT embeddings having a source and target label")
    train_df=pd.read_csv("E:\\train_graph\\train.csv")
    source_label='question_body'
    target_label='category'
    print("Input parameters are hidden units , number of layers,number of heads,mode,subset (values of entries to be considered for embeddings),epochs ")
    hidden_units=32
    num_layers=4
    subset=34
    epochs=40
    num_heads=8
    mode='concat'
    gat_emb,gat_graph=gat.get_gat_embeddings(hidden_units,train_df,source_label,target_label,epochs,num_layers,num_heads,mode,subset)
    print(gat_emb.shape)
    return gat_emb,gat_graph

def GAT_Base_helper(adj_dim,feature_dim,n_hidden, num_class, num_layers,num_heads,mode,activation=tf.nn.relu,dropout_rate=0.5, l2_reg=0 ):
    Adj = tf.keras.layers.Input(shape=(feature_dim,), sparse=True,name='first')
    
    X_in = tf.keras.layers.Input(shape=(feature_dim,), sparse=False,name='second')
    emb = tf.keras.layers.Embedding(adj_dim, feature_dim,embeddings_initializer=tf.keras.initializers.Identity(1.0), trainable=False)
    X_emb = emb(X_in)
    H=X_emb
    for i in range(num_layers):
        if i == num_layers - 1:
            activation = tf.nn.softmax
            n_hidden = num_class
        h =gat_base.MultiheadAttentionBase(n_hidden,num_heads,mode, activation=activation, dropout_rate=dropout_rate, l2_reg=l2_reg)([H,Adj])
    output = h
    model = tf.keras.models.Model(inputs=[X_in,Adj], outputs=output)
    return model
def test_gat_base_layer():
    train_df_temp=pd.read_csv("E:\\train_graph\\train.csv")
    source_label='question_body'
    target_label='category'
    print("Input parameters are hidden units , number of layers,mode,number of heads,subset (values of entries to be considered for embeddings),epochs ")
    hidden_units=32
    num_layers=4
    subset=34
    epochs=40
    num_heads=8
    mode='concat'
    
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
    A=gat.preprocess_adj(A)
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

    model = GAT_Base_helper(A.shape[-1],feature_dim, hidden_units, y_train.shape[-1],num_layers,num_heads,mode)
    model.compile(optimizer='adam', loss='categorical_crossentropy',weighted_metrics=[ 'acc'])
    print(model.summary())



if __name__=='__main__':
    test_gat_embeddings()
    test_gat_base_layer()