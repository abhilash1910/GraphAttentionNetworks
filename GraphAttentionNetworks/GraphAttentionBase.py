# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 01:39:37 2021

@author: Abhilash
"""
import tensorflow as tf

class GraphAttentionBase(tf.keras.layers.Layer):
    def __init__(self, units,
                 activation=tf.nn.relu, dropout_rate=0.5,
                 use_bias=True, l2_reg=0, 
                 seed=1024, **kwargs):
        super(GraphAttentionBase, self).__init__(**kwargs)
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
        self.kernel_attention_self=self.add_weight(shape=(self.units,1),
            trainable=True,
            initializer=tf.keras.initializers.glorot_uniform(seed=self.seed),
            regularizer=tf.keras.regularizers.l2(self.l2_reg),name='kernel_Aself'
             )
        self.kernel_attention_cross=self.add_weight(shape=(self.units,1),
            trainable=True,
            initializer=tf.keras.initializers.glorot_uniform(seed=self.seed),
            regularizer=tf.keras.regularizers.l2(self.l2_reg),name='kernel_Across'
             )
        self.flatten=tf.keras.layers.Flatten()
        self.dense=tf.keras.layers.Dense(self.units)

        self.built=True
        
    def call(self,inputs,**kwargs):
        
        #[X,A]
        features, A = inputs
        A=tf.sparse.to_dense(A)
        #[X*W^{T}]
        feature_transform = tf.matmul(features,self.kernel,transpose_b=True)
        #[X_f*units*1]
        self_attention=tf.matmul(feature_transform,self.kernel_attention_self)
        #[X_f*units*1]
        cross_attention=tf.matmul(feature_transform,self.kernel_attention_cross)
        #[X_f*units*1]+#[X_f*units*1].T
        attention_combined=(self_attention+tf.transpose(cross_attention))
        #leaky relu -> 0.2 empirical
        attention_scores=tf.nn.leaky_relu(attention_combined,alpha=0.2)
        #masking
        mask = -10e9 * (1.0 - A)
        #additive
        attention_masked=attention_scores+mask
        #softmax
        softmax_attention=tf.nn.softmax(attention_masked,axis=0)
        #dense [X_f*units*1*X_f]->[N*X_f]
        output=tf.matmul(softmax_attention,feature_transform)
        output=self.flatten(output)
        output=self.dense(output)
        #output=tf.nn.sigmoid(output)
        
        return output
         
class MultiheadAttentionBase(tf.keras.layers.Layer):
    def __init__(self, units,num_heads,aggregation,
                 activation=tf.nn.relu, dropout_rate=0.5,
                 use_bias=True, l2_reg=0, 
                 seed=1024, **kwargs):
        self.units=units
        self.num_heads=num_heads
        self.aggregation=aggregation
        self.activation=activation
        self.attn_layers=[GraphAttentionBase(self.units) for _ in range(self.num_heads)]
        super(MultiheadAttentionBase,self).__init__(**kwargs)
        
    
    def call(self,inputs,**kwargs):
        features, A = inputs
        features_stack=[self.attn_layers[j]([features,A]) for j in range(self.num_heads)]
        if self.aggregation=='concat':
            return self.activation(tf.concat(features_stack,axis=-1))
        return self.activation(tf.reduce_mean(tf.stack(features_stack, axis=-1), axis=-1))
