# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 22:20:30 2021

@author: Abhilash
"""

from distutils.core import setup
setup(
  name = 'GraphAttentionNetworks',         
  packages = ['GraphAttentionNetworks'],   
  version = '0.1',       
  license='MIT',        
  description = 'A Graph Attention Framework for extracting Graph Attention embeddings and implementing Multihead Graph Attention Networks',   
  long_description='This package is used for extracting Graph Attention Embeddings and provides a framework for a Tensorflow Graph Attention Layer which can be used for knowledge graph /node base semantic tasks. It determines the pair wise embedding matrix for a higher order node representation and concatenates them with an attention weight. It then passes it through a leakyrelu activation for importance sampling and damps out negative effect of a node.It then applies a softmax layer for normalization of the attention results and determines the final output scores.The GraphAttentionBase.py script implements a Tensorflow/Keras Layer for the GAT which can be used and the GraphMultiheadAttention.py is used to extract GAT embeddings.',
  author = 'ABHILASH MAJUMDER',
  author_email = 'debabhi1396@gmail.com',
  url = 'https://github.com/abhilash1910/GraphAttentionNetworks',   
  download_url = 'https://github.com/abhilash1910/GraphAttentionNetworks/archive/v_01.tar.gz',    
  keywords = ['Anisotropic Embeddings','Graph Convolution Network','Graph Attention Network','Chebyshev networks','Higher order Graph embeddings','Multihead Graph Attention Framework','Tensorflow'],   
  install_requires=[           

          'numpy',         
          'tensorflow',
          'keras',
          'sklearn',
          'pandas',
          'networkx',
          'scipy',
          'plotly'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',      
    'Programming Language :: Python :: 3.8',

    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
