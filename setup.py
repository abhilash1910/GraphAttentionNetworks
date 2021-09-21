# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 22:20:30 2021

@author: Abhilash
"""

from distutils.core import setup
setup(
  name = 'GraphAttentionNetworks',         
  packages = ['GraphAttentionNetworks'],   
  version = '0.2',       
  license='MIT',        
  description = 'A Graph Attention Framework for extracting Graph Attention embeddings and implementing Multihead Graph Attention Networks',   
  long_description='',
  author = 'ABHILASH MAJUMDER',
  author_email = 'debabhi1396@gmail.com',
  url = 'https://github.com/abhilash1910/GraphAttentionNetworks',   
  download_url = 'https://github.com/abhilash1910/GraphAttentionNetworks/archive/v_02.tar.gz',    
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
