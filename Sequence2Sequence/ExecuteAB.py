#Execution of AnnoyingBrotherModel
from AnnoyingBrotherModel import ABModel, AnnoyingBrotherConfig
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

'''
ExecuteAB imports generated data set and models defined in AnnoyingBrotherModel.py

Data are a set of "questions" which are randoml integers from 0 to 500

"Real Answers" are determined by a matrix operation on these questions
*Tensorflow requires input as a tensor 
(a Tensorflow variable) of the dimensions [batch_size, sequence_length, input_dimension] 

The goal is to have the Generator learn this answer key
'''

#Read Data
n = 1000
Q = np.genfromtxt('ab_data.csv',dtype='int32',delimiter=',')
Q = np.reshape(Q,(n,5)) #[ [1,2,3,4,5],
						   #[1,2,3,4,5], ... ]
#print(Q)
print("config")
config = AnnoyingBrotherConfig()
print("model")
model = ABModel(config) 
model.train(Q)