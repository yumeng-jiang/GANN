# coding=utf-8
from cifar_load import *#load_cifar( ) returns data,label
from utils import * 
from conditional_GANN_model import ConditionalGANNConfig, DCGAN
import time
import numpy as np
import tensorflow as tf

print("loading data")
trainx, trainy, namesy = load_cifar()
#reshape to 50000 images, 32x32 pixels, RBG
trainx = trainx.reshape(50000,32,32,3)
#standardize
trainx = transform(trainx)

print("load config")
config = ConditionalGANNConfig()
print("build model")
model = DCGAN(config)
print("train")
model.train(trainx,trainy) 
