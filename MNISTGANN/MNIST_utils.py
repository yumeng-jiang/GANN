import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import scipy.misc

'''
Utility functions for all MNIST GAN models

some are written by me (Austin Slakey), others by carpdem20's 
implementation https://github.com/carpedm20/DCGAN-tensorflow
'''

#This function performns a leaky relu activation, which is needed for the discriminator network.
def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)
    
#The below functions are taken from carpdem20's implementation https://github.com/carpedm20/DCGAN-tensorflow
#They allow for saving sample images from the generator to follow progress
def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def inverse_transform(images):
    return (images+1.)/2.

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image

    return img
def save_single(image, path):
    im = inverse_transform(image)
    return scipy.misc.imsave(path,im)

def word2vec(labels,embeddings):
    vectors = np.empty((len(labels),100))
    for idx in range(len(labels)):
        vectors[idx] = embeddings[labels[idx]]
    return vectors

def int_to_words(labels):
    words = []
    lookup = np.array(['zero','one','two','three','four','five','six','seven','eight','nine'])
    for idx in range(len(labels)):
        words.append(lookup[labels[idx]])
    return np.asarray(words)


