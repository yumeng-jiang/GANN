from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import os
import scipy.misc
import scipy
import tensorflow as tf 
import MNIST_data
from MNIST_utils import *

'''
The noisey Word to Vector GAN Model

In order to generalize to the word feature space,
We add a small random uniform (-.1,.1) to the embedding vectors

This should allow the Generator to learn what similar words might look like
'''

WORK_DIRECTORY = 'data'
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
learning_rate = 0.0002
decay = 0.5

batch_size = 40
iterations = 500000 #Total number of iterations to use.

print("loading data")
mnist = MNIST_data.read_data_sets("MNIST_data/", one_hot=False)



#define Generator with input z AND y
def generator(z, y):
    '''
    Generator is like a reverse convolutional Neural Network
    Taking in the input vector and projecting it into a large 'meaning' vector
    
    see https://arxiv.org/pdf/1511.06434.pdf
    '''

    #throw both z and y through feedforward before convolution
    zy = tf.concat(1, [z,y])

    zP = slim.fully_connected(zy,4*4*256,normalizer_fn=slim.batch_norm,\
        activation_fn=tf.nn.relu,scope='g_project',weights_initializer=initializer)
    zCon = tf.reshape(zP,[-1,4,4,256])
    
    gen1 = slim.convolution2d_transpose(\
        zCon,num_outputs=64,kernel_size=[5,5],stride=[2,2],\
        padding="SAME",normalizer_fn=slim.batch_norm,\
        activation_fn=tf.nn.relu,scope='g_conv1', weights_initializer=initializer)
    
    gen2 = slim.convolution2d_transpose(\
        gen1,num_outputs=32,kernel_size=[5,5],stride=[2,2],\
        padding="SAME",normalizer_fn=slim.batch_norm,\
        activation_fn=tf.nn.relu,scope='g_conv2', weights_initializer=initializer)
    
    gen3 = slim.convolution2d_transpose(\
        gen2,num_outputs=16,kernel_size=[5,5],stride=[2,2],\
        padding="SAME",normalizer_fn=slim.batch_norm,\
        activation_fn=tf.nn.relu,scope='g_conv3', weights_initializer=initializer)
    
    g_out = slim.convolution2d_transpose(\
        gen3,num_outputs=1,kernel_size=[32,32],padding="SAME",\
        biases_initializer=None,activation_fn=tf.nn.tanh,\
        scope='g_out', weights_initializer=initializer)
    
    return g_out

#define Conditional Discriminator
def discriminator(bottom, y, reuse=False):
    
    dis1 = slim.convolution2d(bottom,16,[4,4],stride=[2,2],padding="SAME",\
        biases_initializer=None,activation_fn=lrelu,\
        reuse=reuse,scope='d_conv1',weights_initializer=initializer)
    
    dis2 = slim.convolution2d(dis1,32,[4,4],stride=[2,2],padding="SAME",\
        normalizer_fn=slim.batch_norm,activation_fn=lrelu,\
        reuse=reuse,scope='d_conv2', weights_initializer=initializer)
    
    dis3 = slim.convolution2d(dis2,64,[4,4],stride=[2,2],padding="SAME",\
        normalizer_fn=slim.batch_norm,activation_fn=lrelu,\
        reuse=reuse,scope='d_conv3',weights_initializer=initializer)

    #flatten the last convolution and append the conditional input!
    d_project = slim.fully_connected(tf.concat(1, [slim.flatten(dis3),y]),64,normalizer_fn=slim.batch_norm,\
    activation_fn=tf.nn.relu,reuse = reuse, scope='d_project',weights_initializer=initializer)

    #output probability
    d_out = slim.fully_connected(d_project,1,activation_fn=tf.nn.sigmoid,\
        reuse=reuse,scope='d_out', weights_initializer=initializer)
    
    return d_out

#BUILD GRAPH
print("building graph")
tf.reset_default_graph()

z_size = 100 #Size of z vector used for generator. same as batch so that we can grab labels

#This initializaer is used to initialize all the weights of the network.
initializer = tf.truncated_normal_initializer(stddev=0.02)

#These two placeholders are used for input into the generator and discriminator, respectively.
z_in = tf.placeholder(shape=[None,z_size],dtype=tf.float32) #Random vector
y_in = tf.placeholder(shape=[None,z_size],dtype=tf.float32) #labels as word vectors

real_in = tf.placeholder(shape=[None,32,32,1],dtype=tf.float32) #Real images

Gz = generator(z_in, y_in) #Generates images from random z vectors
Dx = discriminator(real_in, y_in) #Produces probabilities for real images
Dg = discriminator(Gz, y_in, reuse=True) #Produces probabilities for generator images

#These functions together define the optimization objective of the GAN.
d_loss = -tf.reduce_mean(tf.log(Dx) + tf.log(1.-Dg)) #This optimizes the discriminator.
g_loss = -tf.reduce_mean(tf.log(Dg)) #This optimizes the generator.

tvars = tf.trainable_variables()

#The below code is responsible for applying gradient descent to update the GAN.
trainerD = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=decay) 
trainerG = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=decay)
d_grads = trainerD.compute_gradients(d_loss,tvars[9:]) #Only update the weights for the discriminator network.  After the first 5 layers (5*2 = 10)
g_grads = trainerG.compute_gradients(g_loss,tvars[0:9]) #Only update the weights for the generator network.  This is the first 10 sets of weights and biases

update_D = trainerD.apply_gradients(d_grads)
update_G = trainerG.apply_gradients(g_grads)

print("training")
sample_directory = './noiseyw2vmnistfigs' #Directory to save sample images from generator in.
model_directory = './noiseyw2vmnistmodels' #Directory to save trained model to.
#to read embeddings.  Note that words zero to nine are indexed 0:9
embeddings = np.genfromtxt('word2vec.csv')

init = tf.initialize_all_variables()
saver = tf.train.Saver()
with tf.Session() as sess:  
    sess.run(init)
    for i in range(iterations):
        zs = np.random.uniform(-1.0,1.0,size=[batch_size,z_size]).astype(np.float32) #Generate a random z batch
        xs,labels = mnist.train.next_batch(batch_size) #Draw a sample batch from MNIST dataset.
        #train on noisey word vectors
        noise = np.random.uniform(-0.1,0.1,size=[batch_size,z_size]).astype(np.float32)
        ys = word2vec(labels,embeddings) + noise #adding uniform(-.1,.1)

        #reshaping images
        xs = (np.reshape(xs,[batch_size,28,28,1]) - 0.5) * 2.0 #Transform it to be between -1 and 1
        xs = np.lib.pad(xs, ((0,0),(2,2),(2,2),(0,0)),'constant', constant_values=(-1, -1)) #Pad the images so the are 32x32

        _,dLoss = sess.run([update_D,d_loss],feed_dict={z_in:zs, y_in:ys, real_in:xs}) #Update the discriminator
        _,gLoss = sess.run([update_G,g_loss],feed_dict={z_in:zs, y_in:ys}) #Update the generator, twice for good measure.
        _,gLoss = sess.run([update_G,g_loss],feed_dict={z_in:zs, y_in:ys})

        #save some images
        if i % 1000 == 0:
            print("iteration "+str(i))
            print ("Gen Loss: " + str(gLoss) + " Disc Loss: " + str(dLoss))
            print(labels[0:36])
            #z2 = np.random.uniform(-1.0,1.0,size=[batch_size,z_size]).astype(np.float32) #Generate another z batch
            newZ = sess.run(Gz,feed_dict={z_in:zs, y_in:ys}) #Use new z to get sample images from generator.
            if not os.path.exists(sample_directory):
                os.makedirs(sample_directory)
            #Save sample generator images for viewing training progress.
            save_images(np.reshape(newZ[0:36],[36,32,32]),[8,8],sample_directory+'/fig'+str(i)+'.png') #just saving the first 36 of them
        
        #save model
        if i % 1000 == 0:
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)
            #saver.save(sess,model_directory+'/model-'+str(i)+'.cptk')
            saver.save(sess, sample_directory+'/my-model') #apparently a new update to tensorflow.  this method will save a "my-model.meta" file
            print ("Saved Model")   
