import time
import numpy as np
import tensorflow as tf
import os
from ops import * #batch normalization, linear, etc
from utils import * #some code for saving images etc.

class ConditionalGANNConfig(object):
    """Define HyperParameters"""
    batch_size = 50
    max_epoch = 30

    #hyperparameter advice
    #https://arxiv.org/pdf/1511.06434.pdf
    #and
    #https://github.com/openai/improved-gan/blob/master/mnist_svhn_cifar10/train_cifar_feature_matching.py

    init_scale = 0.05
    learning_rate = .0003 #suggested .0002 or .001
    lr_decay = 0.5 #suggested .5
    max_grad_norm = 5

    input_dim = 100
    gen_filter_size = 32
    disc_filter_size = 32
    gen_full_size = 1024
    disc_full_size = 1024

    keep_prob = 0.5 #for dropout, Not Implemented


    color = 3
    output_size = 32

    #Checkpoint Directory to save model and images!
    checkpoint_dir = "/checkpoints"

class DCGAN(object):
    #follow structure from first model

    def __init__(self, config, image_size=108,
                 batch_size=32, sample_size = 32,
                 y_dim=None, dataset_name='default',
                 checkpoint_dir=None, sample_dir=None):

        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [32]
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [32]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [32]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        #initialize:
        self.lr = config.learning_rate
        self.lr_decay = config.lr_decay

        self.batch_size = config.batch_size
        self.epoch = config.max_epoch

        self.z_dim = config.input_dim
        self.y_dim = None #some wierd optional attribute from the original DCGAN code
        self.gf_dim = config.gen_filter_size
        self.df_dim = config.disc_filter_size
        self.gfc_dim = config.gen_full_size
        self.dfc_dim = config.disc_full_size
        self.c_dim = config.color
        self.output_size = config.output_size

        # batch normalization : deals with poor initialization helps gradient flow
        #http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        #INITIALIZING GRAPH
        self.g = tf.Graph()
        with self.g.as_default():

            #This initializer is used to initialize all the weights of the network (not done until train() is called)
            #initializer = tf.truncated_normal_initializer(stddev=self.init_scale)
            self.build_model()


        #build model
        #train model
    def build_model(self):

        #PLACEHOLDERS
        #input to discriminator placeholder
        self.images = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size, self.output_size, self.c_dim],
                                    name='real_images')
        #self.sample_images= tf.placeholder(tf.float32, [self.sample_size] + [self.c_dim, self.output_size, self.output_size],
         #                               name='sample_images')

        #input to generator placeholder
        self.z = tf.placeholder(tf.float32, [None, self.z_dim],
                                name='z')

        #Link Discriminator and Generator together
        self.G = self.generator(self.z)
        self.D_real = self.discriminator(self.images)
        self.D_gen = self.discriminator(self.G, reuse=True)
        

        self.d_sum = tf.histogram_summary("d", self.D_real)
        self.d__sum = tf.histogram_summary("d_", self.D_gen)
        self.g_sum = tf.image_summary("G", self.G)

        #Loss Function
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_real, tf.ones_like(self.D_real)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_gen, tf.zeros_like(self.D_gen)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_gen, tf.ones_like(self.D_gen)))

        self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)
                                                    
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_params = [var for var in t_vars if 'd_' in var.name]
        self.g_params = [var for var in t_vars if 'g_' in var.name]
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=self.lr_decay).minimize(self.d_loss, var_list=self.d_params)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=self.lr_decay).minimize(self.g_loss, var_list=self.g_params)
        self.saver = tf.train.Saver()

        self.writer = tf.train.SummaryWriter('summary')

        self.init = tf.initialize_all_variables()
        print("done builiding graph")
        '''
        self.trainerD = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.lr_decay)
        self.trainerG = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.lr_decay)

        #Compute and Apply GRADIENTS
        self.d_grads = self.trainerD.compute_gradients(self.d_loss,self.d_params) #Only update the weights for the discriminator network.
        self.g_grads = self.trainerG.compute_gradients(self.g_loss,self.g_params) #Only update the weights for the generator network.

        self.update_D = self.trainerD.apply_gradients(self.d_grads)
        self.update_G = self.trainerG.apply_gradients(self.g_grads)
        '''
        



    def train(self, data_X, data_Y):
        '''
        Inputs: 
            data_X - a (50000,32,32,3) matrix of images that have been transformed between -1 and 1 (thats 50000 images of 32 by 32 RBG pixels)
            data_Y - labels of those 50000 images
        Raises:
            trains the model. Saving images and models every 1000 iterations
        Returns:
            None
        '''
        print("Training")
        sample_directory = './figs' #Directory to save sample images from generator in.
        model_directory = './models' #Directory to save trained model to.
        image_size = np.array([32,32])

        with tf.Session(graph=self.g) as sess:
            sess.run(self.init)

            counter = 1
            start_time = time.time()

            for epoch in range(self.epoch):

                iterations = (len(data_X)) // self.batch_size

                for idx in range(0, iterations):
                    #Generate/grab inputs
                    batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                    #take in a batch from data
                    batch_images = data_X[idx*self.batch_size:(idx+1)*self.batch_size]

                    # Update D network
                    _, summary_str = sess.run([self.d_optim, self.d_sum],
                        feed_dict={ self.images: batch_images, self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = sess.run([self.g_optim, self.g_sum],
                        feed_dict={ self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _, summary_str = sess.run([self.g_optim, self.g_sum],
                        feed_dict={ self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)
                    
                    errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                    errD_real = self.d_loss_real.eval({self.images: batch_images})
                    errG = self.g_loss.eval({self.z: batch_z})

                    if idx % 1000 == 0:
                        
                        print("Gen Loss: " + str(errG) + " Disc Loss: " + str(errD_real+errD_fake))

                        #get an image
                        newA = sess.run(self.G,feed_dict={ self.z: batch_z })
                        #concatenate with the real images
                        images = np.concatenate((batch_images,newA))
                        #save it
                        if not os.path.exists(sample_directory):
                            os.makedirs(sample_directory)
                            #images,size,path
                        #print("saving images",newA)
                        save_images(images, image_size, sample_directory+'/fig'+str(epoch)+"_"+str(idx))
                        #Save array
                        #should be (40,32,32,3) can use plot cifar module to view (first 20 are images from data)
                        with open(sample_directory+'/txt'+str(epoch)+"_"+str(idx)+".csv",'wb') as f:
                            np.savetxt(f,inverse_transform(images).flatten(),fmt='%.18e')
                        
                    if idx % 1000 == 0:
                        if not os.path.exists(model_directory):
                            os.makedirs(model_directory)
                        self.saver.save(sess,model_directory+'/model-'+str(epoch)+'.cptk')
                        print("Saved Model")

                #Epoch down!

                print("Epoch: "+str(epoch)+" time: "+str(time.time() - start_time))

            self.writer.close()
        return

    #GENERATOR
    def generator(self, z, y=None):
        s = self.output_size
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

        # project `z` and reshape
        self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s16*s16, 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(self.z_, [-1, s16, s16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(self.h0))

        self.h1, self.h1_w, self.h1_b = deconv2d(h0, 
            [self.batch_size, s8, s8, self.gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = deconv2d(h1,
            [self.batch_size, s4, s4, self.gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = deconv2d(h2,
            [self.batch_size, s2, s2, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv2d(h3,
            [self.batch_size, self.output_size, self.output_size, self.c_dim], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)

    #DISCRIMINATOR
    def discriminator(self, image, y=None, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        #ERROR on third CON layer Here
        #h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        #conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        #ValueError: Filter must not be larger than the input: Filter: (5, 5) Input: (4, 4)
        #now throwing an error that it must be defined
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

        return h4

'''
    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s_%s" % ('Cifar100', self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
            '''
