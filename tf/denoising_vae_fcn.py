import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
from scipy.misc import imsave


class DenoisingVAE(object):
    def __init__(self, sample_size, latent_size):
        self.input = tf.placeholder(tf.float32, shape=[None, sample_size])
        #self.z = tf.placeholder(tf.float32, shape=[None, latent_size])
        self.sample_num = tf.placeholder(tf.int32)
        self.sample_size = sample_size
        self.latent_size = latent_size
        self.eh_dim = 128
        self.dh_dim = 128
        self.noise_factor = 0.05
        z_mu, z_var = self.encoder()
        z = self.sample_z(z_mu, z_var)
        with tf.variable_scope('decoder', reuse = False):
            imgs, logits = self.decoder(z)
        self.losses = self.loss(logits, z_mu, z_var)
        self.train_op = self.train(self.losses)
        self.inference_op = self.inference()


    def encoder(self):
        with tf.variable_scope('encoder'):
            w1 = tf.get_variable('encoder_w1', shape=[self.sample_size, self.eh_dim], initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.zeros(shape=[self.eh_dim]), name = 'encoder_w1')
            w2_mu = tf.get_variable('encoder_w2_mu', shape=[self.eh_dim, self.latent_size], initializer=tf.contrib.layers.xavier_initializer())
            b2_mu = tf.get_variable('encoder_b2_mu', shape=[self.latent_size], initializer=tf.constant_initializer(0))
            w2_sigma = tf.get_variable('encoder_w2_sigma', shape=[self.eh_dim, self.latent_size], initializer=tf.contrib.layers.xavier_initializer())
            b2_sigma = tf.get_variable('encoder_b2_sigma', shape=[self.latent_size], initializer=tf.constant_initializer(0))
            # add noise
            input_noise = self.input + self.noise_factor * tf.random_normal(tf.shape(self.input))
            input_noise = tf.clip_by_value(input_noise, 0., 1.)
            net = tf.nn.xw_plus_b(input_noise, w1, b1, name='encoder_layer1')
            net = tf.nn.relu(net)
            z_mu = tf.nn.xw_plus_b(net, w2_mu, b2_mu, name='encoder_mean')
            z_logvar =  tf.nn.xw_plus_b(net, w2_sigma, b2_sigma, name='encoder_var')
            return (z_mu, z_logvar)

    def sample_z(self, z_mu, z_var):
        eps = tf.random_normal(shape=tf.shape(z_mu))
        return z_mu + tf.exp(z_var/2) * eps

    def decoder(self, z):
        w1 = tf.get_variable('decoder_w1', shape=[self.latent_size, self.dh_dim], initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable('decoder_b1', shape=[self.dh_dim], initializer=tf.constant_initializer(0))
        w2 = tf.get_variable('decoder_w2', shape=[self.dh_dim, self.sample_size], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('decoder_b2', shape=[self.sample_size], initializer=tf.constant_initializer(0))
        net = tf.nn.xw_plus_b(z, w1, b1, name='decoder_layer1')
        net = tf.nn.relu(net)
        net = tf.nn.xw_plus_b(net, w2, b2, name='decoder_layer2')
        logits = net
        net = tf.nn.sigmoid(net)
        return net, logits

    def loss(self, logits, z_mu, z_var):
        recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.input), 1)
        kl_loss = 0.5*tf.reduce_sum(tf.exp(z_var) + z_mu**2 -1. - z_var, 1)
        vae_loss = tf.reduce_mean(recon_loss + kl_loss)
        return vae_loss

    def train(self, vae_loss):
        train_op = tf.train.AdamOptimizer().minimize(vae_loss)
        return train_op

    def inference(self):
        with tf.variable_scope('decoder', reuse = True):
            z = tf.random_normal(shape=[self.sample_num, self.latent_size])
            _, imgs = self.decoder(z)
            return imgs


if __name__ == '__main__':
    mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
    mb_size = 64
    sample_size = 784
    z_dim = 100

    imgs_folder = './out'
    if not os.path.exists(imgs_folder):
        os.makedirs(imgs_folder)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default(), tf.device('/gpu:1'):
            dvae = DenoisingVAE(sample_size, z_dim)
            sess.run(tf.global_variables_initializer())
            for i in range(1000000):
                x_mb, _ = mnist.train.next_batch(mb_size)
                _, loss = sess.run([dvae.train_op, dvae.losses], feed_dict={dvae.input: x_mb})
                if i % 1000 == 0:
                    print('Iteration:%d, losses:%f.' % (i, loss))
                    imgs = sess.run(dvae.inference_op, feed_dict={dvae.sample_num:1})
                    imsave(os.path.join(imgs_folder, '%d.png') % i, imgs[0].reshape(28, 28))
