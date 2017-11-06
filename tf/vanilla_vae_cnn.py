import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
from scipy.misc import imsave


class VAE(object):
    def __init__(self, sample_size, latent_size):
        self.input = tf.placeholder(tf.float32, shape=[None, sample_size])
        #self.z = tf.placeholder(tf.float32, shape=[None, latent_size])
        self.sample_num = tf.placeholder(tf.int32)
        self.sample_size = sample_size
        self.latent_size = latent_size
        self.eh_dim = 128
        self.dh_dim1 = 128
        z_mu, z_var = self.encoder()
        z = self.sample_z(z_mu, z_var)
        with tf.variable_scope('decoder', reuse = False):
            imgs, logits = self.decoder(z)
        self.losses = self.loss(logits, z_mu, z_var)
        self.train_op = self.train(self.losses)
        self.inference_op = self.inference()


    def encoder(self):
        with tf.variable_scope('encoder'):
            net = tf.reshape(self.input, [-1, 28, 28, 1])
            net = layers.conv2d(net, 32, 5, stride=2)
            net = layers.conv2d(net, 64, 5, stride=2)
            net = layers.conv2d(net, 128, 5, stride=2, padding='VALID')
            net = layers.dropout(net, keep_prob=0.9)
            net = layers.flatten(net)
            net = layers.fully_connected(net, 2*self.latent_size, activation_fn=None)
            return (net[:, :self.latent_size], net[:, self.latent_size:])

    def sample_z(self, z_mu, z_var):
        eps = tf.random_normal(shape=tf.shape(z_mu))
        return z_mu + tf.exp(z_var/2) * eps

    def decoder(self, z):
        net = tf.expand_dims(z, 1)
        net = tf.expand_dims(net, 1)
        net = layers.conv2d_transpose(net, 128, 3, padding='VALID')
        net = layers.conv2d_transpose(net, 64, 5, padding='VALID')
        net = layers.conv2d_transpose(net, 32, 5, stride=2)
        net = layers.conv2d_transpose(
        #net, 1, 5, stride=2, activation_fn=tf.nn.sigmoid)
            net, 1, 5, stride=2, activation_fn=None)
        net = layers.flatten(net)
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
            vae = VAE(sample_size, z_dim)
            sess.run(tf.global_variables_initializer())
            for i in range(1000000):
                x_mb, _ = mnist.train.next_batch(mb_size)
                _, loss = sess.run([vae.train_op, vae.losses], feed_dict={vae.input: x_mb})
                if i % 1000 == 0:
                    print('Iteration:%d, losses:%f.' % (i, loss))
                    imgs = sess.run(vae.inference_op, feed_dict={vae.sample_num:1})
                    imsave(os.path.join(imgs_folder, '%d.png') % i, imgs[0].reshape(28, 28))
