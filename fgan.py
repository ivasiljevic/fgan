#Testing ideas from f-GAN paper: https://arxiv.org/pdf/1606.00709v1.pdf
#GAN setup adapted from https://github.com/peteykun/Simple-GAN

import numpy as np
import tensorflow as tf

batch_size = 1024
learning_rate = 0.01
beta1 = 0.5

def discriminator(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    W1 = tf.get_variable("d_W1",[1,64],initializer=tf.truncated_normal_initializer(stddev=0.05))
    b1 = tf.get_variable("d_b1", [64], initializer=tf.constant_initializer(0.0))
    W2 = tf.get_variable("d_W2",[64,64],initializer=tf.truncated_normal_initializer(stddev=0.05))
    b2 = tf.get_variable("d_b2", [64], initializer=tf.constant_initializer(0.0))
    W3 = tf.get_variable("d_W3",[64,1],initializer=tf.truncated_normal_initializer(stddev=0.05))
    b3 = tf.get_variable("d_b3", [1], initializer=tf.constant_initializer(0.0))

    h1 = tf.nn.tanh(tf.matmul(x,W1) + b1)
    h2 = tf.nn.tanh(tf.matmul(h1,W2) + b2)
    h3 = tf.matmul(h2,W3) + b3

    #return tf.nn.sigmoid(h3), h3
    return h3, h3 #maybe this is KL?

def generator(z):
    W = tf.get_variable("g_W",[1,1],initializer=tf.constant_initializer(1.0)) #1.0
    b = tf.get_variable("g_b", [1], initializer=tf.constant_initializer(0.1))
    g_z = b + tf.matmul(z,W)
    return g_z

real_sample = tf.placeholder(tf.float32, [None,1], name='real_images')
fake_sample = tf.placeholder(tf.float32, [None,1], name='sample_images')
z = tf.placeholder(tf.float32, None, name='z')

G = generator(z)
D, D_logits = discriminator(real_sample)
D_, D_logits_ = discriminator(G, reuse=True)

d_loss_real = -tf.reduce_sum(tf.ones_like(D) * tf.log(D_logits))
#d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logits, tf.ones_like(D)))
#d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logits_, tf.zeros_like(D_)))
d_loss_fake = -tf.reduce_sum(tf.zeros_like(D_) * tf.log(tf.exp(D_logits_-1)))
d_loss = d_loss_real + d_loss_fake
#d_loss = d_loss_real + d_loss_fake

#g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logits_, tf.ones_like(D_)))
g_loss = -tf.reduce_sum(tf.ones_like(D_) * tf.log(tf.exp(D_logits_-1)))

# Optimizers
t_vars = tf.trainable_variables()

d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]
#FIX SCOPE STUFF

#adam seems to find the better mean?
d_optim = tf.train.AdamOptimizer(learning_rate,beta1=beta1).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(learning_rate,beta1=beta1).minimize(g_loss, var_list=g_vars)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()

sample_z = np.random.normal(0, 1, size=batch_size)

counter = 1

w = 0.33
m1 = -1
s1 = 0.0625
m2 = 2
s2 = 2

for epoch in range(10000):
    mix = w*np.random.normal(m1, s1, batch_size) + (1-w)*np.random.normal(m2,s2,batch_size)
    true_dist = np.reshape(mix,[batch_size,1])
    batch_z = np.reshape(np.random.normal(0,1,batch_size),[batch_size,1])
    # Update D network
    sess.run([d_optim], feed_dict={real_sample: true_dist, z: batch_z})
    # Update G network
    #if you run g twice doesnt converge
    sess.run([g_optim], feed_dict={z: batch_z})

    errG = g_loss.eval({z: batch_z}, session=sess)

    counter += 1
    if counter % 100 == 0:
        #d_pred = sess.run(D_, feed_dict={real_sample:true_dist})
        d_pred = sess.run(D_, feed_dict={z:batch_z})
        print(d_pred)
        print("Percent wrong by D: ",np.mean(d_pred<=0.5))
        W = [v for v in tf.all_variables() if v.name == u'g_W:0'][0]
        b = [v for v in tf.all_variables() if v.name == u'g_b:0'][0]
        print("Current sigma = ",W.eval(session=sess)[0])
        print("Current mu    = ",  b.eval(session=sess)[0]) 
        #print("True mu = ", np.mean(mix))
        #print("True sig = ", np.std(mix))
        print
sess.close()
