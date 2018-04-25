"""
In python3
Actor_Critic:
    to solve high dim statespace and high dim actionspace
env:
    env.reset():return the initial state
    env.render():return update the env after take the action
    env.step():return s_,r,done,info
"""

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)
GAMMA=0.9

class Actor(object):

    def __init__(self,sess,n_features,action_bound,learning_rate=0.001):
        self.sess=sess

        self.s=tf.placeholder(tf.float32,[1,n_features],'state')
        self.a=tf.placeholder(tf.float32,None,'act')
        self.td_error=tf.placeholder(tf.float32,None,'tf_error')

        w_initializer,b_initializer=tf.random_normal_initializer(0.0,0.3),tf.constant_initializer(0.1)

        ll=tf.layers.dense(self.s,30,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='ll')

        mu=tf.layers.dense(ll,1,tf.nn.tanh,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='mu')
        sigma=tf.layers.dense(ll,1,tf.nn.softplus,kernel_initializer=w_initializer,b_initializer=b_initializer,name='sigma')

        self.mu,self.sigma=tf.squeeze(mu*2),tf.squeeze(sigma+0.1)
        self.norm_dist=tf.distributions.Normal(self.mu,self.sigma)

        self.action=tf.clip_by_value(self.norm_dist.sample(1),action_bound[0],action_bound[1])

        with tf.variable_scope('exp_v'):
            log_prob=self.norm_dist.log_prob(self.a)
            self.exp_v=log_prob*self.td_error
            self.exp_v+=0.01*self.norm_dist.entropy() # Add cross entropy cost to encourage exploration

        with tf.variable_scope('train'):
            self.opt=tf.train.AdamOptimizer(learning_rate).minimize(-self.exp_v)

    def learn(self,s,a,td):
        s=s[np.newaxis,:]
        feed_dict={self.s:s,self.a:a,self.td_error:td}
        _,exp_v=self.sess.run([self.opt,self.exp_v],feed_dict)
        return exp_v

    def choose_action(self,s):
        return self.sess.run(self.action,{self.s:s[np.newaxis,:]})

class Critic(object):

    def __init__(self,sess,n_features,learning_rate=0.001):
        self.sess=sess

        w_initializer, b_initializer = tf.random_normal_initializer(0.0, 0.3), tf.constant_initializer(0.1)

        with tf.variable_scope('inputs'):
            self.s=tf.placeholder(tf.float32,[1,n_features],'state')
            self.v_=tf.placeholder(tf.float32,[1,1],'v_next')
            self.r=tf.placeholder(tf.float32,name='r')

        with tf.variable_scope('Critic'):
            ll=tf.layers.dense(self.s,30,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='ll')

            self.v=tf.layers.dense(ll,1,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='v')

        with tf.variable_scope('td_error'):
            self.td_error=tf.reduce_mean(self.r+GAMMA*self.v_-self.v)
            self.loss=tf.square(self.td_error)
        with tf.variable_scope('train'):
            self.opt=tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def learn(self,s,r,s_):
        s,s_=s[np.newaxis,:].s_[np.newaxis,:]

        v_=self.sess.run(self.v,{self.s:s_})
        td_error,_=self.sess.run([self.td_error,self.opt],{self.s:s,self.v_:v_,self.r:r})
        return td_error