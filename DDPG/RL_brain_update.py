"""
In python3
DDPG_uodate:
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
LR_A = 0.001
LR_C = 0.001
TAU = 0.01
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32


state_dim=10
action_dim=3
action_bound=5

class DDPG_update(object):

    def __init__(self,a_dim,s_dim,a_bound):
        self.memory=np.zeros((MEMORY_CAPACITY,2*s_dim+a_dim+1),dtype=np.float32)
        self.pointer=0

        self.sess=tf.Session()

        self.a_dim,self.s_dim,self.a_bound=a_dim,s_dim,a_bound
        self.s=tf.placeholder(tf.float32,[None,s_dim],'s')
        self.s_=tf.placeholder(tf.float32,[None,s_dim],'s_')
        self.r=tf.placeholder(tf.float32,[None,1],'r')

        self.a=self._build_a(self.s,)
        self.q=self._build_c(self.s,self.a)

        a_params=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'Actor')
        c_params=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'Critic')
        ema=tf.train.ExponentialMovingAverage(decay=1-TAU)

        self.target_update=[ema.apply(a_params),ema.apply(c_params)]

        def ema_getter(getter,name,*args,**kwargs):
            return ema.average(getter(name,*args,**kwargs))

        self.a_=self._build_a(self.s_,reuse=True,custom_getter=ema_getter)
        self.q_=self._build_c(self.s_,self.a_,reuse=True,custom_getter=ema_getter)

        self.a_loss=-tf.reduce_mean(self.q)
        self.a_train=tf.train.AdamOptimizer(LR_A).minimize(self.a_loss,var_list=a_params)

        with tf.control_dependencies(self.target_update):
            q_target=self.r+GAMMA*self.q_
            td_error=tf.losses.mean_squared_error(labels=q_target,predictions=self.q)
            self.c_train=tf.train.AdamOptimizer(LR_C).minimize(td_error)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self,s):
        s=s[np.newaxis,:]
        return self.sess.run(self.a,{self.s:s})[0]

    def store_transition(self,s,a,r,s_):
        transition=np.hstack((s,a,[r],s_))
        index=self.pointer%MEMORY_CAPACITY
        self.memory[index,:]=transition
        self.pointer+=1

    def _build_a(self,s,reuse=None,custom_getter=None):
        trainable=True if reuse is None else False
        with tf.variable_scope('Actor',custom_getter=custom_getter,reuse=reuse):
            net=tf.layers.dense(s,30,tf.nn.relu,trainable=trainable)
            a=tf.layers.dense(net,self.a_dim,tf.nn.tanh,trainable=trainable)
        return tf.multiply(a,self.a_bound,name='scaled_a')

    def _build_c(self,s,a,reuse=None,custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic',reuse=reuse,custom_getter=custom_getter):
            n_ll=30
            w1_s=tf.get_variable('w1_s',[self.s_dim,n_ll],trainable=trainable)
            w1_a=tf.get_variable('w1_a',[self.a_dim,n_ll],trainable=trainable)
            b=tf.get_variable('b',[1,n_ll],trainable=trainable)
            net=tf.matmul(s,w1_s)+tf.matmul(a,w1_a)+b
            q=tf.layers.dense(net,1,trainable=trainable)
        return q

    def learn(self):
        indices=np.random.choice(MEMORY_CAPACITY,size=BATCH_SIZE)
        bm=self.memory[indices,:]

        bs=bm[:,:self.s_dim]
        ba=bm[:,self.s_dim:self.s_dim+self.a_dim]
        br=bm[:,self.s_dim+self.a_dim]
        bs_=bm[:,-self.s_dim:]

        self.sess.run(self.a_train,{self.s:bs})
        self.sess.run(self.c_train,{self.s:bs,self.a:ba,self.r:br,self.s_:bs_})








