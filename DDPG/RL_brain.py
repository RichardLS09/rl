"""
In python3
DDPG:
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
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32


state_dim=10
action_dim=3
action_bound=5


S=tf.placeholder(tf.float32,[None,state_dim],'S')
S_=tf.placeholder(tf.float32,[None,state_dim],'S_')
R=tf.placeholder(tf.float32,[None,1],'R')

class Actor(object):

    def __init__(self,sess,action_dim,action_bound,learning_rate,replacement):
        self.sess=sess
        self.a_dim=action_dim
        self.a_bound=action_bound
        self.lr=learning_rate
        self.replacement=replacement
        self.t_replace_counter=0

        with tf.variable_scope('Actor'):
            self.a=self._build_net(s,scope='eval_net',trainable=True)
            self.a_=self._build_net(s_,scope='target_net',trainable=False)

        self.e_params=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Actor/eval_net')
        self.t_params=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Actor/target')

        if self.replacement['name']=='hard':
            self.hard_replace=[tf.assign(t,e) for t,e in zip(self.t_params,self.e_params)]
        else:
            self.soft_replace=[tf.assign(t,(1-self.replacement['tau'])*t+self.replacement['tau']*e)
                               for t,e in zip(self.t_params,self.e_params)]

    def _build_net(self,s,scope=None,trainable=True):

        w_initializer,b_initializer=tf.random_normal_initializer(0.0,0.3),tf.constant_initializer(0.1)
        with tf.variable_scope(scope):
            net=tf.layers.dense(s,30,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,trainable=trainable,name='net')
            actions=tf.layers.dense(net,self.a_dim,tf.nn.tanh,kernel_initializer=w_initializer,bias_initializer=b_initializer,trainable=trainable,name='a')
            scaled_a=tf.multiply(actions,self.a_bound,name='scaled_a')
        return scaled_a

    def choose_action(self,s):
        s=s[np.newaxis,:]
        return self.sess.run(self.a,{S:s})[0]

    def learn(self,s):
        self.sess.run(self.opt,{S:s})

        if self.replacement['name']=='hard':
            if self.t_replace_counter%self.replacement['rep_iter_a']==0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter+=1
        else:
            self.sess.run(self.soft_replace)

    def add_grad_to_graph(self,a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads=tf.gradients(self.a,self.e_params,a_grads)

        with tf.variable_scope('train'):
            opt=tf.train.AdamOptimizer(-self.lr)
            self.opt=opt.apply_gradients(zip(self.policy_grads,self.e_params))

class Critic(object):

    def __init__(self,sess,state_dim,action_dim,learning_rate,gamma,replacement,a,a_):
        self.sess=sess
        self.s_dim=state_dim
        self.a_dim=action_dim
        self.lr=learning_rate
        self.gamma=gamma
        self.replacement=replacement


        with tf.variable_scope('Critic'):
            self.a = a
            self.q=self._build_net(S,self.a,'eval_net',trainable=True)
            self.q_=self._build_net(S_,self.a_,'target_net',trainable=False)

        with tf.variable_scope('q_target'):
            self.q_target=R+self.gamma*self.q_

        with tf.variable_scope('td_error'):
            self.loss=tf.reduce_mean(tf.squared_difference(self.q_target,self.q))

        with tf.variable_scope('c_train'):
            self.opt=tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grad=tf.gradients(self.q,self.a)[0]

        if self.replacement['name']=='hard':
            self.hard_replace=[tf.assign(t,e) for t,e in zip(self.t_params,self.e_params)]
        else:
            self.soft_replace=[tf.assign(t,(1-self.replacement['tau'])*t+self.replacement['tau']*e)
                               for t,e in zip(self.t_params,self.e_params)]

    def _build_net(self,s,a,scope=None,trainable=True):
        w_initializer, b_initializer = tf.random_normal_initializer(0.0, 0.3), tf.constant_initializer(0.1)
        with tf.variable_scope(scope):
            n_ll=30
            w1_s=tf.get_variable('w1_s',[self.s_dim,n_ll],initializer=w_initializer,trainable=trainable)
            w1_a=tf.get_variable('w1_a',[self.a_dim,n_ll],initializer=w_initializer,trainable=trainable)
            b=tf.get_variable('b',[1,n_ll],initializer=b_initializer,trainable=trainable)
            net=tf.nn.relu(tf.matmul(s,w1_s)+tf.matmul(a,w1_a)+b)

            q=tf.layers.dense(net,1,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='q')
        return q

    def learn(self,s,a,r,s_):
        self.sess.run(self.opt,{S:s,R:r,self.a:a,S_:s_})

        if self.replacement['name']=='hard':
            if self.t_replace_counter%self.replacement['rep_iter_a']==0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter+=1
        else:
            self.sess.run(self.soft_replace)

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


