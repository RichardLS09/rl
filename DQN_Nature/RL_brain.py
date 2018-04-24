"""
In python3
DQN_Nature:
    to solve high dim statespace and low dim actionspace
env:
    env.reset():return the initial state
    env.render():return update the env after take the action
    env.step():return s_,r,done,info
"""

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class DQN_Nature(object):

    def __init__(self,n_actions,n_features,learning_rate=0.001,reward_decay=0.9,
                 e_greedy=0.9,replace_target_iter=200,memory_size=1000,batch_size=32,
                 e_greedy_increment=None,output_graph=False):
        self.n_actions=n_actions
        self.n_features=n_features
        self.lr=learning_rate
        self.gamma=reward_decay
        self.epsilon_max=e_greedy
        self.eilon_increment=e_greedy_increment
        self.epsilon=0 if e_greedy_increment is not None else e_greedyps
        self.replace_target_iter=replace_target_iter
        self.replace_counter=0
        self.memory_size=memory_size
        self.memory_counter=0
        self.batch_size=batch_size
        self.learn_step_counter=0

        self.memory=np.zeros(shape=(self.memory_size,self.n_features*2+2))

        self._build_net()

        e_params=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,'eval_net')
        t_params=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,'target_net')

        with tf.variable_scope('replace_target'):
            self.replace_op=[tf.assign(t,e) for t,e in zip(t_params,e_params)]

        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if output_graph:
            tf.summary.FileWriter('./log',graph=self.sess.graph)

        self.cost_his=[]

    def _build_net(self):

        self.s=tf.placeholder(tf.float32,[None,self.n_features],'s')
        self.s_=tf.placeholder(tf.float32,[None,self.n_features],'s_')
        self.q_target=tf.placeholder(tf.float32,[None,self.n_actions],'q_target')

        w_initializer,b_initializer=tf.random_normal_initializer(0.0,0.3),tf.constant_initializer(0.1)

        with tf.variable_scope('eval_net'):
            layer=tf.layers.dense(self.s,20,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='layer',)
            self.q_eval=tf.layers.dense(layer,self.n_actions,kernel_initializer=w_initializer,bias_initializer=w_initializer,name='q_now')

        with tf.variable_scope('target_net'):
            layer = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='layer',trainable=False)
            self.q_next = tf.layers.dense(layer, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=w_initializer, name='q_next',trainable=False)

        with tf.variable_scope('loss'):
            self.loss=tf.reduce_mean(tf.squared_difference(self.q_eval,self.q_target))

        with tf.variable_scope('train'):
            self.opt=tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self,s,a,r,s_):
        index=self.memory_counter%self.memory_size
        transition=np.hstack((s,[a,r],s_))
        self.memory[index,:]=transition
        self.memory_counter+=1

    def choose_action(self,s):

        if np.random.uniform()<self.epsilon:
            s=s[np.newaxis,:]
            actions=self.sess.run(self.q_eval,feed_dict={self.s:s})
            action=np.argmax(actions,)
        else:
            action=np.random.randint(0,self.n_actions)

        return action

    def learn(self):
        if self.learn_step_counter%self.replace_target_iter==0:
            self.sess.run(self.replace_op)
            print('\ntarget_params_replaced\n')

        idx=self.memory_counter if self.memory_counter<self.memory_size else self.memory_size
        memory=self.memory[idx,:]

        s=memory[:,:self.n_features]
        a=memory[:,self.n_features].astype(int)
        r=memory[:,self.n_features+1]
        s_=memory[:,-self.n_features:]

        q_next,q_eval=self.sess.run([self.q_next,self.q_eval],feed_dict={self.s:s,self.s_:s_})

        q_target=q_eval.copy()

        q_target[:,a]=r+self.gamma*np.max(q_next,axis=1)

        _,loss=self.sess.run([self.opt,self.loss],feed_dict={self.s:s,self.q_target:q_target})

        self.cost_his.append(loss)

        self.epsilon=self.epsilon+self.eilon_increment if self.epsilon<self.epsilon_max else self.epsilon_max
        self.learn_step_counter+=1

    def plot_cost(self):
        from matplotlib import pyplot as plt
        plt.plot(np.arange(len(self.cost_his)),self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('steps')
        plt.show()




