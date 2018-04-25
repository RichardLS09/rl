"""
In python3
Policy_Gradients:
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

class Policy_Gradients(object):

    def __init__(self,n_actions,n_features,learning_rate=0.001,reward_decay=0.9,output_graph=False):
        self.n_actions=n_actions
        self.n_features=n_features
        self.lr=learning_rate
        self.gamma=reward_decay

        self.ep_obs,self.ep_as,self.ep_rs=[],[],[]

        self._build_net()

        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if output_graph:
            tf.summary.FileWriter('./log',self.sess.graph)

    def _build_net(self):
        w_initializer,b_initializer=tf.random_normal_initializer(0.0,0.3),tf.constant_initializer(0.1)

        with tf.variable_scope('input'):
            self.tf_obs=tf.placeholder(tf.float32,[None,self.n_features],'obs')
            self.tf_acts=tf.placeholder(tf.float32,[None,],'as')
            self.tf_vt=tf.placeholder(tf.float32,[None,],'vt')

        layer=tf.layers.dense(self.tf_obs,10,tf.nn.tanh,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='fc1')

        all_act=tf.layers.dense(layer,self.n_actions,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='all_act')

        self.all_act_prob=tf.nn.softmax(all_act,name='all_act_prob')

        with tf.variable_scope('loss'):
            neg_log_prob=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act,labels=self.tf_acts)
            loss=tf.reduce_mean(neg_log_prob*self.tf_vt)

        with tf.variable_scope('train'):
            self.opt=tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self,s):
        prob_weights=self.sess.run(self.all_act_prob,{self.tf_obs:s[np.newaxis,:]})
        action=np.random.choice(range(prob_weights.shape[1]),p=prob_weights.ravel())
        return action

    def store_transition(self,s,a,r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        discounted_ep_rs_norm=self._discount_and_norm_rewards()

        self.sess.run(self.opt,{self.tf_obs:np.vstack(self.ep_obs),
                                self.tf_acts:np.array(self.ep_as),
                                self.tf_vt:discounted_ep_rs_norm}
                    )
        self.ep_obs,self.ep_as,self.ep_rs=[],[],[]
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        discounted_ep_rs=np.zeros_like(self.ep_rs)

        running_add=0
        for t in reversed(range(0,len(self.ep_rs))):
            running_add=running_add*self.gamma+self.ep_rs[t]
            discounted_ep_rs[t]=running_add

        discounted_ep_rs-=np.mean(discounted_ep_rs)
        discounted_ep_rs/=np.std(discounted_ep_rs)
        return discounted_ep_rs

