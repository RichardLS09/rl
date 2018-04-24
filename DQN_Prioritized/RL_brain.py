"""
In python3
DQN_Prioritized:
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

class SumTree(object):

    data_pointer=0

    def __init__(self,capacity):
        self.capacity=capacity
        self.tree=np.zeros(2*self.capacity-1)
        self.data=np.zeros(self.capacity,dtype=object)

    def add(self,p,data):
        self.data[self.data_pointer]=data
        tree_idx=self.data_pointer+self.capacity-1
        self.update(tree_idx,p)

        self.data_pointer+=1
        if self.data_pointer>=self.capacity:
            self.data_pointer=0

    def update(self,tree_idx,p):
        change=p-self.tree[tree_idx]
        self.tree[tree_idx]=p

        while tree_idx!=0:
            tree_idx=(tree_idx-1)//2
            self.tree[tree_idx]+=change

    def get_leaf(self,v):


        parent_idx=0
        while True:
            cl_idx=2*parent_idx+1
            cr_idx=cl_idx+1

            if cl_idx>=len(self.tree):
                leaf_idx=parent_idx
                break
            else:
                if v<=self.tree[cl_idx]:
                    parent_idx=cl_idx
                else:
                    v-=self.tree[cl_idx]
                    parent_idx=cr_idx
        data_idx=leaf_idx+1-self.capacity
        return leaf_idx,self.tree[leaf_idx],self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]

class Memory(object):

    epsilon=0.01
    alpha=0.6
    beta=0.4
    beta_increment_per_sampling=0.001
    abs_err_upper=1.0

    def __init__(self,capacity):
        self.tree=SumTree(capacity)

    def store(self,transition):
        max_p=np.max(self.tree.tree[-self.tree.capacity:])
        if max_p==0:
            max_p=self.abs_err_upper
        self.tree.add(max_p,transition)

    def sample(self,n):

        b_idx,b_memory,ISWeights=np.empty([n,],dtype=np.int32),np.empty([n,self.tree.data[0].shape]),np.empty([n,1])
        pri_seg=self.tree.total_p/n
        self.beta=np.min([1.0,self.beta+self.beta_increment_per_sampling])

        min_prob=np.min(self.tree.tree[-self.tree.capacity:])/self.tree.total_p

        for i in range(n):
            a,b=pri_seg*i,pri_seg*(i+1)
            v=np.random.uniform(a,b)
            idx,p,data=self.tree.get_leaf(v)

            prob=p/self.tree.total_p
            ISWeights[i,0]=np.power(prob/min_prob,-self.beta)
            b_idx[i],b_memory[i,:]=idx,data
        return b_idx,b_memory,ISWeights

    def batch_update(self,tree_idx,abs_error):
        abs_error+=self.epsilon
        clipped_errors=np.minimum(abs_error,self.abs_err_upper)
        ps=np.power(clipped_errors,self.alpha)
        for ti,p in zip(tree_idx,ps):
            self.tree.update(ti,p)

class DQN_Prioritized(object):

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
        self.batch_size=batch_size
        self.learn_step_counter=0

        self.memory=Memory(memory_size)

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
        self.ISWeights=tf.placeholder(tf.float32,[None,1],'isweights')

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
            self.abs_error=tf.reduce_sum(tf.abs(self.q_target-self.q_eval),axis=1)
            self.loss=tf.reduce_mean(self.ISWeights*tf.squared_difference(self.q_eval,self.q_target))

        with tf.variable_scope('train'):
            self.opt=tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store(self,s,a,r,s_):
        transition=np.hstack((s,[a,r],s_))
        self.memory.store(transition)

    def choose_action(self,s):
        if np.random.uniform()<self.epsilon:
            s=s[np.newaxis,:]
            actions=self.sess.run(self.q_eval,{self.s:s})
            action=np.argmax(actions)
        else:
            action=np.random.randint(0,self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter%self.replace_target_iter==0:
            self.sess.run(self.replace_op)
            print('\ntarget_params_replaced\n')
        tree_idx,batch_memory,ISWeights=self.memory.sample(self.batch_size)

        s=batch_memory[:,:self.n_features]
        a=batch_memory[:,self.n_features].astype(int)
        r=batch_memory[:,self.n_features+1]
        s_=batch_memory[:,-self.n_features:]

        q_next,q_eval=self.sess.run([self.q_next,self.q_eval],{self.s:s,self.s_:s_})
        q_target=q_eval.copy()
        q_target[:,a]=r+self.gamma*np.max(q_next,axis=1)

        _,abs_errors,loss=self.sess.run([self.opt,self.abs_error,self.loss],
                                       {self.s:s,self.q_target:q_target,self.ISWeights:ISWeights})
        self.memory.batch_update(tree_idx,abs_errors)

        self.cost_his.append(loss)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

