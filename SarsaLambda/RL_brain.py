"""
In python3

SarsaLambdaTable:
    to solve low dim statespace and actionspace

env:
    env.reset():return the initial state
    env.render():return update the env after take the action
    env.step():return s_,r,done,info

"""

import numpy as np
import pandas as pd

class SarsaLambdaTable(object):

    def __init__(self,actions,learning_rate=0.001,reward_decay=0.9,e_greedy=0.9,trace_decay=0.9):
        self.actions=actions #a list
        self.lr=learning_rate
        self.gamma=reward_decay
        self.epsilon=e_greedy
        self.q_table=pd.DataFrame(columns=self.actions,dtype=np.float32)

        self.lam=trace_decay
        self.eligibility_table=self.q_table.copy()

    def choose_action(self,s):
        self.check_state_exist(s)

        if np.random.uniform()<self.epsilon:
            actions=self.q_table.loc[s,:]
            actions=actions.reindex(np.random.permutation(actions.index))
            action=actions.idxmax()

        else:
            action=np.random.choice(self.actions)
        return action

    def check_state_exist(self,s):
        if s not in self.q_table.index:
            data=pd.Series([0]*len(self.actions),index=self.actions,name=s)
            self.q_table.append(data)
            self.eligibility_table.append(data)

    def learn(self,s,a,r,s_,a_):
        self.check_state_exist(s_)

        q_predict=self.q_table.loc[s,a]

        if s_!='terminal':
            q_target=r+self.gamma*self.q_table.loc[s_,a_]
        else:
            q_target=r
        error=q_target-q_predict

        # Method 1:
        # self.eligibility_trace.loc[s, a]+=1

        # Method 2:
        self.eligibility_trace.loc[s, :]*=0
        self.eligibility_trace.loc[s, a]=1

        self.q_table+=self.lr*error*self.eligibility_table

        self.eligibility_table*=self.gamma*self.lam
