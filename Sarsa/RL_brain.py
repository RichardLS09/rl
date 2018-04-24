"""
In python3

SarsaTable:
    to solve low dim statespace and actionspace

env:
    env.reset():return the initial state
    env.render():return update the env after take the action
    env.step():return s_,r,done,info

"""

import numpy as np
import pandas as pd

class SarsaTable(object):

    def __init__(self,actions,learning_rate=0.001,reward_decay=0.9,e_greedy=0.9):
        self.actions=actions #a list
        self.lr=learning_rate
        self.gamma=reward_decay
        self.epsilon=e_greedy
        self.q_table=pd.DataFrame(columns=self.actions,dtype=np.float32)

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
            self.q_table.append(
                pd.Series([0]*len(self.actions),index=self.actions,name=s)
            )

    def learn(self,s,a,r,s_,a_):
        self.check_state_exist(s_)

        q_predict=self.q_table.loc[s,a]

        if s_!='terminal':
            q_target=r+self.gamma*self.q_table.loc[s_,a_]
        else:
            q_target=r
        self.q_table.loc[s,a]+=self.lr*(q_target-q_predict) #update
