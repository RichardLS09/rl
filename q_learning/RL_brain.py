"""
In python3

QLearningTable:
    to solve low dim statespace and actionspace

env:
    env.reset():return the initial state
    env.render():return update the env after take the action
    env.step():return s_,r,done,info

"""


import numpy as np
import pandas as pd

class QLearningTable(object):

    def __init__(self,actions,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9):
        self.actions=actions # a list
        self.lr=learning_rate
        self.gamma=reward_decay
        self.epsilon=e_greedy
        self.qtable=pd.DataFrame(columns=self.actions,dtype=np.float32)

    def choose_action(self,s):
        self.check_state_exist(s)

        if np.random.uniform()<self.epsilon:
            actions=self.qtable.loc[s,:]
            actions=actions.reindex(np.random.permutation(actions.index))
            action=actions.idxmax() #argmax(Q(S,:))
        else:
            action=np.random.choice(self.actions)
        return action

    def check_state_exist(self,s):
        if s not in self.qtable.index:
            self.q_table=self.qtable.append(
                pd.Series([0]*len(self.actions),index=self.actions,name=s)
            )

    def learn(self,s,a,r,s_):
        self.check_state_exist(s_)

        q_predict=self.qtable.loc[s,a]

        if s_!='terminal':
            q_target=r+self.gamma*self.qtable.loc[s_,:].max()
        else:
            q_target=r

        self.qtable.loc[s,a]+=self.lr*(q_target-q_predict) #update