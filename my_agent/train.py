import pickle
import random
#from collections import namedtuple, deque
from typing import List
import numpy as np

import events as e
from sklearn.multioutput import MultiOutputRegressor as MOR

from sys import path
path.append(r"/Users/lijiahui/LightGBM/python-package")
from lightgbm import LGBMRegressor as LGBMR
import os
from sklearn.ensemble import GradientBoostingRegressor

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
n_actions = len(ACTIONS)

# hyperparameters
alpha = 0.75 # learning rate
gama = 0.95 # discount

# features [up, right, down, left, nearest coin(coordinate x), nearest coin(coordinate y), crate x, crate y, dead end x, dead end y, bomb x, bomb y, opponent x, opponent y, opponent flag, bomb flag, crates flag, bomb timer]
n_features = 18

#is_first_r = False

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if not os.path.isfile("states.csv"):
        self.last_actions = np.zeros((0,1),dtype=np.int8) # empty at present
            # model construction
        self.Q_value = np.zeros((0,n_actions))
        self.states = np.zeros((0,n_features))
        self.rewards = np.zeros((0,n_actions))
        
    self.n_rounds = np.array([0]) # record the number of round


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.logger.debug(f'EVENTS: {events}')
    if len(events) != 0:
    
        # update rewards
        current_reward = np.zeros((1, n_actions))
        current_reward[0][self.last_actions[-1]] = reward_from_events(self, events)
        self.rewards = np.vstack((self.rewards, current_reward))

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
   
    # update the rewards list
    current_reward = np.zeros((1, n_actions))
    current_reward[0][self.last_actions[-1]] = reward_from_events(self, events)
    self.rewards = np.vstack((self.rewards, current_reward))
   
    # update rule
    self.Q_value = np.zeros((0,n_actions))
    if os.path.isfile("states.csv"):
        q_vs = np.amax(self.regression.predict(self.states), axis=1) # predict Q(s,a)
        #print(q_vs.shape)
        for i in range(len(q_vs) - 1):
            a = self.last_actions[i][0]
            R = self.rewards[i][a] # current reward
            TD = R + gama * q_vs[i+1] - q_vs[i] # Temporal difference of Q-learning
            Q_s_a = np.zeros((1,n_actions)) # new Q(s,a)
            Q_s_a[0][a] = np.add(q_vs[i], np.multiply(alpha, TD))
            self.Q_value = np.vstack((self.Q_value, Q_s_a))
        self.Q_value = np.vstack((self.Q_value, self.rewards[i+1]))
            #print(self.Q_value.shape)
    else:
        self.Q_value = self.rewards

    #print(self.Q_value.shape[0])
    #print(self.states.shape)
   
    # Store the model
    #self.regression = MOR(LGBMR(zero_as_missing=True, use_missing=False)) # NAN or zeros are treated as missing
    #print("111111")
    self.regression = MOR(GradientBoostingRegressor())
    self.regression.fit(self.states,self.Q_value) # cause errors with different shape
    self.n_rounds = self.n_rounds + 1
    #np.savetxt('rounds.csv', self.n_rounds)
    np.savetxt('states.csv', self.states, delimiter=',')
    np.savetxt('Q_value.csv', self.Q_value, delimiter=',')
    np.savetxt('last_actions.csv', self.last_actions, delimiter=',')
    np.savetxt('rewards.csv', self.rewards, delimiter=',')
    
    
def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.WAITED: -3,
        e.CRATE_DESTROYED: 5,
        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0,
        e.COIN_COLLECTED: 300,
        e.INVALID_ACTION: -5,
        e.SURVIVED_ROUND: 100,
        e.KILLED_OPPONENT: 300,
        e.KILLED_SELF: -100,
        e.GOT_KILLED: -100,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
            if event == e.COIN_COLLECTED:
                if e.KILLED_SELF in events:
                    reward_sum += game_rewards[e.KILLED_SELF]
                if e.GOT_KILLED in events:
                    reward_sum += game_rewards[e.GOT_KILLED]
            if event == e.BOMB_DROPPED:
                # search for crates
                if ((self.states[-1][6], self.states[-1][7]) == (0, 0) and
                        self.states[-1][-2] == 1 and self.states[-1][-3] == 0):
                    reward_sum += 10

                # search for opponent
                if (self.states[-1][14] == 1 and self.states[-1][-3] == 0):
                    reward_sum += 100

                # no crates and opponent
                if ((self.states[-1][6], self.states[-1][7]) == (0, 0) and
                        self.states[-1][-2] == 0 and self.states[-1][14] == 0
                        and self.states[-1][-3] == 0):
                    reward_sum -= 10

        else:
            reward_sum -= 1
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum



