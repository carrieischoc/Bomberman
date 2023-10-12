import os
import pickle
import random
from random import shuffle
import numpy as np
from sklearn.multioutput import MultiOutputRegressor as MOR
from sklearn.ensemble import GradientBoostingRegressor
#from sys import path
#path.append(r"/Users/lijiahui/LightGBM/python-package")
#from lightgbm import LGBMRegressor as LGBMR


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

#is_first_round = False


def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]



def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train and not os.path.isfile("states.csv"):
        self.logger.info("Setting up model from scratch.")
        
    else:
        self.states = np.loadtxt('states.csv', delimiter = ',')
        self.Q_value = np.loadtxt('Q_value.csv', delimiter = ',')
        #self.regression = MOR(LGBMR(zero_as_missing=True, use_missing=False)) # NAN or zeros are treated as missing
        self.regression = MOR(GradientBoostingRegressor())
        self.regression.fit(self.states,self.Q_value) # cause errors with different shape
        #self.n_rounds = np.array((np.loadtxt('rounds.csv', delimiter = ',').astype(int)))
        self.last_actions = np.loadtxt('last_actions.csv', delimiter = ',').astype(int).reshape(-1,1)
        self.rewards = np.loadtxt('rewards.csv', delimiter = ',')
        
        
    
        
        


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    self.logger.info('Picking action according to rule set')

    # game_state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [(x, y) for ((x, y), t) in bombs]
    others = [(x, y) for (n, s, b, (x, y)) in game_state['others']]
    coins = game_state['coins']
    
    dead_ends = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 1)]
    #print(0000000000)
    # find immediate targets
    free_space = (arena == 0)
    targets = []
    targets.append(look_for_targets(free_space, (x,y), coins))
    targets.append(look_for_targets(free_space, (x,y), crates))
    targets.append(look_for_targets(free_space, (x,y), dead_ends))
    targets.append(look_for_targets(free_space, (x,y), bomb_xys))
    targets.append(look_for_targets(free_space, (x,y), others))

    # calculate adjacent fields (up, right, down, left), free-0, wall/crate/explosion-1
    state = []
    directions = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
    for d in directions:
        state.append(arena[d] in (-1, 1) or game_state['explosion_map'][d] > 0 or d in bomb_xys)
        self.logger.debug(f'{d}\t{state[-1]}')
    
    # check targets or (0,0) if not found
    for target in targets:
        if target != None:
            target = (target[0]-x, target[1]-y)
        else:
            target = (0,0)

        state.append(target[0])
        state.append(target[1])

    # Check others
    state.append(0)
    directions = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    for d in directions:
        if d in others:
            state[-1] = 1
    
    # bomb flag
    state.append(not bombs_left)
    
    #check crates
    state.append(0)
    directions = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
    for d in directions:
        if arena[d] == 1:
            state[-1] = 1
            
    #check bomb timer
    bomb_position = look_for_targets(free_space,(x,y),bomb_xys)
    state.append(-1)
    for (x_,y_),t in bombs:
        if x_ == bomb_position[0] and y_ == bomb_position[1]:
            state[-1] = t
            break


    # add current state into states
    self.states = np.vstack((self.states, np.asarray(state)))

    if self.train:
        random_prob = .65
        if self.n_rounds % 100 == 0:
            random_prob -= self.n_rounds / 100 * 0.065
#    # epsilon-decreasing method
#    if self.n_rounds > 20:
#        random_prob = .2
#    if self.n_rounds > 40:
#        random_prob = .2
#    if self.n_rounds > 60:
#        random_prob = .1
#    if self.n_rounds > 80:
#        random_prob = .05
#    if self.n_rounds > 100:
#        random_prob = .1
#    if self.n_rounds > 50:
#        random_prob = .4
#    if self.n_rounds > 100:
#        random_prob = .5
#    if self.n_rounds > 150:
#        random_prob = .4
#    if self.n_rounds > 200:
#        random_prob = .3
#    if self.n_rounds > 250:
#        random_prob = .2
#    if self.n_rounds > 300:
#        random_prob = .3
#    if self.n_rounds > 350:
#        random_prob = .2
#    if self.n_rounds > 400:
#        random_prob = .1
        
    if self.train:
        n_rows = self.last_actions.shape[0]
        # If agent has been in the same location three times recently, it's a loop
        if n_rows > 5:
            if self.last_actions[-2] == self.last_actions[-4] == self.last_actions[-6] and self.last_actions[-1] == self.last_actions[-3] == self.last_actions[-5]:
                random_prob = .8
    
    
        
    #print("11111111")
    
    # to do Exploration vs exploitation
    if self.train:
        np.random.seed()
        if random.random() < random_prob:
            self.logger.debug("Choosing action purely at random.")
            action = np.random.choice(ACTIONS,p=[.176, .176, .176, .176, .176, .12])
            self.last_actions = np.vstack((self.last_actions, ACTIONS.index(action)))
            return action
        else:
            if os.path.isfile("states.csv"):
                self.logger.debug("Querying model for action.")
                action = np.argmax(self.regression.predict(np.asarray(state).reshape(1,-1)))
                self.last_actions = np.vstack((self.last_actions, action))
                #print(action,"aaaa")
                return ACTIONS[action]
            else:
                self.logger.debug("Choosing action purely at random.")
                action = np.random.choice(ACTIONS)
                self.last_actions = np.vstack((self.last_actions, ACTIONS.index(action)))
                #print(action,1111)
                return action
    else:
        self.logger.debug("Querying model for action.")
        #print(self.model)
        #self.regression.fit(self.states, self.Q_value)
        action = np.argmax(self.regression.predict(np.asarray(state).reshape(1,-1)))
        return ACTIONS[action]

