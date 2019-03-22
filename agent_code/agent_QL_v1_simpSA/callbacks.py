#TO ACT

import numpy as np
from random import shuffle
from time import time, sleep
from collections import deque
# from tempfile import TemporaryFile
import os.path
import pickle
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from decimal import Decimal

from settings import s, e

#Simple agent
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
        neighbors = [(x,y) for (x,y) in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if free_space[x,y]]
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

#posibles targets
    # others = [(x,y) for (x,y,n,b,s) in self.game_state['others']]
    # coins = self.game_state['coins']
    # Compile a list of 'targets' the agent should head towards
    # dead_ends = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 0)
    #                 and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(0) == 1)]
    # crates = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 1)]


def simple_act(self):
    # Gather information about the game state
    arena = self.game_state['arena']
    x, y, _, bombs_left, score = self.game_state['self']
    bombs = self.game_state['bombs']
    bomb_xys = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b,s) in self.game_state['others']]
    coins = self.game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for xb,yb,t in bombs:
        for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i,j] = min(bomb_map[i,j], t)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x,y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x,y))

    # Check which moves make sense at all
    directions = [(x,y), (x+1,y), (x-1,y), (x,y+1), (x,y-1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
            (self.game_state['explosions'][d] <= 1) and
            (bomb_map[d] > 0) and
            (not d in others) and
            (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x-1,y) in valid_tiles: valid_actions.append('LEFT')
    if (x+1,y) in valid_tiles: valid_actions.append('RIGHT')
    if (x,y-1) in valid_tiles: valid_actions.append('UP')
    if (x,y+1) in valid_tiles: valid_actions.append('DOWN')
    if (x,y)   in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x,y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    dead_ends = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 0)
                    and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(0) == 1)]
    crates = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x,y), targets, self.logger)
    if d == (x,y-1): action_ideas.append('UP')
    if d == (x,y+1): action_ideas.append('DOWN')
    if d == (x-1,y): action_ideas.append('LEFT')
    if d == (x+1,y): action_ideas.append('RIGHT')
    if d is None:
        self.logger.debug('All targets gone, nothing to do anymore')
        action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at dead end
    if (x,y) in dead_ends:
        action_ideas.append('BOMB')
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x,y) and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for xb,yb,t in bombs:
        if (xb == x) and (abs(yb-y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb-x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for xb,yb,t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            self.next_action = a
            break

    # Keep track of chosen action for cycle detection
    if self.next_action == 'BOMB':
        self.bomb_history.append((x,y))

def regg_forest_fit(self):
    self.flag_forest = 1

    # we create the random data, in order to use it, in regression forest
    self.logger.debug(f'Forest fit at: {self.episodes} Episodes')

    X_full = np.zeros((self.replay_minimum_size, self.state_size))
    y_full = np.zeros((self.replay_minimum_size, self.num_actions))

    for i in range(self.replay_minimum_size):
        StateStr, QVals = random.choice(list(self.StatesIndex.items()))
        StateFlat = np.array(list(StateStr), dtype=float)
        X_full[i] = StateFlat
        y_full[i] = QVals

    max_depth = 40
    self.regr_rf = RandomForestRegressor(n_estimators=100, max_depth=max_depth, random_state=2)
    self.regr_rf.fit(X_full, y_full)


    # print (X_full, y_full)

def predict_Forest(self,state_to_predict):

    if self.flag_forest:
        act_pred = self.regr_rf.predict(state_to_predict.reshape(1, -1))
        #print("Dont exist, and run Regression forrest")
        return act_pred
    else:
        state_str = str(''.join(map(str, state_to_predict)))
        if state_str in self.StatesIndex:
            act_pred = self.StatesIndex[state_str]
            self.logger.debug(f'We have the state...')
            return act_pred
        else:
            act_pred = np.random.rand(self.num_actions) * 0.05
            self.logger.debug(f'Initial guess Q vec with randoms ')

    return act_pred


    # # To perform actions just with Policy
    # state_str = str(''.join(map(str, state_to_predict)))
    # if state_str in self.StatesIndex:
    #     act_pred = self.StatesIndex[state_str]
    #     print("Exist")
    #     self.logger.debug(f'We have the state...')
    #     return act_pred
    # else:
    #     act_pred = self.regr_rf.predict(state_to_predict.reshape(1, -1))
    #     print("Predicting")
    #     return act_pred


def flag_fit_to_bomb(self):
    # Gather information about the game state
    arena = self.game_state['arena']
    x, y, _, bombs_left, score = self.game_state['self']
    others = [(x, y) for (x, y, n, b, s) in self.game_state['others']]
    # Compile a list of 'targets' the agent should head towards
    dead_ends = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x,y), dead_ends, self.logger)

    # Add proposal to drop a bomb if at dead end
    if (x,y) in dead_ends:
        return 1
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            return 1
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x,y) and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(1) > 0):
        return 1
    return 0

def get_free_cells(self, xa, ya, arena, bomb_dic, explosion_map, time):
    queue = deque([(xa, ya)])
    visited = {}
    visited[(xa, ya)] = 1
    free = 0
    while (len(queue) > 0):
        curr_x, curr_y = queue.popleft()
        curr = (curr_x, curr_y)
        directions = [(curr_x, curr_y - 1), (curr_x, curr_y + 1), (curr_x - 1, curr_y), (curr_x + 1, curr_y)]

        if (visited[curr]) == time:
            break

        for (xd, yd) in directions:
            d = (xd, yd)
            if ((arena[d] == 0) and
                    (explosion_map[curr] <= visited[curr] + 1) and
                    (not d in bomb_dic or not (visited[curr] + 1) in bomb_dic[d]) and
                    (not d in visited)):
                queue.append(d)
                visited[d] = visited[curr] + 1
                if not d in bomb_dic:
                    free += 1

    return free

def sel_action_target(self,targets):

    arena = self.game_state['arena']
    x, y, _, bombs_left, score = self.game_state['self']
    bombs = self.game_state['bombs']
    bomb_xys = [(x,y) for (x,y,t) in bombs]

    coins = self.game_state['coins']

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]
    # Take a step towards the most immediately interesting target
    free_space = arena == 0

    # if self.ignore_others_timer > 0:
    #     for o in others:
    #         free_space[o] = False
    #
    d = look_for_targets(free_space, (x,y), targets, self.logger)
    action = 'WAIT'
    if d == (x,y-1): action = 'UP'
    if d == (x,y+1): action ='DOWN'
    if d == (x-1,y): action = 'LEFT'
    if d == (x+1,y): action = 'RIGHT'

    return action

def mappping(self):
    # State definition
    #arena = np.zeros((s.rows,s.cols))
    # Gather information about the game state
    arena = self.game_state['arena']
    # aux_arena = np.zeros((s.rows, s.cols))
    # aux_arena[:,:] = arena
    x, y, _, bombs_left, score = self.game_state['self']
    bombs = self.game_state['bombs']
    bombs_xys = [(x, y, t) for (x, y, t) in bombs]
    bomb_dic = {}
    # others = [(x,y) for (x,y,n,b,s) in self.game_state['others']]
    coins = self.game_state['coins']
    explosion_map = self.game_state['explosions']

    bomb_map_timer = np.zeros(arena.shape)

    # dictionary of bombs
    for (xb, yb, t) in bombs:
        # when other agent mark arena as well
        vec_dir = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for v in vec_dir:
            hx, hy = v
            for i in range(4):
                xcoord = xb + hx * i
                ycoord = yb + hy * i
                if ((0 < xcoord < arena.shape[0]) and
                        (0 < ycoord < arena.shape[1]) and
                        (arena[(xcoord, ycoord)] == 1 or arena[(xcoord, ycoord)] == 0)):
                    bomb_map_timer[(xcoord, ycoord)] = t
                    if (xcoord, ycoord) in bomb_dic:
                        bomb_dic[(xcoord, ycoord)].append(t)
                    else:
                        bomb_dic[(xcoord, ycoord)] = [t]
                else:
                    break

    # map of bombs
    bomb_map = np.zeros(arena.shape)

    for (xb, yb, t) in bombs:
        vec_dir = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for v in vec_dir:
            hx, hy = v
            for i in range(0, 4 - t + 1):
                xcoord = xb + hx * i
                ycoord = yb + hy * i
                if ((0 < xcoord < arena.shape[0]) and
                        (0 < ycoord < arena.shape[1]) and
                        (arena[(xcoord, ycoord)] == 1 or arena[(xcoord, ycoord)] == 0)):
                    bomb_map[(xcoord, ycoord)] = 1
                else:
                    break

    self.dead_zone = bomb_map

    # General case
    # state = np.zeros(32, dtype = int)

    # Coins case
    state = np.zeros(self.state_size, dtype = int)

    # 0. UP ->    (x  , y-1)
    # 1. DOWN ->  (x  , y+1)
    # 2. LEFT ->  (x-1, y  )
    # 3. RIGHT -> (x+1, y  )

    # 4 bits for valid position
    valid = np.array([0, 0, 0, 0])

    directions = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
    for i in range(4):
        d = directions[i]
        if (arena[d] == 0 and explosion_map[d] <= 1 and bomb_map[d] == 0):
            valid[i] = 1

    state[:4] = valid

    # 1 bit  for flag bomb
    state[4] = bombs_left


    #Building the target state..
    # posibles targets
    # others = [(x,y) for (x,y,n,b,s) in self.game_state['others']]
    coins = self.game_state['coins']
#    targets = coins + dead_ends + crates
    bit_action = sel_action_target(self, coins)

    if (bit_action != 'WAIT'):
        idx_bit = self.map_actions[bit_action]
        state[5 + idx_bit] = 1
    else:
        crates = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 1)]
        bit_action = sel_action_target(self, crates)
        if (bit_action != 'WAIT'):
            idx_bit = self.map_actions[bit_action]
            state[5 + idx_bit] = 1


    # Compile a list of 'targets' the agent should head towards
    # dead_ends = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 0)
    #                 and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(0) == 1)]
    # crates = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 1)]

    #def sel_action_target(self, targets):


    # Scape route
    if (bomb_map_timer[x, y] > 0):
        free_max = 5
        for i in range(4):
            x_curr, y_curr = directions[i]
            d = directions[i]

            if ((arena[d] == 0) and
                    (explosion_map[d] <= 1)):
                time = bomb_map_timer[x, y]
                free_curr = get_free_cells(self, x_curr, y_curr, arena, bomb_dic, explosion_map, time)
                if free_curr > free_max:
                    free_curr = free_max
                    #idx_direction = i
                state[9 + i] = free_curr


        # if free_max > 0:
        #     state[9 + idx_direction] = 1

    # Number of crates

    number_crates = 0
    vec_dir = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    for v in vec_dir:
        hx, hy = v
        for i in range(1, 4):
            xcoord = x + hx * i
            ycoord = y + hy * i
            if ((0 < xcoord < arena.shape[0]) and
                    (0 < ycoord < arena.shape[1]) and
                    (arena[(xcoord, ycoord)] == 1 or arena[(xcoord, ycoord)] == 0)):
                if arena[(xcoord, ycoord)] == 1:
                    number_crates += 1
            else:
                break

    if number_crates < 10:
        state[13] = number_crates
    else:
        state[13] = 9

    # when is next to the crate or opponent (State coins all zeros) activate
    if not state[5:9].any() and state[4] == 1:
        state[14] = 1
    #flag_fit_to_bomb(self)
    # self.logger.debug(f'STATE VALID: {state[:4]}')
    # self.logger.debug(f'STATE BOMB: {state[4]}')
    # self.logger.debug(f'STATE COINS: {state[5:9]}')
    # self.logger.debug(f'STATE SCAPE: {state[9:13]}')
    # self.logger.debug(f'STATE CRATES: {state[13]}')

    print("STATE VALID: ",state[:4])
    print("STATE AVALIBLE BOMB: ",state[4])
    print("STATE COINS: ",state[5:9])
    print("STATE SCAPE: ",state[9:13])
    print("STATE CRATES: ",state[13])
    print("STATE DROP BOMB: ", state[14], "\n")

    return state

def get_reward(self):
    self.next_state = mappping(self)

    x, y, _, bombs_left, score = self.game_state['self']

    reward = 0

    # The agent took a invalid action and it should be punished
    if e.INVALID_ACTION in self.events:
        reward += self.reward_list['INVALID_ACTION']
        self.actions_invalid += 1
        if self.flag_actions_taken_model == 1:
            self.actions_taken_model_invalid += 1
            # Flag, if the action belongs to the model to 0
            self.flag_actions_taken_model = 0

        self.logger.debug("INVALID ACTION")

    # A coin was found, therefore the agent receives a reward for that
    if e.COIN_COLLECTED in self.events:
        reward += self.reward_list['COIN_COLLECTED']
        self.coins_collected += 1
        self.logger.debug("COIN COLLECTED")
    # else:      #In case of crates, we dont mind about optimazed path
    #    reward += self.reward_list['VALID']

    # In order to incentevi the optimal number of crates destroyed per Bomb dropped:
    # We give a reward proportional to NCrates*NCrates.
    if (e.CRATE_DESTROYED in self.events) and ( not e.KILLED_SELF in self.events):
        self.logger.debug("SURVIVE BOMB")
        NCrates = list(self.events).count(9)
        self.number_crates_destroyed += NCrates
        reward += NCrates * NCrates * self.reward_list['CRATE_DESTROYED']/2
        self.logger.debug(f'CRATES DESTROYED: {NCrates}')

    if e.COIN_FOUND in self.events:
        reward += self.reward_list['COIN_FOUND']
        self.logger.debug("COIN_FOUND")

    if e.KILLED_SELF in self.events:
        reward += self.reward_list['KILLED_SELF']
        self.logger.debug("KILLED_SELF")
        self.actions_killed_self += 1

    # if e.BOMB_DROPPED in self.events:
    #     self.actions_bomb_dropped += 1
    #     reward += self.reward_list['BOMB_DROPPED']
    #     self.logger.debug("DROP_BOMB")

    if self.dead_zone[x, y] > 0:
        self.actions_dead_zone += 1
        reward += self.reward_list['DEAD_ZONE']

    self.total_reward = reward

    self.reward_episode += self.total_reward

def remember(self, state, action, reward, next_state, done):
    # Remember previous experiences from the game
    self.memory.append((state, action, reward, next_state, done))

def replay(self):
    regg_forest_fit(self)

def replay_quick(self):
    state, action, reward, next_state, done = self.memory[len(self.memory) - 1]

    state_flat = np.asarray(state).reshape(-1)
    state_str = str(''.join(map(str, state_flat)))
    #print(state_str)

    target = reward
    #print(self.StatesIndex)
    if not done:
        target += self.gamma * np.amax(predict_Forest(self,next_state))
        if state_str in self.StatesIndex:
            Q_state = self.StatesIndex[state_str]
            #print("Q_state: ",Q_state)
            Q_state_action = Q_state[0, action]
            target -= Q_state_action
            Q_state[0, action] = Q_state_action + self.learning_rate * target
            self.StatesIndex[state_str] = Q_state
        else:
            Q_temp =  np.zeros([1,self.num_actions])
            Q_temp[0, action] = target
            self.StatesIndex[state_str] = Q_temp

            #print("Q_temp",Q_temp)

def update_epsilon(self):
    if self.epsilon > self.epsilon_min:
        self.epsilon -= self.epsilon_decay

def update_model(self, done):
    # Store the tuple (state, action, reward, next_state, done) via remember function
    # done is a boolean that indicates if the episode is finished.
    remember(self, self.state, self.idx_action, self.total_reward, self.next_state, done)

    # Train if the memory is large enough (this is defined by replay_minimum_size) via replay function
    if done:
        if len(self.memory) >= self.replay_minimum_size:
            replay(self)
        else:
            replay_quick(self)
    else:
        replay_quick(self)

    # Decrease epsilon each time step
    update_epsilon(self)

def load_policy(self):
    if os.path.isfile(self.model_path_load):
        with open(self.model_path_load, 'rb') as handle:
            self.StatesIndex = pickle.load(handle)
        len_dic = len(self.StatesIndex)

        self.logger.debug(f'We start with a dictionary of len {len_dic}')
        # Number next index state related
        if len_dic < self.replay_minimum_size:
            self.replay_minimum_size = len_dic
        regg_forest_fit(self)
        print("Begining with #states: ", len_dic)
        # print(self.StatesIndex)
    else:
        print("Hey, is lack of initial Q guess")

def save_policy(self):
    print(self.persistence_update_frequency, " episodes")
    with open(self.model_path_save, 'wb') as handle:
        pickle.dump(self.StatesIndex, handle, protocol=pickle.HIGHEST_PROTOCOL)

def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.logger.debug('Successfully entered setup code')
    np.random.seed()

    # ACTIONS

    # Init the 6 possible actions
    # 0. UP ->    (x  , y-1)
    # 1. DOWN ->  (x  , y+1)
    # 2. LEFT ->  (x-1, y  )
    # 3. RIGHT -> (x+1, y  )
    # 4. WAIT ->  (x  , y  )
    # 5. BOMB ->  (x  , y  )

    self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']

    # Case to just collect the coins
    #self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    # Init map of actions
    self.map_actions = {
        'UP': 0,
        'DOWN': 1,
        'LEFT': 2,
        'RIGHT': 3,
        'WAIT': 4,
        'BOMB': 5
    }

    # Number of possible actions
    self.num_actions = len(self.actions)

    # action index (this is really the variable used as action)
    self.idx_action = 0

    # STATE

    # state size
    self.state_size = 15

    self.state = np.zeros(self.state_size,dtype=int)

    # next_state defined as state
    self.next_state = np.zeros(self.state_size,dtype=int)

    # HYPERPARAMETERS

    # Init gamma
    self.gamma = 0.95

    # Init the memory size
    self.memory_size = 20000

    # Init the minimum length to fit reggresion forest
    self.replay_minimum_size = 500

    # Exploration rate when training
    self.epsilon = 1.0

    # Exploration steps
    self.exploration_steps = 2000000

    # Minimum value of epsilon, after this value
    # epsilon does not decrease anymore
    self.epsilon_min = 0.1

    # This hyperparameter is to decrease the number
    # of explorations as the agent gets better

    self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.exploration_steps

    # Exploration rate when playing
    self.epsilon_play = 0.02

    self.learning_rate = 0.001

    # This hyperparameter controls how often the NN is saved
    self.persistence_update_frequency = 100

    # MEMORY

    # List for the previous experiences
    # It's a deque because once the maxlen is reached, the oldest
    # experiences are forgotten (the newest experiences is what matters)
    # (state, action, reward, next_action, done)
    self.memory = deque(maxlen=self.memory_size)

    # REWARDS

    # Reward accumulated for every 4 frames
    self.total_reward = 0

    # Reward List

    number_of_free = (s.cols - 2) * (s.rows // 2) + (s.rows // 2 - 1) * ((s.cols - 2) // 2 + 1)
    number_of_crates = s.crate_density * (number_of_free - 12)
    reward_coin = 100
    if (number_of_crates > 0):
        ratio_coins_crates = number_of_crates / 9
        reward_crate = int(reward_coin / ratio_coins_crates)
    else:
        reward_crate = 0

    self.reward_list = {
        'OPPONENT_ELIMINATED': 500,
        'COIN_COLLECTED': reward_coin,
        'CRATE_DESTROYED': reward_crate,
        'INVALID_ACTION': -8,
        'DEAD_ZONE': -1,
        'VALID': -2,
        'DIE': -500,
        'COIN_FOUND': 20,
        'KILLED_SELF': -30,
        'BOMB_DROPPED': 8
    }


    # COUNTERS

    # Total steps
    self.total_steps = 0
    # Random actions taken
    self.actions_taken_random = 0

    # Simple agent actions taken
    self.actions_taken_simple = 0
    # Actions taken based on the model
    self.actions_taken_model = 0
    # Invalid Actions taken based on the model
    self.actions_taken_model_invalid = 0
    # Flag, if the action belongs to the model
    self.flag_actions_taken_model = 0

    # Coins collected
    self.coins_collected = 0
    # Number of episodes
    self.episodes = 0
    # Number of invalid actions
    self.actions_invalid = 0

    # Measures
    self.start_time1 = time()
    self.start_time2 = time()
    self.reward_episode = 0
    self.reward_total = 0
    self.elapsed_time_action = 0.0
    self.elapsed_time_model = 0.0
    self.q_mean = 0.0
    self.distance_coins_total = 0.0
    self.actions_bomb_dropped = 0
    self.actions_killed_self = 0
    self.number_crates_destroyed = 0
    self.actions_dead_zone = 0

    # Simple agent
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0

    # Regg Forest
    # Were to save Qs
    self.model_path_save = './agent_code/agent_QL_v1_simpSA/Poly-Update.pickle'
    # Were to load Qs
    self.model_path_load = './agent_code/agent_QL_v1_simpSA/Poly_SA.pickle'
    # Flag to start the regg forest
    self.flag_forest = 0
    # Load Dictionary
    self.StatesIndex = {
    }
    # Load Policy
    load_policy(self)

def act(self):
    """Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via self.game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    Set the action you wish to perform by assigning the relevant string to
    self.next_action. You can assign to this variable multiple times during
    your computations. If this method takes longer than the time limit specified
    in settings.py, execution is interrupted by the game and the current value
    of self.next_action will be used. The default value is 'WAIT'.
    """

    # Gather information about the game state

    self.state = mappping(self)
    #print(self.state)
    # Increase the counter of total time steps
    self.total_steps += 1

    # Exploration
    # if self.state[4] == 0 :
    #     simple_act(self)
    #     self.logger.info('Picking Simple Agent action')
    #     self.idx_action = self.map_actions[self.next_action]
    #     # Increase the number of simple agent actions that has been taken
    #     self.actions_taken_simple += 1
    # else:
    # # When playing use
    # # if np.random.rand() <= self.epsilon_play:
    #
    # # When training use
    #     ran = np.random.rand()
    #     if ran <= self.epsilon:
    #         self.logger.info('Picking RANDOM action')
    #         idx_random = np.random.randint(self.num_actions)
    #         self.next_action = self.actions[idx_random]
    #         self.idx_action = idx_random
    #         # Increase the number of random actions that has been taken
    #         self.actions_taken_random += 1

        # #if ran <= 1.0:
        #     # if ran <= self.epsilon*:
        #     #     simple_act(self)
        #     #     self.logger.info('Picking Simple Agent action')
        #     #     self.idx_action = self.map_actions[self.next_action]
        #     #     # Increase the number of simple agent actions that has been taken
        #     #     self.actions_taken_simple += 1
        #     # else:
        #     #
        #     #     self.logger.info('Picking RANDOM action')
        #     #     idx_random = np.random.randint(self.num_actions)
        #     #     self.next_action = self.actions[idx_random]
        #     #     self.idx_action = idx_random
        #     #     # Increase the number of random actions that has been taken
        #     #     self.actions_taken_random += 1
        # #Exploitation
        # else:
    start_time1Act = time()
    self.logger.info('Picking action according to the MODEL')
    q_values = predict_Forest(self,self.state)
    # q_values = self.model.predict( self.state.reshape((1, self.state_size)) )
    self.idx_action = np.argmax(q_values)
    self.next_action = self.actions[self.idx_action]
    # Increase the number of actions that has been taken based on the model
    self.actions_taken_model += 1
    # Flag, if the action belongs to the model
    self.flag_actions_taken_model = 1
    #self.q_mean += np.sum(q_values) / self.num_actions  ##neu
    # time end action
    self.elapsed_time_action += time() - start_time1Act


def reward_update(self):
    """Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.
    """
    #self.logger.debug(f'Encountered {len(self.events)} game event(s)')

    # Get the reward based on the events from the previous step
    get_reward(self)

    # Update the model
    start_time1Model = time()
    update_model(self, False)
    self.elapsed_time_model += time() - start_time1Model


def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
   # self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')

    for StateStr, QVals in list(self.StatesIndex.items()):
        self.q_mean += np.sum(QVals)/len(QVals)

    self.q_mean /= len(self.StatesIndex)

    # Get the reward based on the events from the previous step
    get_reward(self)

    # Update the model
    update_model(self, True)

    # Increase the counter for the number of episodes
    self.episodes += 1

    if self.episodes % 2 == 1:
        self.start_time1 = time()
        elapsed_time = time() - self.start_time2
    else:
        self.start_time2 = time()
        elapsed_time = time() - self.start_time1

    total_actions = self.actions_taken_random + self.actions_taken_model
    #self.q_mean /= total_actions
    self.reward_total += self.reward_episode

    x, y, _, bombs_left, score = self.game_state['self']

    # some conventions, P-'Performance' S-'Setings' A-'Action' T-'Time'
    self.logger.debug('----------------------------------------------')
    self.logger.debug(f'P-Score: {score}')
    self.logger.debug(f'P-Reward acommulated: {self.reward_episode}')
    self.logger.debug(f'P-RewardsTotal: {self.reward_total}')
    self.logger.debug(f'S-Episode: {self.episodes}')
    self.logger.debug(f'S-Epsilon: {self.epsilon}')
    self.logger.debug(f'A-Invalid: {self.actions_invalid}')
    self.logger.debug(f'A-Random: {self.actions_taken_random}')
    self.logger.debug(f'A-Simple: {self.actions_taken_simple}')
    self.logger.debug(f'A-Model: {self.actions_taken_model}')
    self.logger.debug(f'A-InvalidModel: {self.actions_taken_model_invalid}')
    self.logger.debug(f'A-DeadZone: {self.actions_dead_zone}')
    self.logger.debug(f'A-Total: {total_actions}')
    #self.logger.debug(f'T-Action: {self.elapsed_time_action / total_actions}')
    self.logger.debug(f'T-TimeEpisodes: {elapsed_time} :s')

    if self.actions_taken_model > 0:
        self.logger.debug(f'T-Model: {self.elapsed_time_model / (self.actions_taken_model)}')
    else:
        self.logger.debug("T-Model: -1")

    if self.actions_taken_model > 0:
        self.logger.debug(
            f'M-Model_Invalid/ModelAct: {(self.actions_taken_model_invalid) / (self.actions_taken_model)}')
    else:
        self.logger.debug("M-Model_Invalid/ModelAct: -1")

    if self.distance_coins_total > 0:
        self.logger.debug(f'M-Steps/OptimalDistance: {total_actions / self.distance_coins_total}')
    else:
        self.logger.debug("M-Steps/OptimalDistance: -1")

    if self.actions_bomb_dropped > 0:
        self.logger.debug(f'M-KilledSelfRate: {self.actions_killed_self / self.actions_bomb_dropped}')
        self.logger.debug(f'M-Crates/BombsDroped: {self.number_crates_destroyed/ self.actions_bomb_dropped}')
    else:
        self.logger.debug("M-KilledSelfRate: -1")
        self.logger.debug("M-Crates/BombsDroped: -1")

    self.logger.debug(f'QMean: {self.q_mean}')

    # Init counters once an episode has ended
    self.coins_collected = 0
    self.actions_taken_model = 0
    self.actions_taken_random = 0
    self.actions_taken_simple = 0
    self.actions_invalid = 0
    self.reward_episode = 0
    self.q_mean = 0.0
    self.actions_taken_model_invalid = 0
    self.actions_bomb_dropped = 0
    self.actions_killed_self = 0
    self.number_crates_destroyed = 0
    self.actions_dead_zone = 0

    if self.episodes % self.persistence_update_frequency == 0:
        save_policy(self)
