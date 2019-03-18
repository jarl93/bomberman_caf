
import numpy as np
import random
from time import time, sleep
from collections import deque

from random import shuffle

from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.layers.core import Dense, Dropout
from keras.activations import relu
from keras.models import load_model

from settings import s, e

#Simple agent Function
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

#Random function
def RandomAct(self):
    self.logger.info('Picking RANDOM action')
    idx_random = np.random.randint(self.num_actions)
    self.next_action = self.actions[idx_random]
    self.idx_action = idx_random

def simpleAct(self):
    self.logger.info('Picking Simple Agent action')

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
    #self.logger.debug(f'Valid actions: {valid_actions}')

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
        #self.logger.debug('All targets gone, nothing to do anymore')
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

def build_model(self):
    
    # Neural network for Deep-Q learning Model
    
    model = Sequential()
    
    model.add(Dense(120, input_dim=self.state_size, activation='relu'))
    model.add(Dropout(0.15))
    
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.15))
    
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.15))
    
    model.add(Dense(self.num_actions, activation='softmax'))
    
    opt = Adam(self.learning_rate)
    
    model.compile(loss='mse', optimizer=opt)
    
    return model

#Comprobado con user_agent
def get_region(xa, ya, xo, yo):
    # upper
    if (xo == xa and yo < ya):
        return 0
    # lower
    if (xo == xa and yo > ya):
        return 1
    # left
    if (xo < xa and yo == ya):
        return 2
    # right
    if (xo > xa and yo == ya):
        return 3

    # upper-left
    if (xo < xa and yo < ya):
        return 4
    # lower-left
    if (xo < xa and yo > ya):
        return 5
    # lower-right
    if (xo > xa and yo > ya):
        return 6
    # upper-right
    if (xo > xa and yo < ya):
        return 7
    return -1

def get_distance_coins(self, xa, ya, arena, coins):
    queue = deque([(xa, ya)])
    visited = {}
    visited[(xa, ya)] = 0
    num_coins = 0
    dist_path = 0
    coins_region = []

    # self.logger.debug(f'AGENT:\n {xa, ya}')

    for (xc, yc) in coins:
        arena[xc, yc] = 2
        num_coins += 1

    # self.logger.debug(f'REGION:\n {region}')
    # self.logger.debug(f'ARENA:\n {arena}')

    while (len(queue) > 0) and (len(coins_region) < num_coins):

        curr_x, curr_y = queue.popleft()

        if (arena[(curr_x, curr_y)] == 2):
            coins_region.append((curr_x, curr_y))
            dist_path += visited[(curr_x, curr_y)]
            arena[(curr_x, curr_y)] = 0
            queue = deque([(curr_x, curr_y)])
            visited.clear()
            visited[(curr_x, curr_y)] = 0

        directions = [(curr_x, curr_y - 1), (curr_x, curr_y + 1), (curr_x - 1, curr_y), (curr_x + 1, curr_y)]
        for (xd, yd) in directions:
            d = (xd, yd)
            if (arena[d] == 0 or arena[d] == 2) and (not d in visited):
                queue.append(d)
                visited[d] = visited[(curr_x, curr_y)] + 1

    return dist_path

#Comprobado con user_agent
def get_ratio_coins(self, xa, ya, region, arena, coins):
    queue = deque([(xa, ya)])
    visited = {}
    visited[(xa, ya)] = 0
    num_coins = 0
    dist_path = 0
    coins_region = []
    ratio = 0

    # self.logger.debug(f'AGENT:\n {xa, ya}')

    for (xc, yc) in coins:
        if get_region(xa, ya, xc, yc) == region:
            arena[xc, yc] = 2
            num_coins += 1

    # self.logger.debug(f'REGION:\n {region}')
    # self.logger.debug(f'ARENA:\n {arena}')
    if num_coins > 0:
        while (len(queue) > 0) and (len(coins_region) < num_coins):

            curr_x, curr_y = queue.popleft()

            if (arena[(curr_x, curr_y)] == 2):
                coins_region.append((curr_x, curr_y))
                dist_path += visited[(curr_x, curr_y)]
                arena[(curr_x, curr_y)] = 0
                queue = deque([(curr_x, curr_y)])
                visited.clear()
                visited[(curr_x, curr_y)] = 0

            directions = [(curr_x, curr_y - 1), (curr_x, curr_y + 1), (curr_x - 1, curr_y), (curr_x + 1, curr_y)]
            for (xd, yd) in directions:
                d = (xd, yd)
                if (arena[d] == 0 or arena[d] == 2) and (not d in visited):
                    queue.append(d)
                    visited[d] = visited[(curr_x, curr_y)] + 1

        # self.logger.debug(f'COINS REGION:\n {coins_region}')

        if (dist_path > 0):
            ratio = num_coins / dist_path

    # self.logger.debug(f'COINS, DISTANCE:\n {num_coins, dist_path}')

    return ratio

#Comprobado con user_agent
def mappping(self):
    # State definition

    # Gather information about the game state
    arena = self.game_state['arena']
    aux = np.zeros((s.rows, s.cols), dtype=int)
    aux[:, :] = self.game_state['arena']
    x, y, _, bombs_left, score = self.game_state['self']
    # bombs = self.game_state['bombs']
    # bomb_xys = [(x,y,t) for (x,y,t) in bombs]
    # others = [(x,y) for (x,y,n,b,s) in self.game_state['others']]
    coins = self.game_state['coins']
    # explosion_map = self.game_state['explosions']
    # bomb_map = np.zeros(arena.shape)

    # map for bombs
    # for (xb, yb, t) in bombs:
    #    for (i, j, h) in [(xb+h, yb, h) for h in range(-3,4)] + [(xb, yb+h, h) for h in range(-3,4)]:
    #        if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
    #            if (t - abs(h) > 0):
    #                bomb_map[i,j] = (t - abs(h))

    if self.game_state['step'] == 1:
        self.distance_coins_total = get_distance_coins(self, x, y, arena, coins)
        #print("Aca: ",self.distance_coins_total )
    # Coins case
    state = np.zeros(self.state_size)

    # 0. UP ->    (x  , y-1)
    # 1. DOWN ->  (x  , y+1)
    # 2. LEFT ->  (x-1, y  )
    # 3. RIGHT -> (x+1, y  )

    # 4 bits for position

    directions = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
    for i in range(4):
        d = directions[i]
        if (arena[d] == 0):
            state[i] = 1

    # for coord in coins:
    #    aux[coord] = 2

    # self.logger.debug(f'ARENA AUX:\n {aux}')

    # get the ratio number of coins/distance to the coins
    state_coins = np.zeros(8)
    for r in range(8):
        state_coins[r] = get_ratio_coins(self, x, y, r, arena, coins)

    # tie - breaker
    idx = np.where(state_coins == np.amax(state_coins))
    list_idx = list(idx[0])

    if (len(list_idx) > 1):
        #self.logger.debug("TIE")

        for i in list_idx:
            state_coins[i] *= np.random.uniform(0.7, 1)

    #Inicializacion con un ruido random para probar si mejora la convergencia:
    for r in range(8):
        state_coins[r] += np.random.uniform(0.001, 0.03)


    state[4:] = state_coins

    #print(state)
    #self.logger.debug(f'POSITION:\n {x, y}')
    #self.logger.debug(f'STATE VALID: {state[:4]}')
    #self.logger.debug(f'STATE COINS: {state[4:]}')

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

        #self.logger.debug("INVALID ACTION")
        
    # A coin was found, therefore the agent receives a reward for that
    if e.COIN_COLLECTED in self.events:
        reward += self.reward_list['COIN_COLLECTED']
        self.coins_collected += 1
        #self.logger.debug("COIN COLLECTED")
    else:
        reward += self.reward_list['VALID']
    
    self.total_reward = reward
    
    self.reward_episode += self.total_reward
    
def remember(self, state, action, reward, next_state, done):
    # Remember previous experiences from the game
    self.memory.append((state, action, reward, next_state, done))

def replay(self):
    
    # Get a random sample from the memory
    batch = random.sample(self.memory, self.batch_size)
    
    for state, action, reward, next_state, done in batch:
        
        target = reward
        
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            
        target_f = self.model.predict(np.array([state]))
        
        target_f[0][action] = target
        
        self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
     
    # Fit out of the for statement
        #list_states.append(state)
        #list_targets.append(target_f)
        
    # self.model.fit (list_states, list_targets, epochs= 32 , verbose=0)

def replay_quick(self):
    
    state, action, reward, next_state, done = self.memory [len(self.memory) - 1]
    
    target = reward
    
    if not done:
        target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, self.state_size)))[0])
    target_f = self.model.predict(state.reshape((1, self.state_size)))
    target_f[0][action] = target
    self.model.fit(state.reshape((1, self.state_size)), target_f, epochs=1, verbose=0)

def update_epsilon (self):
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

    
    #Save the model every model_persistence_update_frequency steps
    if self.total_steps % self.persistence_update_frequency == 0:
        self.model.save(self.model_path) #saving model

def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    #self.logger.debug('Successfully entered setup code')
    np.random.seed()

    #ACTIONS
    
    # Init the 6 possible actions
    # 0. UP ->    (x  , y-1)
    # 1. DOWN ->  (x  , y+1)
    # 2. LEFT ->  (x-1, y  )
    # 3. RIGHT -> (x+1, y  )
    # 4. WAIT ->  (x  , y  )
    # 5. BOMB ->  (x  , y  )
    
    #self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']
    
    # Case to just collect the coins
    self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    #Init map of actions
    self.map_actions = {
        'UP': 0, 
        'DOWN': 1 , 
        'LEFT': 2,
        'RIGHT': 3,
        'WAIT': 4,
        'BOMB': 5
    }
    
    # Number of possible actions
    # self.num_actions = 6
    
    # Case to just collect the coins
    self.num_actions = 4
    
    # action index (this is really the variable used as action)
    self.idx_action = 0
    
    #STATE
    
    # state size
    self.state_size = 12   

    self.state = np.zeros(self.state_size)
    
    # next_state defined as state
    self.next_state = np.zeros(self.state_size)
       
    
    #HYPERPARAMETERS
    
    # Init gamma
    self.gamma = 0.95
    
    # Init the memory size 
    self.memory_size = 200000
    
    # Init the minimum length to do a replay
    self.replay_minimum_size = 1000
    
    #Init the batch size
    self.batch_size = 1000
    
    # Exploration rate when training
    self.epsilon = 1.0
    
    # Exploration steps
    self.exploration_steps = 100000
    
    # Minimum value of epsilon, after this value
    # epsilon does not decrease anymore
    self.epsilon_min = 0.1
    
    # This hyperparameter is to decrease the number 
    # of explorations as the agent gets better
    
    self.epsilon_decay = (self.epsilon - self.epsilon_min)/self.exploration_steps
    
    #Exploration rate when playing
    self.epsilon_play = 0.02
    
    self.learning_rate = 0.0005
    
    # This hyperparameter controls how often the NN is saved
    self.persistence_update_frequency = 1000

    #MEMORY
    
    # List for the previous experiences
    # It's a deque because once the maxlen is reached, the oldest
    # experiences are forgotten (the newest experiences is what matters)
    # (state, action, reward, next_action, done)
    self.memory = deque(maxlen= self.memory_size)

    # MODEL
    self.model_path = './agent_code/1_agentNN1/model.h5'


    # NN for training
    self.model = build_model(self)

    # NN for playing or
    #self.model = load_model(self.model_path)


    #REWARDS
    
    # Reward 
    self.total_reward = 0
    
    # Reward List
    
    number_of_free = (s.cols-2) * (s.rows//2) + (s.rows//2 - 1)*((s.cols-2)//2 + 1)
    number_of_crates = s.crate_density * (number_of_free - 12)
    reward_coin = 100
    if (number_of_crates > 0):
        ratio_coins_crates = number_of_crates/9
        reward_crate = int (reward_coin / ratio_coins_crates)
    else:
        reward_crate = 0
    
    self.reward_list = {
            'OPPONENT_ELIMINATED': 500,
            'COIN_COLLECTED' : reward_coin,
            'CRATE_DESTROYED' : reward_crate,
            'INVALID_ACTION': -8,
            'VALID' : -2,
            'DIE' : -500
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
    #Flag, if the action belongs to the model
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

    #Simple agent
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0

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
    
    self.logger.info(f'STATE SIZE: {self.state.shape}')
    # Gather information about the game state
    self.state = mappping(self)
    
    # Increase the counter of total time steps
    self.total_steps += 1
    
    #Exploration
        
    #When playing use
    #if np.random.rand() <= self.epsilon_play:
    
    #When training use
    ran = np.random.rand()
    #print("random is: ",ran,"And epsilon is: ", self.epsilon)
    if ran <= self.epsilon or len(self.memory) < self.replay_minimum_size:
        if ran <=  self.epsilon/2:
            RandomAct(self)
            # Increase the number of random actions that has been taken
            self.actions_taken_random += 1
        else:
            simpleAct(self)
            self.actions_taken_simple += 1
    #Exploitation
    else:
        #Start counting time action
        start_time1Act = time()
        self.logger.info('Picking action according to the MODEL')
        q_values = self.model.predict(self.state.reshape((1, self.state_size)) )
        self.idx_action = np.argmax(q_values[0])
        self.next_action = self.actions[self.idx_action]
        # Increase the number of actions that has been taken based on the model
        self.actions_taken_model += 1
        # Flag, if the action belongs to the model
        self.flag_actions_taken_model = 1
        self.q_mean += np.sum(q_values[0])/self.num_actions  ##neu
        #time end action
        self.elapsed_time_action = time() - start_time1Act

    # Decrease epsilon each time step
    update_epsilon(self)

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
    self.elapsed_time_model = time() - start_time1Model

def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')
    
    # Get the reward based on the events from the previous step
    get_reward(self)
    
    # Update the model
    update_model(self, True)
       
    #Increase the counter for the number of episodes
    self.episodes += 1
    
    if self.episodes % 2 == 1:
        self.start_time1 = time()
        elapsed_time = time() - self.start_time2
    else:
        self.start_time2 = time()
        elapsed_time = time() - self.start_time1

    totalsteps = self.actions_taken_random + self.actions_taken_model + self.actions_taken_simple
    self.q_mean /= totalsteps
    self.reward_total += self.reward_episode

    x, y, _, bombs_left, score = self.game_state['self']

    #some conventions, P-'Performance' S-'Setings' A-'Action' T-'Time'
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
    self.logger.debug(f'A-Total: {totalsteps}')
    self.logger.debug(f'T-Action: {self.elapsed_time_action/totalsteps}')
    self.logger.debug(f'T-Model: {self.elapsed_time_model/totalsteps}')
    self.logger.debug(f'T-TimeEpisodes: {elapsed_time} :s')
    self.logger.debug(f'M-Model_Invalid/ModelAct: {(self.actions_taken_model_invalid)/(self.actions_taken_model+1)}')
    self.logger.debug(f'M-Steps/OptimalDistance: {totalsteps/self.distance_coins_total}')


    #Init counters once an episode has ended
    self.coins_collected = 0
    self.actions_taken_model = 0
    self.actions_taken_random = 0
    self.actions_taken_simple = 0
    self.actions_invalid = 0
    self.reward_episode = 0
    self.q_mean = 0.0
    self.actions_taken_model_invalid = 0
