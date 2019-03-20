
import numpy as np
import random
from time import time, sleep
from collections import deque

from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.layers.core import Dense, Dropout
from keras.activations import relu
from keras.models import load_model

from settings import s, e

def build_model(self):
    
    # Neural network for Deep-Q learning Model
    
    model = Sequential()
    model.add(Dense(self.state_size, input_dim=self.state_size, activation='relu'))
    model.add(Dropout(0.15))
    
    model.add(Dense(self.state_size, activation='relu'))
    model.add(Dropout(0.15))
    
    model.add(Dense(self.state_size, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(self.num_actions, activation='softmax'))
    
    opt = Adam(self.learning_rate)
    model.compile(loss='mse', optimizer=opt)
    
    return model

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


def distance_bfs (self, xa, ya, xo, yo, arena):
    queue  = deque([(xa,ya)])
    visited = {}
    visited[(xa,ya)] = 0
    dist = 1000000
    while (len(queue)>0):
        curr_x, curr_y = queue.popleft()
        
        if (curr_x == xo and curr_y == yo):
            dist = visited[(curr_x, curr_y)]
            break
        directions = [(curr_x, curr_y-1), (curr_x, curr_y+1), (curr_x-1, curr_y), (curr_x+1, curr_y)]  
        for (xd, yd) in directions:
            d = (xd, yd)
            if (arena[d] == 0) and (not d in visited):
                queue.append(d)
                visited[d] = visited[(curr_x, curr_y)] + 1
    return dist

def mappping(self):
    
    # State definition
    
    # Gather information about the game state
    arena = self.game_state['arena']
    #aux_arena = np.zeros((s.rows, s.cols))
    #aux_arena[:,:] = arena
    x, y, _, bombs_left, score = self.game_state['self']
    bombs = self.game_state['bombs']
    bomb_xys = [(x,y,t) for (x,y,t) in bombs]
    #others = [(x,y) for (x,y,n,b,s) in self.game_state['others']]
    coins = self.game_state['coins']
    explosion_map = self.game_state['explosions']
    
    #Get optimal distance:
    if self.game_state['step'] == 1:
        self.distance_coins_total = get_distance_coins(self, x, y, arena, coins)
        
    bomb_map = np.zeros(arena.shape)
    
    for (xb, yb, t) in bombs:            
        vec_dir = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for v in vec_dir:
            hx, hy = v
            for i in range(0, 4-t):
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
    state = np.zeros(self.state_size)
    
    # 0. UP ->    (x  , y-1)
    # 1. DOWN ->  (x  , y+1)
    # 2. LEFT ->  (x-1, y  )
    # 3. RIGHT -> (x+1, y  )
    
    # 4 bits for valid position
    valid = np.array([0,0,0,0])
    
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    for i in range(4):
        d = directions[i]
        if (arena[d] == 0 and explosion_map[d] <= 1 and bomb_map[d] == 0):
            valid[i] = 1
    
    state[:4] = valid 
    
    list_dist = []
    # 4 bits for nearest coin
    for (xc, yc) in coins:
        #aux_arena[(xc,yc)] = 2
        list_dist.append( distance_bfs(self, x, y, xc, yc, arena) )
    
    #aux_arena[x,y] = 5
    
    #self.logger.debug(f'ARENA:\n {aux_arena}')
    
    #self.logger.debug(f'Distance coins: {list_dist}')
    if len(list_dist) > 0:
        min_dist = np.min(np.array(list_dist))
        if min_dist < 1000000:
            dist_min = 1000000
            idx_min = np.argmin(np.array(list_dist))
            x_min, y_min = coins[idx_min]
            for i in range (4):
                x_curr, y_curr = directions[i]
                dist_curr = distance_bfs(self, x_curr, y_curr, x_min, y_min, arena)
                if dist_curr < dist_min:
                    dist_curr = dist_min
                    idx_direction = i
    
            state[4+idx_direction] = 1
        
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
    state[8] = number_crates
    
    self.logger.debug(f'STATE VALID: {state[:4]}')
    self.logger.debug(f'STATE COINS: {state[4:8]}')
    self.logger.debug(f'STATE CRATES: {state[8]}')
    

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
    #else:      #In case of crates, we dont mind about optimazed path
    #    reward += self.reward_list['VALID']

    # In order to incentevi the optimal number of crates destroyed per Bomb dropped:
    # We give a reward proportional to NCrates*NCrates.
    if e.CRATE_DESTROYED in self.events:
        NCrates =  list(self.events).count(9)
        self.number_crates_destroyed += NCrates
        reward += NCrates*NCrates*self.reward_list['CRATE_DESTROYED']
        self.logger.debug(NCrates ,"DESTROYED")

    if e.COIN_FOUND in self.events:
        reward += self.reward_list['COIN_FOUND']
        self.logger.debug("COIN_FOUND")

    if e.KILLED_SELF in self.events:
        reward += self.reward_list['KILLED_SELF']
        self.logger.debug("KILLED_SELF")
        self.actions_killed_self += 1

    if e.BOMB_DROPPED in self.events:
        self.actions_bomb_dropped += 1
        reward += self.reward_list['BOMB_DROPPED']
        self.logger.debug("DROP_BOMB")
    
    if self.dead_zone[x,y] > 0:
        self.actions_dead_zone += 1
        reward += self.reward_list['DEAD_ZONE']
    
    
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
    
    # Decrease epsilon each time step
    update_epsilon(self)
    
    #Save the model every model_persistence_update_frequency steps
    if self.total_steps % self.persistence_update_frequency == 0:
        save_model_NN(self)

def save_model_NN(self):
    self.model.save(self.model_path)
    
def load_model_NN(self):
    return load_model(self.model_path)        
        
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
    
    #ACTIONS
    
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
    self.num_actions = 6
    
    # Case to just collect the coins
    # self.num_actions = 4
    
    # action index (this is really the variable used as action)
    self.idx_action = 0
    
    #STATE
    
    # state size
    self.state_size = 9

    self.state = np.zeros(self.state_size)
    
    # next_state defined as state
    self.next_state = np.zeros(self.state_size)
    
    # dead zone
    self.dead_zone = np.zeros((s.rows, s.cols))
       
    
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
    self.exploration_steps = 150000
    
    # Minimum value of epsilon, after this value
    # epsilon does not decrease anymore
    self.epsilon_min = 0.1
    
    # This hyperparameter is to decrease the number 
    # of explorations as the agent gets better
    
    self.epsilon_decay = (self.epsilon - self.epsilon_min)/self.exploration_steps
    
    #Exploration rate when playing
    self.epsilon_play = 0.02
    
    self.learning_rate = 0.0001
    
    # This hyperparameter controls how often the NN is saved
    self.persistence_update_frequency = 1000

    #MEMORY
    
    # List for the previous experiences
    # It's a deque because once the maxlen is reached, the oldest
    # experiences are forgotten (the newest experiences is what matters)
    # (state, action, reward, next_action, done)
    self.memory = deque(maxlen= self.memory_size)
    
    #MODEL
    self.model_path = './agent_code/agent_v1_simp/model.h5'

    # NN for training
    self.model = build_model(self)
    
    # NN for playing
    #self.model = load_model_NN(self)
      
    #REWARDS
    
    # Reward accumulated for every 4 frames
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
            'DEAD_ZONE': -40,
            'VALID' : -2,
            'DIE' : -1500,
            'COIN_FOUND' :  20,
            'KILLED_SELF' : -500,
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



    #Lists
    self.list_reward = []
    self.list_score = np.zeros(10)
    self.list_invalid_actions = []
    self.list_total_actions = []

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
    if np.random.rand() <= self.epsilon or len(self.memory) < self.replay_minimum_size:
        self.logger.info('Picking RANDOM action')
        idx_random = np.random.randint(self.num_actions)
        self.next_action =  self.actions[idx_random]
        self.idx_action = idx_random
        # Increase the number of random actions that has been taken
        self.actions_taken_random += 1
    #Exploitation
    else:
        start_time1Act = time()
        self.logger.info('Picking action according to the MODEL')
        q_values = self.model.predict( self.state.reshape((1, self.state_size)) )
        self.idx_action = np.argmax(q_values[0])
        self.next_action = self.actions[self.idx_action]
        # Increase the number of actions that has been taken based on the model
        self.actions_taken_model += 1
        # Flag, if the action belongs to the model
        self.flag_actions_taken_model = 1
        self.q_mean += np.sum(q_values[0])/self.num_actions  ##neu
        #time end action
        self.elapsed_time_action += time() - start_time1Act


def reward_update(self):
    """Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s)')
    
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

    total_actions = self.actions_taken_random + self.actions_taken_model
    self.q_mean /= total_actions
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
    #self.logger.debug(f'A-Simple: {self.actions_taken_simple}')
    self.logger.debug(f'A-Model: {self.actions_taken_model}')
    self.logger.debug(f'A-InvalidModel: {self.actions_taken_model_invalid}')
    self.logger.debug(f'A-DeadZone: {self.actions_dead_zone}')
    self.logger.debug(f'A-Total: {total_actions}')
    self.logger.debug(f'T-Action: {self.elapsed_time_action/total_actions}')
    self.logger.debug(f'T-TimeEpisodes: {elapsed_time} :s')
    
    if self.actions_taken_model > 0:
        self.logger.debug(f'T-Model: {self.elapsed_time_model/(self.actions_taken_model)}')
    else:
        self.logger.debug("T-Model: -1")
        
    if self.actions_taken_model > 0:
        self.logger.debug(f'M-Model_Invalid/ModelAct: {(self.actions_taken_model_invalid)/(self.actions_taken_model)}')
    else:
        self.logger.debug("M-Model_Invalid/ModelAct: -1")
        
    if self.distance_coins_total > 0:
        self.logger.debug(f'M-Steps/OptimalDistance: {total_actions/self.distance_coins_total}')
    else:
        self.logger.debug("M-Steps/OptimalDistance: -1")
        
    if self.actions_bomb_dropped > 0:    
        self.logger.debug(f'M-KilledSelfRate: {self.actions_killed_self / self.actions_bomb_dropped}')
        self.logger.debug(f'M-Crates/BombsDroped: {self.actions_killed_self / self.actions_bomb_dropped}')
    else:
        self.logger.debug("M-KilledSelfRate: -1")
        self.logger.debug("M-Crates/BombsDroped: -1")
        
    self.logger.debug(f'QMean: {self.q_mean}')


    self.list_reward.append(self.reward_episode)
    self.list_score [score] += 1
    self.list_invalid_actions.append(self.actions_invalid)
    self.list_total_actions.append(total_actions)
    
    if (self.episodes % 100 == 0):
        self.logger.debug(f'JARL-List rewards:\n {self.list_reward}')
        self.logger.debug(f'JARL-List score:\n {self.list_score}')
        self.logger.debug(f'JARL-List invalid:\n {self.list_invalid_actions}')
        self.logger.debug(f'JARL-List total:\n {self.list_total_actions}')
        self.list_reward = []
        self.list_score = np.zeros (10)
        self.list_invalid_actions = []
        self.list_total_actions = []


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