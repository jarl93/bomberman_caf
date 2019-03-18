
import numpy as np
from random import shuffle
from time import time, sleep
from collections import deque
#from tempfile import TemporaryFile
import os.path
import pickle


from settings import s, e

# ~ def MaxQAct(self):
	
	# ~ return 0.0;


# ~ def funtionQ(self.state,self.next_action,self):
    # ~ return self.reward + self.gamma*MaxQAct(self)

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

def RandomAct(self):
    self.logger.info('Pick action at random')
    self.next_action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN','WAIT', 'BOMB'], p=[.2, .2, .2, .2, .2, .0])


def SimpleAct(self):
    ## Gather information about the game state
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
    tempo = self.game_state['self'] 
    
def mappping(self):
    state = np.zeros((s.cols, s.rows),dtype=int)
    state[:,:] = self.game_state['arena']
 
    state[state==0.0] = 2
    state[state==-1.0] = 0 
    state[state==1.0] = 4
    
    	
    coins = self.game_state['coins']
    for coord in coins:
        state[coord] = self.values['COIN']
    x, y, name, b, score = self.game_state['self']
    state[x,y] = self.values['SELF']
    
    others = self.game_state['others']
    for i in range(len(others)):
	    state[others[i][0],others[i][1]] = self.values['OPPO']
	    #print("The ",i," corrd is: ", others[i][0]," , ",others[i][1])

    bombs = self.game_state['bombs']
    for i in range(len(bombs)):
	    state[bombs[i][0],bombs[i][1]] = self.values['BOMB']
	
    #print("The state are: \n", state)
    
    #Due the game symmetri, the state is transformed in order to show always the agent departing from pos (1,1)
    if self.IndexTrans == 1:
	    state = np.flip(state)
    if self.IndexTrans == 2:
	    state = np.fliplr(state)
    if self.IndexTrans == 3:
	    state = np.flipud(state)
	    
    
    return state
    
def setup(self):
	
	#set the Qvalues, to add experiences
    self.QValues = []
	# Reward at begining
    self.reward = 0
    #Hyperparemeters
    self.gamma = 0.95    # discount rate
    self.epsilon = 1.0  # exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.99999
    #Number next index state related
    self.NumSSig = 0
    # Load Dictionary
    self.StatesIndex = {
    }
    
    # if os.path.isfile('QDict.pickle'):
    #     with open('QDict.pickle', 'rb') as handle:
    #         self.StatesIndex = pickle.load(handle)
    #     print(len(self.StatesIndex))
    #


    #index, indicating if is nedded the state transformation
    self.IndexTrans = 0
    #Value of played episodios 
    self.nepisodios = 0
    
    
	# values for objects in the game
    self.values = {
        'WALL' : 0,#-1.0,
        'BOMB' : 1,#-0.5,
        'FREE' : 2,#0.0,
	    'SELF' : 3,#0.5,
        'CRATE': 4,#1.0,
        'COIN' : 5,#2.0,
        'OPPO' : 6,#3.0
    }
     

    self.logger.debug('Successfully entered setup code')
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0

    
def act(self):
    """

    :type self: object
    """

    #We define the IndexTrans, to denoted the transformation nedded in the states, in order to decreases by 4, the number of total states
    if self.game_state['step'] == 1:
        #print("Runing the episode: ", self.nepisodios)
        self.reward = 0.0
        x , y, _, _,_ = self.game_state['self']
        if x == 1 and y == 1:
            self.IndexTrans = 0
            self.logger.debug('No transformations in states')
            print("No transformations")
            # values for actions in the game
            self.VActions = {
                'RIGHT': 0,  # -1.0,
                'LEFT': 1,  # -0.5,
                'UP': 2,  # 0.0,
                'DOWN': 3,  # 0.5,
                'WAIT': 4,  # 1.0,
                'BOMB': 5,  # 2.0,
            }
        if x == 15 and y == 15:
            self.IndexTrans = 1
            self.logger.debug('Transpose transformations in states')
            print("Transpose transform")
            # values for actions in the game
            self.VActions = {
                'RIGHT': 1,  # -1.0,
                'LEFT': 0,  # -0.5,
                'UP': 3,  # 0.0,
                'DOWN': 2,  # 0.5,
                'WAIT': 4,  # 1.0,
                'BOMB': 5,  # 2.0,
            }
        if x == 1 and y == 15:
            self.IndexTrans = 2
            self.logger.debug('x reflect transformatios in states')
            print("x reflect transform")
            # values for actions in the game
            self.VActions = {
                'RIGHT': 0,  # -1.0,
                'LEFT': 1,  # -0.5,
                'UP': 3,  # 0.0,
                'DOWN': 2,  # 0.5,
                'WAIT': 4,  # 1.0,
                'BOMB': 5,  # 2.0,
            }
        if x == 15 and y == 1:
            self.IndexTrans = 3
            self.logger.debug('y refelct transformations in states')
            print("y refelct transform")
            # values for actions in the game
            self.VActions = {
                'RIGHT': 1,  # -1.0,
                'LEFT': 0,  # -0.5,
                'UP': 2,  # 0.0,
                'DOWN': 3,  # 0.5,
                'WAIT': 4,  # 1.0,
                'BOMB': 5,  # 2.0,
            }

    #We create the state, Sefl, boms, coins... etc.
    self.state = mappping(self)
    
    
    # In order to create the initial matrix Q, 
    if np.random.rand() <= self.epsilon:
	    SimpleAct(self)
	    if self.epsilon >  self.epsilon_min:
	        self.epsilon = self.epsilon*self.epsilon_decay
    else:
        RandomAct(self)
        self.epsilon = self.epsilon*self.epsilon_decay
    
    #print("Action ",self.next_action)
    #print("Next value act", self.VActions[self.next_action])
    
    
    
    IndexState = 0
    #Create a flat version of the state    
    self.stateFlat = np.asarray(self.state).reshape(-1)
    #A string version of the flat state   
    self.StrNumState = str(''.join(map(str,self.stateFlat)))

    #print(self.StatesIndex[StrNumState])
    #print(len(self.StatesIndex))
    #print(self.game_state['step'])
    #Qtemp[IndexState,IndexAction] = 0.1
    
    #print(Qtemp.shape)
    

def reward_update(self):
    rewardAct = 0.0
    if (e.MOVED_LEFT in self.events or e.MOVED_RIGHT in self.events or e.MOVED_UP in self.events or e.MOVED_DOWN in self.events or e.WAITED in self.events):
        # print("Delay punish -0.005")
        rewardAct += -0.5
    if e.INVALID_ACTION in self.events:
        # print("Invalid action punish -0.5")
        rewardAct += -0.05
        self.logger.info('Invalid Action, performed')
    if e.COIN_COLLECTED in self.events:
        rewardAct += 10
        # print("Coin recolected +0.5")
    if e.KILLED_SELF in self.events:
        rewardAct -= 100

        # print(self.reward)
    # We build the tupple, where we save all: timeSteps, State, action and reward(t+1)
    self.reward += rewardAct

    # From my dictionary, chek if the state already exist, and get the Index State.
    Qtemp = np.random.rand(1, 6) * 0.001
    #Qtemp = np.zeros([1,6])

    IndexAction = self.VActions[self.next_action]
    if self.StrNumState in self.StatesIndex:
        Qtemp = self.StatesIndex[self.StrNumState]
        if e.INVALID_ACTION in self.events:
            Qtemp[0, IndexAction] = 0.0
        else:
            Qtemp[0, IndexAction] += 0.1*np.absolute(rewardAct)
        self.StatesIndex[self.StrNumState] = Qtemp
        # print("Exist and the index related is: ",IndexState)
    else:
        if e.INVALID_ACTION in self.events:
            Qtemp[0, IndexAction] = 0.0
        else:
            Qtemp[0, IndexAction] += 0.1*np.absolute(rewardAct)
        self.StatesIndex[self.StrNumState] = Qtemp

    pass

def end_of_episode(self):
    self.nepisodios = self.nepisodios+ 1
    if self.nepisodios == s.n_rounds:
	    print(len(self.StatesIndex))
	
    TempValScore = self.game_state['self'][4]
    TempValSteps = self.game_state['step']
    TempValEp = self.epsilon
    self.logger.info(f'Score {TempValScore}')
    self.logger.info(f'NSteps {TempValSteps}')
    self.logger.info(f'Epsilon {TempValEp}')
    print(TempValScore,TempValSteps,TempValEp )  
    

    #We add the last index state at Dictionary. 
    self.state = mappping(self)
    self.stateFlat = np.asarray(self.state).reshape(-1)
    #A string version of the flat state   
    self.StrNumState = str(''.join(map(str,self.stateFlat)))
    Qtemp = np.zeros([1,6])
    self.StatesIndex[self.StrNumState] = Qtemp
    #print("XXX",self.StatesIndex[self.StrNumState])
    #self.StatesIndex[StrNumState] = self.NumSSig
    
    #print(self.state, "\n index+1" , self.NumSSig, "\n " , self.StatesIndex)
    #Save the Policy and Dictionary guide   -> dont save the ones who avoid the justice!
    with open('QDict.pickle', 'wb') as handle:
        pickle.dump(self.StatesIndex, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    #np.save('QGuess.npy', self.MatrixQ)    
    #print("For the state ", self.StatesIndex[StrNumState] ,"We have the values: ",self.MatrixQ[self.StatesIndex[StrNumState]])
    #print(self.StatesIndex)
    
    pass
