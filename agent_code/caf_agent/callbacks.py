
import numpy as np
from random import shuffle
from time import time, sleep
from collections import deque

from settings import s, e


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
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    
    # Init the 6 possible actions
    # 0. UP ->    (x  , y-1)
    # 1. DOWN ->  (x  , y+1)
    # 2. LEFT ->  (x-1, y  )
    # 3. RIGHT -> (x+1, y  )
    # 4. WAIT ->  (x  , y  )
    # 5. BOMB ->    ?
    self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']
    # Init matrix R and Q with zeros
    # a state is defined by the coordinates and the countdown of a bomb in this position
    # there are 6 possible actions
    self.Q = np.zeros((s.cols, s.rows, 2, 6))
    self.R = np.zeros((s.cols, s.rows, 2, 6))
    
    # Init gamma = 0.8 for no good reason
    self.gamma = 0.8
    
    # Init best_value
    self.best_value = -1
    
    # Init idx of action chosen
    # Since the default action is 'WAIT', the default of idx_action is 4
    self.idx_action = 4
    
    # Counters
    self.random = 0


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
    self.logger.info('Picking action according to rule set')

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
    
    # Init matrix R with the posibles states
    # a state is defined by the coordinates and the countdown of a bomb in this position
    # there are 6 possible actions
    # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: WAIT, 5: BOMB
    R = np.zeros((s.cols, s.rows, 2, 6))
    R.fill(-1)
    self.logger.debug(f'Shape of R: {R.shape}')
    
    # (state, action) which leads to an empty tile from an empty tile
    for x0 in range(1,16):
        for y0 in range(1,16):
            directions = [(x0,y0-1), (x0,y0+1), (x0-1,y0), (x0+1,y0), (x0,y0)]
            for i in range(0,len(directions)):
                d = directions[i]
                # i is the action taken 
                if (arena [(x0,y0)]==0 and arena[d] == 0):
                    R[x0,y0,0,i] = 0
                
    # (state, action) which leads to get a coin
    for (x0,y0) in coins:
        directions = [(x0,y0+1), (x0,y0-1), (x0+1,y0), (x0-1,y0)]
        for i in range(0,len(directions)):
            d = directions[i]
            if (arena[d] == 0):
                x1, y1 = d
                R[x1,y1,0,i] = 100
                
    #self.logger.debug(f'COINS:\n {coins}')
    #self.logger.debug(f'ARENA:\n {arena}')
    #self.logger.debug(f'R UP:\n {R[:,:,0,0]}')
    #self.logger.debug(f'R DOWN:\n {R[:,:,0,1]}')
    #self.logger.debug(f'R LEFT:\n {R[:,:,0,2]}')
    #self.logger.debug(f'R RIGHT:\n {R[:,:,0,3]}')
    
    self.R = R
    next_state_actions = []
    
    #At the moment just take into account 5 actions
    for a in range (5):
        next_state_actions = next_state_actions + [(x,y-1,0,a),(x,y+1,0,a),(x-1,y,0,a),(x+1,y,0,a),(x,y,0,a)]
    
    # Get all the possible values of Q based on the next_state_actions list
    list_Qvalues = [self.Q[nsa] for nsa in next_state_actions]
    #Define a new list with values of Q and the next_state_actions tuples in order to sort them
    list_nsa_Qvalues = list(zip(list_Qvalues, next_state_actions))
    list_nsa_Qvalues.sort(reverse=True)
    
    
    self.logger.debug(f'list_Qvalues: {list_nsa_Qvalues}')
    
    
    #Get the first value to see if all values are less or equal than zero
    length_Qvalues = len(list_nsa_Qvalues)
    first_value, _  = list_nsa_Qvalues[0]
    
    self.logger.debug(f'first_value: {first_value}')
    
    # Init the valid directions
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y), (x,y)]
    
    # first_value <= 0 implies that all the values are less or equal than zero,
    # hence the agent select a random possible action
    if (first_value <= 0):
        # Check which moves make sense at all
        valid_tiles, valid_actions, idx_action = [], [], []

        # Go over all the possible directions
        for d in directions:
            if (arena[d] == 0):
                valid_tiles.append(d)
        
        # Save the valid actions to be chosen
        if (x,y-1) in valid_tiles: 
            valid_actions.append('UP')
            idx_action.append(0)
            
        if (x,y+1) in valid_tiles:
            valid_actions.append('DOWN')
            idx_action.append(1) 
            
        if (x-1,y) in valid_tiles:
            valid_actions.append('LEFT')
            idx_action.append(2)
            
        if (x+1,y) in valid_tiles: 
            valid_actions.append('RIGHT')
            idx_action.append(3)
            
        if (x,y)in valid_tiles: 
            valid_actions.append('WAIT')
            idx_action.append(4)
        
        # Get a random action from the valid actions
        idx_rand = np.random.randint(len(valid_actions))
        a_new = valid_actions[idx_rand]
        self.idx_action = idx_action[idx_rand]
        self.best_value = first_value
        self.random = self.random +1 
    else:
        # Here, implies that there is an action which the value of Q is greater than 0
        # Select the best POSSIBLE action by going over the list of Q values  
        for i in range(length_Qvalues):
            Q_value, nsa_tuple = list_nsa_Qvalues[i]
            _, _, _, idx_a = nsa_tuple
            d = directions[idx_a] 
            # Check if a valid action
            if (arena[d]==0):
                a_new = self.actions[idx_a]
                self.idx_action = idx_a
                self.best_value = Q_value
                break
        
    self.next_action = a_new
    
    
             
'''
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
'''


def reward_update(self):
    """Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s)')
        
    x, y, _, bombs_left, score = self.game_state['self']
    
    self.Q[x,y,0,self.idx_action] = self.R[x,y,0,self.idx_action] + self.gamma*self.best_value
    
    # Since a coin was found, the matrix Q should be updated accordingly
    if e.COIN_COLLECTED in self.events:
        self.logger.debug("COIN COLLECTED")
        directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y), (x,y)]
        for d in directions:
            x0, y0 = d
            self.Q[x0,y0,0,:] = 0.0
    

def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
    self.logger.debug(f'Random: {self.random}')
    self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')
