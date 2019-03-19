
import numpy as np
from time import sleep
from collections import deque

from settings import s, e


def distance_bfs(self, xa, ya, xo, yo, arena):
    queue = deque([(xa, ya)])
    visited = {}
    visited[(xa, ya)] = 0
    dist = -1
    while (len(queue) > 0):
        curr_x, curr_y = queue.popleft()

        if (curr_x == xo and curr_y == yo):
            dist = visited[(curr_x, curr_y)]
            break
        directions = [(curr_x, curr_y - 1), (curr_x, curr_y + 1), (curr_x - 1, curr_y), (curr_x + 1, curr_y)]
        for (xd, yd) in directions:
            d = (xd, yd)
            if (arena[d] == 0) and (not d in visited):
                queue.append(d)
                visited[d] = visited[(curr_x, curr_y)] + 1
    return dist


def get_region_valid(self, xa, ya, xo, yo, valid, arena):
    region = np.array([0, 0, 0, 0])
    region_valid = np.array([0, 0, 0, 0])
    directions = [(xa, ya - 1), (xa, ya + 1), (xa - 1, ya), (xa + 1, ya)]

    # upper
    if (xo == xa and yo < ya):
        region = np.array([1, 0, 0, 0])
    # lower
    if (xo == xa and yo > ya):
        region = np.array([0, 1, 0, 0])
    # left
    if (xo < xa and yo == ya):
        region = np.array([0, 0, 1, 0])
    # right
    if (xo > xa and yo == ya):
        region = np.array([0, 0, 0, 1])

    # upper-left
    if (xo < xa and yo < ya):
        region = np.array([1, 0, 1, 0])
    # lower-left
    if (xo < xa and yo > ya):
        region = np.array([0, 1, 1, 0])
    # lower-right
    if (xo > xa and yo > ya):
        region = np.array([0, 1, 0, 1])
    # upper-right
    if (xo > xa and yo < ya):
        region = np.array([1, 0, 0, 1])

    region_valid = valid & region

    if (np.count_nonzero(region_valid) == 0):
        list_valid = []
        for bit in range(4):
            valid_candidate = np.array([0, 0, 0, 0])
            if valid[bit] == 1:
                valid_candidate[bit] = 1
            list_valid.append(valid_candidate)

        if len(list_valid) > 0:
            idx_valid = np.random.choice(len(list_valid))
            region_valid = list_valid[idx_valid]

    if (np.count_nonzero(region_valid) == 2):
        d_min = 1000000
        idx_min = 0
        for i in range(4):
            if region_valid[i] == 1:
                x_curr, y_curr = directions[i]
                d_curr = distance_bfs(self, x_curr, y_curr, xo, yo, arena)
                if d_curr < d_min:
                    idx_min = i
                    d_min = d_curr

        region_valid = np.array([0, 0, 0, 0])
        region_valid[idx_min] = 1

    return region_valid


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


def mappping(self):
    # State definition

    # Gather information about the game state
    arena = self.game_state['arena']
    # aux_arena = np.zeros((s.rows, s.cols))
    # aux_arena[:,:] = arena
    x, y, _, bombs_left, score = self.game_state['self']
    bombs = self.game_state['bombs']
    bombs_xys = [(x, y, t) for (x, y, t) in bombs]
    # others = [(x,y) for (x,y,n,b,s) in self.game_state['others']]
    coins = self.game_state['coins']
    explosion_map = self.game_state['explosions']
    bomb_dic = {}
    bomb_map = np.zeros(arena.shape)

    # map for bombs
    for (xb, yb, t) in bombs_xys:
        arena[xb, yb] = 2
        # when other agent mark arena as well
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < arena.shape[0]) and (0 < j < arena.shape[1]) and (arena[(i, j)] != -1):
                bomb_map[(i, j)] = t
                if (i, j) in bomb_map:
                    bomb_dic[(i, j)].append(t)
                else:
                    bomb_dic[(i, j)] = [t]

    # General case
    # state = np.zeros(32, dtype = int)

    # Coins case
    state = np.zeros(14)

    # 0. UP ->    (x  , y-1)
    # 1. DOWN ->  (x  , y+1)
    # 2. LEFT ->  (x-1, y  )
    # 3. RIGHT -> (x+1, y  )

    # 4 bits for valid position
    valid = np.array([0, 0, 0, 0])

    directions = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
    for i in range(4):
        d = directions[i]
        if (arena[d] == 0 and explosion_map[d] <= 1):
            valid[i] = 1

    state[:4] = valid

    list_dist = []
    # 4 bits for nearest coin
    for (xc, yc) in coins:
        # aux_arena[(xc,yc)] = 2
        list_dist.append(distance_bfs(self, x, y, xc, yc, arena))

    # aux_arena[x,y] = 5

    # self.logger.debug(f'ARENA:\n {aux_arena}')

    # self.logger.debug(f'Distance coins: {list_dist}')
    if len(list_dist) > 0:
        min_dist = np.min(np.array(list_dist))
    if len(list_dist) > 0 and min_dist > -1:
        idx_min = np.argmin(np.array(list_dist))
        x_min, y_min = coins[idx_min]
        state[4:8] = get_region_valid(self, x, y, x_min, y_min, valid, arena)

    # DANGER
    free_cells = np.zeros(4)
    danger = 0
    if bomb_map[(x, y)] > 0:
        danger = 1
        time = bomb_map[(x, y)]
        for i in range(4):
            x_next, y_next = directions[i]
            if arena[(x_next, y_next)] == 0:
                free_cells[i] = get_free_cells(self, x_next, y_next, arena, bomb_dic, explosion_map, time)

    # Number of crates

    number_crates = 0
    for (i, j) in [(x + h, y) for h in range(-3, 4)] + [(x, y + h) for h in range(-3, 4)]:
        if (0 < i < arena.shape[0]) and (0 < j < arena.shape[1]) and (arena[(i, j)] == 1):
            number_crates += 1

    state[8] = danger
    state[9:13] = free_cells
    state[13] = number_crates

    self.logger.debug(f'STATE VALID: {state[:4]}')
    self.logger.debug(f'STATE COINS: {state[4:8]}')
    self.logger.debug(f'STATE DANGER: {state[8]}')
    self.logger.debug(f'STATE ESCAPE: {state[9:13]}')
    self.logger.debug(f'STATE CRATES: {state[13]}')

    print("STATE VALID:",state[:4])
    print("STATE COINS:",state[4:8])
    print("STATE DANGER:",state[8])
    print("STATE ESCAPE:",state[9:13])
    print("STATE CRATES",state[13])
    return state


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
    self.num_actions = 6

    # action index (this is really the variable used as action)
    self.idx_action = 4

    # STATE

    # state size
    self.state_size = 13

    self.state = np.zeros(self.state_size)

    # next_state defined as state
    self.next_state = np.zeros(self.state_size)



    # NN for playing
    # self.model = load_model_NN(self)

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
        'VALID': -2,
        'DIE': -1500
    }




def act(self):
    # Gather information about the game state
    print("\n",mappping(self),"\n")

    self.logger.info('Pick action according to pressed key')
    self.next_action = self.game_state['user_input']

def reward_update(self):

    if e.CRATE_DESTROYED in self.events:
        NCrates =  list(self.events).count(9)
        print("Crates Destroyed: \n", NCrates)
    pass

def learn(self):
    if e.CRATE_DESTROYED in self.events:
        print("Events: \n",self.events)
    pass

    pass
