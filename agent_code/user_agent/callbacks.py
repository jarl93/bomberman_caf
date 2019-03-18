
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
        for i in range(4):
            region_valid_min = np.array([0, 0, 0, 0])
            if region_valid[i] == 1:
                x_curr, y_curr = directions[i]
                d_curr = distance_bfs(self, x_curr, y_curr, xo, yo, arena)
                if d_curr < d_min:
                    region_valid_min[i] = 1
                    d_min = d_curr

        region_valid = region_valid_min

    return region_valid


def mappping(self):
    # State definition

    # Gather information about the game state
    arena = self.game_state['arena']
    # aux_arena = np.zeros((s.rows, s.cols))
    # aux_arena[:,:] = arena
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

    # General case
    # state = np.zeros(32, dtype = int)

    # Coins case
    state = np.zeros(8)

    # 0. UP ->    (x  , y-1)
    # 1. DOWN ->  (x  , y+1)
    # 2. LEFT ->  (x-1, y  )
    # 3. RIGHT -> (x+1, y  )

    # 4 bits for valid position
    valid = np.array([0, 0, 0, 0])

    directions = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
    for i in range(4):
        d = directions[i]
        if (arena[d] == 0):
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
        state[4:] = get_region_valid(self, x, y, x_min, y_min, valid, arena)

    self.logger.debug(f'STATE VALID: {state[:4]}')
    self.logger.debug(f'STATE COINS: {state[4:]}')

    return state


def setup(self):
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

    # self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']

    # Case to just collect the coins
    self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    # Init map of actions
    self.map_actions = {
        'UP': 0,
        'DOWN': 1,
        'LEFT': 2,
        'RIGHT': 3,
        'WAIT': 4,
        'BOMB': 5
    }
    # STATE

    # state size
    self.state_size = 12

    self.state = np.zeros(self.state_size)

    # next_state defined as state
    self.next_state = np.zeros(self.state_size)

    #pass

def act(self):
    # Gather information about the game state
    print(mappping(self))

    self.logger.info('Pick action according to pressed key')
    self.next_action = self.game_state['user_input']

def reward_update(self):
    pass

def learn(self):
    pass
