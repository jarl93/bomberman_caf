
import numpy as np
from time import sleep
from collections import deque

from settings import s, e

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

    # if self.game_state['step'] == 1:
    #     self.distance_coins_total = get_distance_coins(self, x, y, arena, coins)

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
        self.logger.debug("TIE")

        for i in list_idx:
            state_coins[i] *= np.random.uniform(0.7, 1)

    state[4:] = state_coins
    self.logger.debug(f'POSITION:\n {x, y}')
    self.logger.debug(f'STATE VALID: {state[:4]}')
    self.logger.debug(f'STATE COINS: {state[4:]}')

 #   print(state)

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
