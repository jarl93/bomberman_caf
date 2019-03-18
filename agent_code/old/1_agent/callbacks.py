
import numpy as np
from random import shuffle
from time import time, sleep
from collections import deque
#from tempfile import TemporaryFile
import os.path
import pickle
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


from decimal import Decimal


from settings import s, e

def ReggForestSetUp(self):
    #we create the random data, in order to use it, in regression forest
    X_full = np.zeros((self.NumData, 289))
    y_full = np.zeros((self.NumData, 6))

    for i in range(self.NumData):
        StateStr, QVals = random.choice(list(self.StatesIndex.items()))
        StateFlat = np.array(list(StateStr), dtype=float)
        X_full[i] = StateFlat
        y_full[i] = QVals

    max_depth = 30
    self.regr_rf = RandomForestRegressor(n_estimators=100, max_depth=max_depth, random_state=2)
    self.regr_rf.fit(X_full, y_full)
    #print (X_full, y_full)

def PredictAct(self):
    X_to_predict = self.stateFlat.astype(float).reshape(1, -1)

    #IndexAction = self.VActions[self.next_action]
    if self.flagforest:
        act_pred = self.regr_rf.predict(X_to_predict)

        IndAct = np.argmax(act_pred)
        print(act_pred)

        self.next_action = self.ActMap[IndAct]
        print("Dont exist, and run Regression forrest: ", self.next_action)
    else:
        RandomAct(self)
        self.logger.debug(f'Finaly Random action performed ')




def RandomAct(self):
    self.logger.info('Pick action at random')
    self.next_action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN','WAIT', 'BOMB'], p=[.2, .2, .2, .2, .2, .0])
  
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
	
    # Reward at begining
    self.reward = 0
    #Hyperparemeters
    self.gamma = 0.95    # discount rate
    self.epsilon = 1.0  # exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.99999
    self.alpha = 0.01  # Learning rate
        
    #Parameters to regression forest
    self.NumData = 1000
    # Load Dictionary
    self.StatesIndex = {
    }
    self.StatesIndexEpisode = {
    }


    self.IndexTrans = 0
    #Value of played episodios 
    self.nepisodios = 0
    #flag to kown if is abalible the regression forest
    self.flagforest = 0
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
     
    # values for objects in the game
    self.VActions = { 
        'RIGHT': 0,#-1.0,
        'LEFT' : 1,#-0.5,
        'UP'   : 2,#0.0,
	    'DOWN' : 3,#0.5,
        'WAIT' : 4,#1.0,
        'BOMB' : 5,#2.0,
    }

def act(self):
    """

    :type self: object
    """
    #We define the IndexTrans, to denoted the transformation nedded in the states, in order to decreases by 4, the number of total states
    if self.game_state['step'] == 1:
        self.EpisodeList = []
        self.reward = 0.0
        self.StatesIndexEpisode = {
        }
        x , y, _, _,_ = self.game_state['self']
        if x == 1 and y == 1:
            self.IndexTrans = 0
            self.logger.debug('No transformations in states')
            self.ActMap = ['RIGHT', 'LEFT', 'UP', 'DOWN', 'WAIT', 'BOMB']
            print("No transformations")
            self.VActions = dict(RIGHT=0, LEFT=1, UP=2, DOWN=3, WAIT=4, BOMB=5)
        if x == 15 and y == 15:
            self.IndexTrans = 1
            self.logger.debug('Transpose transformations in states')
            self.ActMap = ['LEFT', 'RIGHT', 'DOWN', 'UP', 'WAIT', 'BOMB']
            print("Transpose transform")
            self.VActions = dict(RIGHT=1, LEFT=0, UP=3, DOWN=2, WAIT=4, BOMB=5)
        if x == 1 and y == 15:
            self.IndexTrans = 2
            self.logger.debug('x reflect transformatios in states')
            self.ActMap = ['RIGHT', 'LEFT', 'DOWN', 'UP', 'WAIT', 'BOMB']
            print("x reflect transform")
            self.VActions = dict(RIGHT=0, LEFT=1, UP=3, DOWN=2, WAIT=4, BOMB=5)
        if x == 15 and y == 1:
            self.IndexTrans = 3
            self.logger.debug('y refelct transformations in states')
            self.ActMap = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']
            print("y reflect transform")
            self.VActions = dict(RIGHT=1, LEFT=0, UP=2, DOWN=3, WAIT=4, BOMB=5)


    #We create the state, Sefl, boms, coins... etc.
    #All states, after transformations
    self.state = mappping(self)
    #IndexState = 0
    #Create a flat version of the state
    self.stateFlat = np.asarray(self.state).reshape(-1)
    # A string version of the flat state
    self.StrNumState = str(''.join(map(str, self.stateFlat)))

    #
    # if self.StrNumState in self.StatesIndex:
    #     qtemp = self.StatesIndex[self.StrNumState]
    #     print("The state in time : ", self.game_state['step'], "exist, Q is:\n",qtemp)
    #


    # In order to create the initial matrix Q,
    if np.random.rand() <= (1 - self.epsilon):
        self.logger.debug(f'Agent action performed ')
        PredictAct(self)
        if self.epsilon >  self.epsilon_min:
            self.epsilon = self.epsilon*self.epsilon_decay
    else:
        self.logger.debug(f'Random action performed ')
        RandomAct(self)
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay


    #From my dictionary, chek if the state already exist, and get the Index State.
    # Qtemp = np.zeros([1,6])
    #
    # IndexAction = self.VActions[self.next_action]
    # if self.StrNumState in self.StatesIndex:
    #     Qtemp = self.StatesIndex[self.StrNumState]
    #     Qtemp[0,IndexAction] += 0.01
    #     self.StatesIndex[self.StrNumState] = Qtemp
    #     #print("Exist and the index related is: ",IndexState)
    # else:
    #     Qtemp[0,IndexAction] += 0.01
    #     self.StatesIndex[self.StrNumState] = Qtemp

    #print("1Time step: ", self.game_state['step'])
    #print("The next move: ", self.next_action)

	

def reward_update(self):
    rewardAct = 0.0
    if (e.MOVED_LEFT in self.events or e.MOVED_RIGHT  in self.events or e.MOVED_UP in self.events or e.MOVED_DOWN in self.events or e.WAITED in self.events):
        #print("Delay punish -0.005")
        rewardAct += -0.005
    if e.INVALID_ACTION in self.events:
        #print("Invalid action punish -0.5")
        rewardAct += -0.5
        self.logger.info('Invalid Action, performed')
    if e.COIN_COLLECTED in self.events:
        rewardAct += 0.3
        #print("Coin recolected +0.5")

        #print(self.reward)
    #We build the tupple, where we save all: timeSteps, State, action and reward(t+1)
    self.reward += rewardAct
    timeTemp = self.game_state['step']-2  #To set the t_0 = 1
    # We create the state, Sefl, boms, coins... etc.
    self.state1 = mappping(self)
    # IndexState = 0
    # Create a flat version of the state
    self.stateFlat1 = np.asarray(self.state1).reshape(-1)
    # A string version of the flat state
    self.StrNumState1 = str(''.join(map(str, self.stateFlat1)))
    tuptemp = (timeTemp,self.StrNumState,self.StrNumState1, self.next_action ,rewardAct)
    self.EpisodeList.append(tuptemp)
    #print(self.EpisodeList[0])

    pass

def end_of_episode(self):
    self.nepisodios = self.nepisodios+ 1
    print("End of Round: ", self.nepisodios)
    if self.nepisodios == s.n_rounds:
        print("End of episodes")
	
    TempValScore = self.game_state['self'][4]
    TempValSteps = self.game_state['step']
    TempValEp = self.epsilon
    self.logger.debug(f'/n SCORE {TempValScore}')
    self.logger.debug(f'NSTEPS {TempValSteps}')
    self.logger.debug(f'EPSILON {TempValEp}')
    print("Score,Steps, Epsilon: ",TempValScore,TempValSteps,TempValEp )

   # step, state, state1, action, reward = list(self.StatesIndexEpisode.items())

    #print(list(self.StatesIndexEpisode.items()))
    #if state in self.StatesIndex:



    #Guessing the Q values, randomly.
    TotalSteps = 0
    for (step, state, state1, action, reward) in self.EpisodeList:
        #print("Step", step)
        # print("state ", state)
        # print("state1 ", state1)
        # print("action ", action)
        # print("reward ", reward)
        #
        if state in self.StatesIndex:
            aa=0
            #print(self.StatesIndex[state])
        else:
            #print("Generating state:")
            Qtemp = np.random.rand(1, 6) * 0.01  # np.zeros([1, 6])
            Qtemp[0,5] = 0.0
            self.StatesIndex[state] = Qtemp
            #print(self.StatesIndex[state])
        TotalSteps += 1
    #We choose fill the final state with
    self.StatesIndex[self.StrNumState1] = np.zeros([1, 6])

    #print(self.StatesIndex[self.StrNumState1])

    for (step, state, state1, action, reward) in self.EpisodeList:
        IndexAction = self.VActions[action]
        #assert isinstance(state, object)
        #print(self.StatesIndex[state])
        Qtemp = self.StatesIndex[state]
        Qtemp1 = self.StatesIndex[state1]
        valQSA = Qtemp[0,IndexAction]
        indextemp = np.argmax(Qtemp1)
        maxQtem1 = Qtemp1[0,indextemp]
        valRGamma = reward + self.gamma*maxQtem1 - valQSA
        Qtemp[0,IndexAction] = valQSA + self.alpha*valRGamma
        self.StatesIndex[state] = Qtemp
        #print("Exist and the index related is: ",IndexState)


    if len(self.StatesIndex) > self.NumData :
        self.flagforest = 1
        ReggForestSetUp(self)

    with open('QDict.pickle', 'wb') as handle:
        pickle.dump(self.StatesIndex, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pass
