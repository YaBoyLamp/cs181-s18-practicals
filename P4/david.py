# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

from SwingyMonkey import SwingyMonkey
from collections import defaultdict
from matplotlib import pyplot as plt

class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.q = defaultdict(lambda: 0)
        self.alpha = 0.5
        self.gamma = 0.9
        self.epsilon = 0.1
        self.is_low_gravity = None
        self.count = 0
        self.epochs = 1
        self.gravitys = []

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.is_low_gravity = None
        self.epochs += 1

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        self.count += 1
        new_action = 0

        # swing on the first epoch
        if self.last_state == None:
            self.last_action = 0
            self.last_state = state
            return 0
        
        # infer gravity
        if self.is_low_gravity == None:
            self.is_low_gravity = state['monkey']['vel'] == -1
            self.gravitys.append(state['monkey']['vel'] == -1)

        # transform states into our specific state representation and update q value for the last state action pair
        ls = (self.last_state['tree']['dist'] / 70, self.last_state['tree']['top'] / 70, self.last_state['monkey']['vel'] / 10, self.last_state['monkey']['top'] / 70, self.is_low_gravity)
        s = (state['tree']['dist'] / 70, state['tree']['top'] / 70, state['monkey']['vel'] / 10, state['monkey']['top'] / 70, self.is_low_gravity)
        self.q[(ls, self.last_action)] = ((1 - self.alpha) * self.q[(ls, self.last_action)]) + (self.alpha * ((self.last_reward) + (self.gamma * max([self.q[(s,0)], self.q[(s,1)]]))))

        # always jump if the monkey is about to fall off the bottom and always swing if the monkey is about to rise off the top
        if state['monkey']['top'] + state['monkey']['vel'] < 0:
            self.last_action = 1
            self.last_state = state 
            return 1
        if state['monkey']['bot'] + state['monkey']['top'] > 400:
            self.last_action = 0
            self.last_state = state
            return 0

        # change hyperparameters based on a schedule
        if self.epochs > 250:
            self.epsilon = 0.05
        if self.epochs > 500:
            self.epsilon = 0.025
        if self.epochs > 900:
            self.epsilon = 0
        if self.epochs > 500:
            self.alpha = 0.25
        if self.epochs > 750:
            self.alpha = 0.1


        # epsilon greedy exploration vs explotation decision
        if npr.rand() < self.epsilon:
            new_action = 1 if npr.rand() < 0.1 else 0
        else:
            if self.q[(s, 0)] > self.q[(s, 1)]:
                new_action = 0
            elif self.q[(s, 0)] < self.q[(s, 1)]:
                new_action = 1
            else: # choose action randomly if q values for jumping and swinging are equal
                new_action = 1 if npr.rand() < 0.1 else 0

        new_state  = state

        self.last_action = new_action
        self.last_state  = new_state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':

    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games. 
    run_games(agent, hist, 1000, 0)

    # Save history. 
    np.save('hist',np.array(hist))

    lowgravcount = 0
    highgravcount = 0
    lowgravsum = 0
    highgravsum = 0
    for trial in range(899,1000):
        if agent.gravitys[trial]:
            lowgravcount += 1
            lowgravsum += hist[trial]
        else:
            highgravcount += 1
            highgravsum += hist[trial]

    print 'high gravity average score for epochs 900-1000:', float(highgravsum) / highgravcount
    print 'low gravity average score for epochs 900-1000:', float(lowgravsum) / lowgravcount

    lowgravavgs = []
    highgravavgs = []
    for bin in range(0,10):
        lowgravsum = 0
        lowgravcount = 0
        highgravsum = 0
        highgravcount = 0
        for trial in range(0,100):
            if agent.gravitys[bin * 100 + trial]:
                lowgravcount += 1
                lowgravsum += hist[bin * 100 + trial]
            else:
                highgravcount += 1
                highgravsum += hist[bin * 100 + trial]
        lowgravavgs.append(float(lowgravsum) / lowgravcount)
        highgravavgs.append(float(highgravsum) / highgravcount)

    plt.figure(1)
    plt.bar(range(0, 10), lowgravavgs)
    plt.xticks(range(0,10), ('1-100', '101-200', '201-300', '301-400', '401-500', '501-600', '601-700', '701-800', '801-900', '901-1000'))
    plt.xlabel('Epoch')
    plt.ylabel('Average Score')
    plt.title('Low Gravity Average Score per 100 Epochs')

    plt.figure(2)
    plt.bar(range(0, 10), highgravavgs)
    plt.xticks(range(0,10), ('1-100', '101-200', '201-300', '301-400', '401-500', '501-600', '601-700', '701-800', '801-900', '901-1000'))
    plt.xlabel('Epoch')
    plt.ylabel('Average Score')
    plt.title('High Gravity Average Score per 100 Epochs')

    plt.show()

