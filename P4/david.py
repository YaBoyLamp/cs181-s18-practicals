# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

from SwingyMonkey import SwingyMonkey
from collections import defaultdict

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
        self.epsilon = 0.05

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        if self.last_state == None:
            self.last_action = 0
            self.last_state = state
            return 0

        s = (state['tree']['dist'] / 10, state['tree']['top'] / 10, state['monkey']['vel'] / 10, state['monkey']['top'] / 10)
        self.q[(s, self.last_action)] = ((1 - self.alpha) * self.q[(s, self.last_action)]) + (self.alpha * (self.last_reward + self.gamma * max([self.q[(s,0)], self.q[(s,1)]])))

        if npr.rand() < self.epsilon:
            new_action = 1 if npr.rand() < 0.1 else 0
        else:
            if self.q[(s, 0)] > self.q[(s, 1)]:
                new_action = 0
            elif self.q[(s, 0)] < self.q[(s, 1)]:
                new_action = 1
            else:
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
    run_games(agent, hist, 2000, 0)

    # Save history. 
    np.save('hist',np.array(hist))

    # print 'treegaps = ', agent.treegaps
    # print 'treetops = ', agent.treetops
    # print 'treegaps = ', agent.treebots
    # print 'treemids = ', agent.treemids

    # print 'max treetop = ', max(agent.treetops)
    # print 'min treetop = ', min(agent.treetops)
    # print 'max treebot = ', max(agent.treebots)
    # print 'min treebot = ', min(agent.treebots)
    # print 'min treemid = ', min(agent.treemids)
    # print 'max treemid = ', max(agent.treemids)
    # print 'max monkeyvel = ', max(agent.monkeyvels)
    # print 'min monkeyvel = ', min(agent.monkeyvels)
    print len(agent.q)
    print max(hist)

