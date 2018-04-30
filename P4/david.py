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
        self.epsilon = 0.05
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

        if self.last_state == None:
            self.last_action = 0
            self.last_state = state
            return 0
        

        if self.is_low_gravity == None:
            self.is_low_gravity = state['monkey']['vel'] == -1
            self.gravitys.append(state['monkey']['vel'] == -1)

        ls = (self.last_state['tree']['dist'] / 30, self.last_state['tree']['top'] / 30, self.last_state['monkey']['vel'] / 10, self.last_state['monkey']['top'] / 30, self.is_low_gravity)
        s = (state['tree']['dist'] / 30, state['tree']['top'] / 30, state['monkey']['vel'] / 10, state['monkey']['top'] / 30, self.is_low_gravity)
        self.q[(ls, self.last_action)] = ((1 - self.alpha) * self.q[(ls, self.last_action)]) + (self.alpha * ((self.last_reward) + (self.gamma * max([self.q[(s,0)], self.q[(s,1)]]))))

        if state['monkey']['top'] + state['monkey']['vel'] < 0:
            self.last_action = 1
            self.last_state = state 
            return 1
        if state['monkey']['bot'] + state['monkey']['top'] > 400:
            self.last_action = 0
            self.last_state = state
            return 0

        if self.epochs > 1900:
            self.epsilon = 0
        if npr.rand() < self.epsilon:
            new_action = 1 if npr.rand() < 0.1 else 0
        else:
            if self.q[(s, 0)] > self.q[(s, 1)]:
                new_action = 0
                # self.count += 1
            elif self.q[(s, 0)] < self.q[(s, 1)]:
                new_action = 1
                # self.count += 1
            else:
                new_action = 1 if npr.rand() < 0.1 else 0
            # else:
            #     if state['tree']['dist'] > 0 and state['monkey']['vel'] < -20 and self.last_action != 2:
            #         if state['monkey']['top'] < (state['tree']['top'] + state['tree']['bot'] / 2):
            #             new_action = 1
            #         else:
            #             new_action = 0
            #     else:
            #         new_action = 1 if npr.rand() < 0.1 else 0

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
    # print sum([x != 0 for x in agent.q.values()])
    # print len(agent.q)
    print (agent.count)
    print max(hist)
    print sum(agent.gravitys)
    # print agent.q.values()

    with open('scores2.txt', 'w') as f:
        for score in xrange(2000):
            f.write('%s:%s:%r\n' % (score, hist[score], agent.gravitys[score]))

    plt.plot(range(1,2001), hist)
    plt.show()

    plt.hist(hist)