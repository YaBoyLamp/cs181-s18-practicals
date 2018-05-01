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
        self.alpha = 0.2
        self.gamma = 0.9
        self.epsilon = 0.10
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

        if self.last_state == None:  # for the very first action
            self.last_action = 0
            self.last_state = state
            # self.epsilon = 1/self.epochs  # reduce epsilon with each epoch
            self.alpha *= 0.95  # reduce learning rate with each epoch
            return 0

        if self.is_low_gravity == None:
            self.is_low_gravity = (state['monkey']['vel'] == -1)
            # print ("VELOCITY at first state: ", state['monkey']['vel'])
            self.gravitys.append(self.is_low_gravity)

        """
        what if we play around with extreme discretization of state space?

        * if monkey is below the bottom of the tree, in the gap, or above the top
        * if the monkey is close, middle, or far from the tree
        * if the monkey's velocity is high or low
        """
        if (self.last_state['monkey']['top'] < self.last_state['tree']['bot']):
            last_monkey_pos = 0
        elif (self.last_state['monkey']['top'] < (self.last_state['tree']['top'] + self.last_state['tree']['bot'])/2):  # below midpoint?
            last_monkey_pos = 1
        elif (self.last_state['monkey']['top'] < self.last_state['tree']['top']): 
            last_monkey_pos = 2
        else:
            last_monkey_pos = 3

        if (state['monkey']['bot'] < state['tree']['bot']):
            monkey_pos = 0
        elif (state['monkey']['top'] < (state['tree']['top'] + state['tree']['bot'])/2):  # below midpoint?
            monkey_pos = 1
        elif (state['monkey']['top'] < state['tree']['top']): 
            monkey_pos = 2
        else:
            monkey_pos = 3

        bin_size = 100
        bin_size_2 = 25
        bin_size_3 = 10
        # ls = (self.last_state['tree']['dist'] // bin_size, (self.last_state['tree']['top'] - self.last_state['monkey']['top']) // bin_size_2, 
        #     self.last_state['monkey']['vel'] // bin_size_3,  self.is_low_gravity)
        # s = (state['tree']['dist'] // bin_size, (state['tree']['top'] - state['monkey']['top']) // bin_size_2, 
        #     state['monkey']['vel'] // bin_size_3,  self.is_low_gravity)
        # # print ("current state: ", s)

        ls = (last_monkey_pos, self.is_low_gravity)
        s = (monkey_pos, self.is_low_gravity)

        # print ("last state: ", ls)
        # print ("current state: ", s)
        self.q[(ls, self.last_action)] = ((1 - self.alpha) * self.q[(ls, self.last_action)]) + (self.alpha * ((self.last_reward) + (self.gamma * max([self.q[(s,0)], self.q[(s,1)]]))))

        # some heuristics
        if state['monkey']['top'] + state['monkey']['vel'] < 0:  # if below screen in next iteration
            self.last_action = 1
            self.last_state = state 
            return 1
        if state['monkey']['bot'] + state['monkey']['vel'] > 400:
            self.last_action = 0
            self.last_state = state
            return 0

        # if self.epochs > 1900:  # no epsilon-greedy past epoch 1900
        #     self.epsilon = 0
        if npr.rand() < self.epsilon:  # if we choose to enter exploration!
            new_action = 1 if npr.rand() < 0.5 else 0  # pick action 0 and 1 50% of the time
            self.epsilon *= 0.80
        else:
            if self.q[(s, 0)] > self.q[(s, 1)]:
                new_action = 0
                # self.count += 1
            elif self.q[(s, 0)] < self.q[(s, 1)]:
                new_action = 1
                # self.count += 1
            else:  # in the case of ties
                new_action = 1 if npr.rand() < 0.5 else 0
            # else:
            #     if state['tree']['dist'] > 0 and state['monkey']['vel'] < -20 and self.last_action != 2:
            #         if state['monkey']['top'] < (state['tree']['top'] + state['tree']['bot'] / 2):
            #             new_action = 1
            #         else:
            #             new_action = 0
            #     else:
            #         new_action = 1 if npr.rand() < 0.1 else 0

        # new_state  = state

        self.last_action = new_action
        self.last_state  = state

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
    iters = 50
    run_games(agent, hist, iters, 0)

    # Save history. 
    np.save('hist',np.array(hist))
    print (hist)
    print (agent.gravitys)

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
    # print (agent.count)
    # print (max(hist))
    # print (sum(agent.gravitys))
    # print agent.q.values()

    # with open('scores2.txt', 'w') as f:
    #     for score in range(iters):
    #         f.write('%s:%s:%r\n' % (score, hist[score], agent.gravitys[score]))

    x_axis = np.arange(iters)
    plt.figure(1)
    plt.xlabel("Epoch")
    plt.ylabel("Game Score")
    plt.plot(x_axis[agent.gravitys], np.array(hist)[agent.gravitys], "-o", label="Low Gravity")
    # plt.show()

    # plt.figure(2)
    plt.plot(x_axis[np.logical_not(agent.gravitys)], np.array(hist)[np.logical_not(agent.gravitys)], "-o", label="High gravity")
    print ("LOW GRAVITY: average: {} for {} epochs".format(np.mean(np.array(hist)[agent.gravitys]), 
        len(x_axis[agent.gravitys])))
    print ("HIGH GRAVITY: average: {} for {} epochs".format(np.mean(np.array(hist)[np.logical_not(agent.gravitys)]),
        len(x_axis[np.logical_not(agent.gravitys)])))
    plt.legend()
    plt.show()


    # plt.hist(hist)