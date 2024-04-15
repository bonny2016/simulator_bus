import numpy as np
from collections import deque
import random

class Memory():
    def __init__(self,members):
        self.memory = {}

        # temp memory to store last-step state and action because of no immediate feedback
        self.temp_memory = {}
        self.experience = 0
        self.memory = deque(maxlen=2000)
        for m in members:
            self.temp_memory[m]={'s':[],'a':[],'r':0, 'is_done':[] }

    def remember_temp_action(self, id, a):
        self.temp_memory[id]['a'].append(a)

    def remember_temp_reward(self, id, r):
        self.temp_memory[id]['r'] += r
    def remember_temp_result(self, id, s, is_done):
        if (s is not None and len(self.temp_memory[id]['s']) == 1 and len(self.temp_memory[id]['a']) == 1):
            prev_state = self.temp_memory[id]['s'].pop(0)
            prev_action = self.temp_memory[id]['a'].pop(0)
            reward = self.temp_memory[id]['r']
            self.remember(prev_state, prev_action, reward, s)
            self.temp_memory[id]['r'] = 0
        if s is not None and not is_done:
            self.temp_memory[id]['s'].append(s)

    def remember_temp(self, id, state, a, r, is_done):
        # previous & current state available, time to save to memory
        if(state is not None and len(self.temp_memory[id]['s']) == 1):
            prev_state = self.temp_memory[id]['s'].pop(0)
            prev_action = self.temp_memory[id]['a']
            reward = self.temp_memory[id]['r']
            self.remember(prev_state, prev_action, reward, state)
            self.temp_memory[id]['r'] = 0
        self.temp_memory[id]['s'].append(state)
        self.temp_memory[id]['r'] += r
        self.temp_memory[id]['a'].append(a)
        self.temp_memory[id]['is_done'].append(is_done)

    def remember(self, state, action, reward, next_state):
        self.experience+=1
        self.memory.append((state, action, reward, next_state))

    def size(self):
        return len(self.memory)
    def sample(self, n):
        return random.sample(self.memory, n)
    def pop_all(self):
        result = list(self.memory)
        self.memory.clear()
        return result
