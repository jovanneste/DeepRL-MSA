from collections import deque

class Memory():
    def __init__(self, max_len):
        self.max_len = max_len
        self.states = deque(maxlen = max_len)
        self.actions = deque(maxlen = max_len)
        self.rewards = deque(maxlen = max_len)

    def store_experience(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = deque(maxlen = self.max_len)
        self.actions = deque(maxlen = self.max_len)
        self.rewards = deque(maxlen = self.max_len)
    
    def __len__(self):
        return len(self.states)
