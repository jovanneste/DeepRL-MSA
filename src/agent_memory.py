from collections import deque
import random 

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
        
        
    def sample(self, batch_size):
        if len(self) < batch_size:
            raise ValueError("Insufficient experiences to sample a batch.")

        indices = random.sample(range(len(self)), batch_size)

        sampled_states = []
        sampled_actions = []
        sampled_rewards = []
        sampled_next_states = []

        for index in indices:
            sampled_states.append(self.states[index])
            sampled_next_states.append(self.states[index+1])
            sampled_actions.append(self.actions[index])
            sampled_rewards.append(self.rewards[index])

        return sampled_states, sampled_next_states, sampled_actions, sampled_rewards
    
    
    def __len__(self):
        return len(self.states)
