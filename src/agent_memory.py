from collections import deque

class Memory():
    def __init__(self,max_len):
        self.max_len = max_len
        self.states = deque(maxlen = max_len)
        self.actions = deque(maxlen = max_len)
        self.rewards = deque(maxlen = max_len)

    def store_experience(self,next_frame, next_frames_reward, next_action, next_frame_terminal):
        self.states.append(next_frame)
        self.actions.append(next_action)
        self.rewards.append(next_frames_reward)

    def clear(self):
        self.states = deque(maxlen = max_len)
        self.actions = deque(maxlen = max_len)
        self.rewards = deque(maxlen = max_len)
    
    def __len__(self):
        return len(self.states)
