from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward'))

class Memory():
    def __init__(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)
    
    def store_experience(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self):
        self.memory = []