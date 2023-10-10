from generator import *
from collections import namedtuple
import pickle
import tqdm
import copy
import torch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.applications.resnet50 import preprocess_input
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def score(msa):
    num_sequences = len(msa)
    alignment_length = len(msa[0])
    score = 0

    for i in range(alignment_length):
        for j in range(num_sequences):
            for k in range(j + 1, num_sequences):
                if i >= len(msa[j]) or i >= len(msa[k]):
                    continue  

                char_j = msa[j][i]
                char_k = msa[k][i]

                if char_j != '_' and char_k != '_':
                    if char_j == char_k:
                        score += 1
    return score


def process(data):
    state = np.stack((data,)*3, axis=-1)  
    state = tf.image.resize(state, (32, 32))  
    state = preprocess_input(state)
    return np.expand_dims(state, axis=0)


def get_features(s):
    state = copy.deepcopy(s)
    state[state=='_']=0
    state = state.astype(int)
    with open("resnet.pkl", "rb") as file:
        pretrained_model = pickle.load(file)
    if pretrained_model:
        features = np.reshape(np.squeeze(pretrained_model.predict(process(state))), (1,2048))
        return features
    else:
        print("Model not loaded...")
        return 0

    
def step(state, coords):
    s_list = state.tolist()
    x, y = coords
    row = s_list[y]

    if row[x] == '_':
        row.pop(x)
        row.append('_')
    else:
        if '_' in row:
            row.pop(row.index('_'))
            row.insert(0, '_')
            
    new_state = np.array(s_list).reshape(state.shape)
    done = True 
#    use convergence check
    return new_state, score(new_state), done


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):
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
    

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=self.input_shape),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.n_actions, activation='linear')
        ])
        return model

    def forward(self, features):
        q_values = self.model.predict(features)
        index = np.argmax(q_values)
        return ((index-1)%5, (index-1)//5)

    
def convergence():
    pass
    # way to say time to stop actions

    
EPISODES = 1
epsilon = 0.95
reduction = epsilon/EPISODES
def epsilonGreedy(model, features):
    global epsilon, reduction
    if np.random.random() < 1-epsilon:
        action = model.forward(features)
    else:
        action = (np.random.randint(0, 4), np.random.randint(0, 4))
    
    if epsilon>0:
        epsilon-=reduction  
    return action


model = DQN((2048,), 25)
loss_function = nn.MSELoss()
def optimise_model():
    transitions = replay.sample(1)
    batch = Transition(*zip(*transitions))
    features = [get_features(state) for state in batch.state]
    q_values = model.forward(features)
    
    

global replay
replay = ReplayMemory()
def train_alignment_agent(sequences):    
    optimizer = tf.keras.optimizers.Adam()
    for episode in tqdm.tqdm(range(EPISODES)):
        state = sequences
        replay.clear()
        done = False
        
        while not done:
            action = epsilonGreedy(model, get_features(state))
            new_state, reward, done = step(state, action)
            replay.store_experience(state, action, new_state, reward)
            state = new_state
            
            if done:
                optimise_model()
                break
        
    


s = generate_sequence(5,5)
print(s)
train_alignment_agent(s)