from generator import *
from collections import namedtuple
import pickle
import torch
import tensorflow as tf
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


def get_features(state):
    with open("resnet.pkl", "rb") as file:
        pretrained_model = pickle.load(file)
    if pretrained_model:
        return np.squeeze(pretrained_model.predict(process(state)))
    else:
        print("Model not loaded...")
        return 0


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):
    def __init__(self, capacity):
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
        



def convergence():
    pass
    # way to say time to stop actions


def getAction(model, state, eplison):
    pass
    # epsilon decay either random or np.argmax(tf.nn.softmax(model.predict(state).numpy()[0]))




s = generate_sequence(5,5)
s[s=='_']=0
s = s.astype(int)
features = get_features(s)
print(features.shape)