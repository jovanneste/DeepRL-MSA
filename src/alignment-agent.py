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
    def __init__(self, m, n):
        super(DQN, self).__init__()
        self.m = m
        self.n = n
        self.model = self.build_model()

    def build_model(self):
        input_layer = Input(shape=(self.m, self.n))
        flatten_layer = Flatten()(input_layer)
        hidden1 = Dense(64, activation='relu')(flatten_layer)
        hidden2 = Dense(64, activation='relu')(hidden1)
        x_output = Dense(self.m, activation='softmax')(hidden2)
        y_output = Dense(self.n, activation='softmax')(hidden2)
        return tf.keras.Model(inputs=input_layer, outputs=[x_output, y_output])

    def forward(self, state):
        pass



def convergence():
    pass
    # way to say time to stop actions


def getAction(model, state, eplison):
    pass
    # epsilon decay either random or np.argmax(tf.nn.softmax(model.predict(state).numpy()[0]))



s = generate_sequence(20,20)
s[s=='_']=0
s = s.astype(int)
features = get_features(s)
print(features.shape)
features = np.squeeze(features)
print(features.shape)