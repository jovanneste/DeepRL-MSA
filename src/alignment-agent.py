from generator import *
from collections import namedtuple
import torch
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input
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


def char_to_int(sequence):
    char_to_number = {chr(65 + i): i + 1 for i in range(26) if chr(65 + i) != '_'}
    return np.array([[char_to_number[c] for c in row] for row in sequence])


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
    def __init__(self, input_shape, m, n):
        super(DQN, self).__init__()
        self.m = m
        self.n = n
        self.model = self.build_model(input_shape)

    def build_model(self, input_shape):
        input_layer = Input(shape=input_shape)
        flatten_layer = Flatten()(input_layer)
        hidden1 = Dense(64, activation='relu')(flatten_layer)
        hidden2 = Dense(64, activation='relu')(hidden1)
        x_output = Dense(self.m, activation='softmax')(hidden2)
        y_output = Dense(self.n, activation='softmax')(hidden2)
        return tf.keras.Model(inputs=input_layer, outputs=[x_output, y_output])

    def forward(self, state):
        x_probs, y_probs = self.model.predict(np.expand_dims(state, axis=0))
        x = np.random.choice(range(self.m), p=x_probs[0])
        y = np.random.choice(range(self.n), p=y_probs[0])
        return x, y

    def train(self, state, x_target, y_target):
        x_target = tf.keras.utils.to_categorical(x_target, num_classes=self.m)
        y_target = tf.keras.utils.to_categorical(y_target, num_classes=self.n)
        self.model.train_on_batch(np.expand_dims(state, axis=0), [x_target, y_target])



def convergence():
    pass
    # way to say time to stop actions


def getAction(model, state, eplison):
    pass
    # epsilon decay either random or np.argmax(tf.nn.softmax(model.predict(state).numpy()[0]))

def performAction(state, action):
    pass
    # insert gap 
    # return new_state, reward, done(bool)






# # Create an instance of the DQN class
# input_shape = (2, 4)  # Adjust the shape based on your input
# m, n = 2, 4  # Adjust these values based on your grid dimensions
# dqn = DQN(input_shape, m, n)


s = generate_sequence(4,4)
print(s)
# print(char_to_int(s))





# Now, you can use numerical_state as input to your DQN
# x, y = dqn.forward(numerical_state)





# print(x,y)
