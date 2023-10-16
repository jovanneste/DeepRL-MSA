from generator import *
from collections import namedtuple
import pickle
import tqdm
import copy
import torch
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.layers import Dense, Flatten, Input
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
    

i=0
def step(state, coords):
    global i
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
    if i==0:
        done = False
        i+=1
    else:
        done = True
        i=0
#    use convergence check
    return new_state, score(new_state), done


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'features'))
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
        self.model.trainable = True

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.input_shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_actions)
        )
        return model
    
    def params(self):
        return self.model.parameters()

    def forward(self, features):
        q_values = self.model(features)
        return torch.argmax(q_values), q_values

def index_to_coords(index):
    return ((index-1)%5, (index-1)//5)

def coords_to_index(coords):
    x,y = coords
    return (y*5)+x+1
    
def convergence():
    pass
    # way to say time to stop actions

    
EPISODES = 2
epsilon = 0.95
reduction = epsilon/EPISODES
def epsilonGreedy(model, features):
    global epsilon, reduction
    if np.random.random() < 1-epsilon:
        print("MODEL ACTION")
        index, q_value = model.forward(torch.tensor(features))
        action = index_to_coords(index)
    else:
        print("RANDOM ACTION")
        action = (np.random.randint(0, 4), np.random.randint(0, 4))
    if epsilon>0:
        epsilon-=reduction  
    return action


model = DQN((2048,), 25)
LR = 1e-4
loss_function = nn.MSELoss()
BATCH_SIZE = 2
GAMMA = 0.99

optimizer = optim.Adam(model.parameters(), lr=LR)
def optimise_model():
    transitions = replay.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    q_values = torch.stack([model.forward(torch.tensor(i))[1] for i in batch.features])
    q_values = q_values.view(BATCH_SIZE, 25)

    non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None], dim=0)

    state_batch = torch.cat(batch.state, dim=0)
    action_batch = torch.cat(batch.action, dim=0)
    reward_batch = torch.cat(batch.reward, dim=0)
    
    state_action_values = torch.gather(q_values, 1, action_batch.view(-1, 1))


    next_state_values = torch.zeros(BATCH_SIZE, dtype=torch.float32)

    # Calculate the expected Q-values for non-final states
    if non_final_mask.any():
        non_final_next_states = non_final_next_states.chunk(2, dim=0)
        q_values = []
        
        for state in non_final_next_states:
            features = get_features(state.numpy())
            q_values.append(model.forward(torch.tensor(features))[1])
        q_values = torch.cat(q_values)
        next_state_values[non_final_mask] = q_values.max(1)[0]

    # Calculate the expected Q-values based on the Bellman equation
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.float()


    # Calculate the loss using the Huber loss
    criterion = torch.nn.SmoothL1Loss()  # Huber loss
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.item())
    exit()

    return loss.item()  # Return the loss as a scalar

    
global replay
replay = ReplayMemory()


def train_alignment_agent(sequences):    
    for episode in tqdm.tqdm(range(EPISODES)):
        state = sequences
        replay.clear()
        done = False
#        optimizer = optim.Adam(model.params(), lr=LR)
        while not done:
            features = get_features(state)
            action = epsilonGreedy(model, features)
            new_state, reward, done = step(state, action)
            state[state=='_']=0
            new_state[new_state=='_']=0
       
            replay.store_experience(torch.tensor(state.astype(int)), torch.tensor([coords_to_index(action)]), torch.tensor(new_state.astype(int)), torch.tensor([reward]), features)
            state = new_state
            
            if done:
                loss = optimise_model()
                print("Loss: ", + str(loss))
                exit()
                break
        
    

s = generate_sequence(5,5)
print(s)
train_alignment_agent(s)