import numpy as np
import sys
sys.path.append('../src')
import scoring
from single_agent.agent import Agent
import matplotlib.pyplot as plt
import random
import tqdm 
import pickle

def split_and_shuffle(dictionary):
    sub_dictionaries = {}

    for key, value in dictionary.items():
        if value not in sub_dictionaries:
            sub_dictionaries[value] = {key: value}
        else:
            sub_dictionaries[value][key] = value

    result_dictionary = {}
    for sub_dict in sub_dictionaries.values():
        keys = list(sub_dict.keys())
        random.shuffle(keys)
        sub_dict = {key: sub_dict[key] for key in keys}

        result_dictionary.update(sub_dict)

    return result_dictionary

    
def step(state, coords):
    s_list = state.tolist()
    x, y = coords
    row = s_list[y]

    if row[x] == 0:
        row.pop(x)
        row.append(0)
    else:
        if 0 in row:
            row.pop(row.index(0))
            row.insert(x, 0)

    new_state = np.array(s_list).reshape(state.shape)
    
    if(new_state==state).all():
        penalty = - 4
    else:
        penalty = 0
    
    return scoring.compute_sp_score(new_state)+penalty


def get_percentile(state, new_state, action):
    action_scores = {}
    if (state==new_state).all():
        penalty = -4
    else:
        penalty = 0

    for iy, ix in np.ndindex(state.shape):
        coords = (ix,iy)
        action_scores[coords] = step(state, coords) + penalty

    action_scores = sorted(action_scores.items(), key=lambda x:x[1], reverse=True)
    sorted_action_scores = dict(action_scores)
    shuffled = split_and_shuffle(sorted_action_scores)
    values = list(shuffled.values())

    indices = [index for index, value in enumerate(list(values)) if value == shuffled[action]]

    return (random.choice(indices)/len(shuffled))*100



def get_model_action_percentiles(state, n_steps):

    dqn_agent = Agent(10, 10)
    dqn_agent.model.load_weights('../src/single_agent/recent_weights.hdf5')
    dqn_agent.model_target.load_weights('../src/single_agent/recent_weights.hdf5')
    dqn_agent.epsilon = 0.0

    model_percentiles = []
    for i in tqdm.tqdm(range(n_steps)):
        action = dqn_agent.get_action(state)
        new_state, score, done = dqn_agent.step(state, action)   
        action_rating = get_percentile(state, new_state, action)

        model_percentiles.append(action_rating)
        state = new_state
        
    with open('10x10percentiles.pkl', 'wb') as file:
        pickle.dump(model_percentiles, file)

    print('Array dumped to file successfully.')
    return model_percentiles



def plot_percentiles(model_percentiles):
    plt.hist(model_percentiles, bins = 5, alpha=0.5, label='Dataset 3', edgecolor='white')
    plt.xlim(0, 100)

    # Add labels and title
    plt.title("Histograms of Datasets")
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.legend()

    # Show the plot
    plt.show()


state = np.asarray([[ 1 , 1,  1, 23,  5,  5,  1,  5, 23, 23],
 [ 1,  1,  5, 23, 23,  0,  0,  0,  0,  0],
 [ 1,  1,  1,  5,  0,  0,  0,  0,  0,  0],
 [ 1,  5,  1,  5, 23,  0,  0,  0,  0,  0],
 [ 1,  1,  1,  1,  5,  1,  5, 23, 23,  0],
 [23,  1,  1, 23,  5,  1, 23,  0,  0,  0],
 [ 1,  5,  5,  7, 23, 23,  0,  0,  0,  0],
 [23,  0,  0,  0,  7,  23, 5,  0,  0,  0],
 [ 1,  7,  5,  1,  7,  0,  0,  0,  0,  0],
 [ 1, 23, 23,  5, 23, 23,  0,  0,  0,  0]])


