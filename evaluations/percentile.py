import numpy as np
import sys
sys.path.append('../src')
import scoring
from single_agent.agent import Agent
import random
import tqdm 
import pickle
import copy

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


def get_model_action_percentiles(s, n_steps, random_actions):
    dqn_agent = Agent(20, 50)
    dqn_agent.model.load_weights('../src/single_agent/recent_weights.hdf5')
    dqn_agent.model_target.load_weights('../src/single_agent/recent_weights.hdf5')
    if random_actions:
        dqn_agent.epsilon = 1
    else:
        dqn_agent.epsilon = 0
        
    model_percentiles = []
    state = copy.deepcopy(s)
    for i in tqdm.tqdm(range(n_steps)):
        action = dqn_agent.get_action(state)
        new_state, score, done = dqn_agent.step(state, action)   
        action_rating = get_percentile(state, new_state, action)
        model_percentiles.append(action_rating)
        state = new_state
        
    if random_actions:
        with open('newmodel/20x50percentilesRANDOM.pkl', 'wb') as file:
            pickle.dump(model_percentiles, file)
        print('Array dumped to file successfully.')
    else:
        with open('newmodel/20x50percentiles.pkl', 'wb') as file:
            pickle.dump(model_percentiles, file)
        print('Array dumped to file successfully.')
        
    return model_percentiles
    

state = np.asarray([
 [3, 3, 16, 2, 2, 13, 16, 13, 2, 13, 16, 3, 13, 3, 13, 3, 16, 3, 13, 16, 2, 13, 2, 13, 3, 3, 2, 2, 3, 3, 3, 3, 3, 16, 3, 2, 3, 2, 2, 16, 16, 3, 3, 16, 3, 3, 13, 13, 13, 13],
 [16, 2, 2, 13, 16, 13, 3, 3, 16, 13, 16, 3, 2, 2, 13, 3, 3, 2, 2, 3, 3, 3, 16, 13, 2, 3, 13, 16, 16, 16, 3, 3, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [3, 2, 2, 13, 16, 2, 13, 16, 13, 3, 16, 3, 13, 16, 3, 13, 13, 3, 3, 3, 16, 3, 2, 3, 2, 2, 16, 16, 3, 3, 13, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],
 [3, 3, 16, 16, 13, 13, 2, 13, 3, 3, 16, 13, 13, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2, 16, 16, 3, 3, 2, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],
 [13, 3, 16, 2, 13, 16, 13, 13, 13, 13, 16, 13, 16, 13, 13, 2, 3, 3, 3, 3, 3, 3, 13, 3, 3, 2, 3, 3, 16, 3, 13, 2, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0],
 [3, 3, 2, 2, 3, 3, 13, 13, 16, 3, 3, 13, 3, 16, 3, 13, 16, 2, 3, 2, 13, 3, 2, 13, 16, 3, 3, 2, 2, 16, 3, 16, 3, 13, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],
 [3, 13, 13, 16, 3, 13, 3, 16, 3, 13, 13, 3, 13, 2, 16, 2, 16, 16, 3, 2, 3, 2, 2, 2, 16, 16, 3, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0],
 [3, 16, 2, 13, 16, 13, 2, 13, 16, 3, 2, 13, 2, 16, 3, 2, 13, 3, 2, 2, 3, 3, 3, 16, 16, 16, 3, 13, 13, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0],
 [3, 16, 2, 13, 16, 13, 2, 13, 16, 3, 2, 13, 2, 16, 3, 2, 13, 3, 2, 2, 3, 3, 3, 16, 16, 16, 3, 13, 13, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0],
 [3, 16, 2, 16, 3, 16, 13, 2, 13, 13, 3, 3, 2, 3, 3, 16, 2, 16, 3, 3, 16, 16, 3, 16, 3, 13, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0],
 [13, 2, 13, 2, 13, 16, 3, 3, 13, 3, 16, 2, 3, 16, 2, 13, 16, 13, 2, 2, 2, 2, 16, 16, 3, 3, 13, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0],
 [3, 2, 16, 2, 13, 2, 13, 3, 13, 13, 13, 3, 16, 3, 2, 13, 3, 3, 2, 2, 3, 16, 16, 2, 16, 16, 3, 3, 2, 13, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0],
 [2, 16, 2, 2, 13, 16, 13, 2, 13, 2, 16, 3, 2, 16, 13, 2, 13, 3, 3, 3, 3, 3, 3, 16, 2, 2, 2, 16, 16, 3, 3, 3, 3, 13, 2, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0],
 [3, 16, 2, 3, 13, 16, 2, 13, 16, 3, 13, 3, 3, 3, 2, 2, 13, 3, 3, 3, 3, 3, 2, 3, 2, 2, 16, 16, 3, 3, 3, 13, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0],
 [3, 3, 16, 13, 3, 3, 2, 16, 13, 16, 3, 2, 3, 3, 16, 3, 3, 16, 3, 3, 16, 16, 16, 16, 2, 3, 13, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0],
 [2, 13, 2, 13, 3, 13, 13, 13, 2, 2, 3, 3, 2, 3, 16, 3, 3, 2, 16, 16, 16, 3, 13, 13, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0],
 [16, 3, 2, 2, 2, 16, 3, 13, 3, 3, 13, 16, 2, 13, 2, 2, 16, 2, 13, 3, 3, 3, 16, 16, 3, 3, 16, 13, 3, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0],
 [2, 16, 13, 2, 3, 3, 13, 2, 2, 13, 13, 3, 3, 3, 3, 3, 13, 2, 2, 2, 2, 16, 3, 16, 3, 2, 13, 13, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0],
 [3, 16, 2, 16, 13, 16, 13, 2, 13, 3, 3, 16, 2, 13, 16, 13, 3, 2, 2, 3, 3, 16, 2, 2, 16, 3, 16, 16, 13, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0],
 [3, 3, 16, 3, 13, 3, 3, 16, 2, 13, 13, 3, 3, 2, 3, 3, 3, 3, 2, 3, 2, 2, 3, 3, 16, 3, 13, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0]
]) 

print(state.shape)

model_percentiles = get_model_action_percentiles(state, 100, False)
print(model_percentiles)