import numpy as np
import sys
sys.path.append('../src')
import scoring
from single_agent.agent import Agent
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
    
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
    return scoring.compute_sp_score(new_state)


def get_percentile(state, new_state, action):
    action_scores = {}
    if (state==new_state).all():
        penalty = -4
    else:
        penalty = 0
        
    print("Penalty: ", penalty)

    for iy, ix in np.ndindex(state.shape):
        coords = (ix,iy)
        action_scores[coords] = step(state, coords) + penalty


    action_scores = sorted(action_scores.items(), key=lambda x:x[1], reverse=True)
    sorted_action_scores = dict(action_scores)
    print(sorted_action_scores)

    values = list(sorted_action_scores.values())

    return 100 - percentileofscore(values, sorted_action_scores[action])







state = np.asarray([[9,23,1,5,9,9],
                    [9,5,1,0,0,0],
                    [23,9,9,0,0,0],
                    [5,9,9,5,9,9],
                    [23,5,9,0,0,0],
                    [9,9,9,0,0,0]])


dqn_agent = Agent(6, 6)
dqn_agent.model.load_weights('../src/single_agent/recent_weights.hdf5')
dqn_agent.model_target.load_weights('../src/single_agent/recent_weights.hdf5')
dqn_agent.epsilon = 0.0


model_percentiles = []
for i in range(1):
    action = dqn_agent.get_action(state)
    new_state, score, done = dqn_agent.step(state, action)   
    print(state)
    print(new_state)
    print("Chosen action: ", action)
    model_percentiles.append(get_percentile(state, new_state, action))
    
    state = new_state

                            
print(model_percentiles)
        
        



# normal chosen action 
# new netork chosen action 
# random chosen action 

# repeat for differnent sizes - get percentages HISTOGRAMS
#
#random_per = [15, 59, 35, 23, 66]
#normal_model_per = [55, 77, 34, 81, 81]
#new_model_per = [98, 98, 78, 88, 84]
#
#plt.hist(random_per, bins=5, alpha=0.5, label='Dataset 1', edgecolor='black')
#plt.hist(normal_model_per, bins=5, alpha=0.5, label='Dataset 2', edgecolor='black')
#plt.hist(new_model_per, bins=5, alpha=0.5, label='Dataset 3', edgecolor='black')
#
## Add labels and title
#plt.title("Histograms of Datasets")
#plt.xlabel("Values")
#plt.ylabel("Frequency")
#plt.legend()
#
## Show the plot
#plt.show()