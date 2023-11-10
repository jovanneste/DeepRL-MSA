import numpy as np
import sys
sys.path.append('../src')
import scoring
from scipy.stats import percentileofscore
    
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



state = np.asarray([[0,1,2,3,4],
                    [1,2,3,4,5],
                    [1,2,3,4,5],
                    [1,2,3,4,5],
                    [1,2,3,4,5]])


action_scores = {}

for iy, ix in np.ndindex(state.shape):
    coords = (ix,iy)
    action_scores[coords] = step(state, coords)


action_scores = sorted(action_scores.items(), key=lambda x:x[1], reverse=True)
sorted_action_scores = dict(action_scores)

input_coord = (0, 0)

# Extract values from the dictionary
values = list(sorted_action_scores.values())

# Calculate the percentile of the input coordinate
percentile = percentileofscore(values, sorted_action_scores[input_coord])
print(percentile)


# normal chosen action 
# new netork chosen action 
# random chosen action 

# repeat for differnent sizes - get percentages HISTOGRAMS