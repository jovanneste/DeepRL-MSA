import numpy as np
import sys
sys.path.append('..')
import scoring
    
    
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



state = np.asarray([[1,0,5,4],
                  [2,5,4,0],
                  [1,2,5,0]])


action_scores = {}

for iy, ix in np.ndindex(state.shape):
    coords = (ix,iy)
    action_scores[coords] = step(state, coords)

action_scores = sorted(action_scores.items(), key=lambda x:x[1], reverse=True)
sorted_action_scores = dict(action_scores)

print(sorted_action_scores)