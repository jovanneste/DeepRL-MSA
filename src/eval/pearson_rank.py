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


for iy, ix in np.ndindex(state.shape):
    print((ix,iy), step(state, (ix,iy)))