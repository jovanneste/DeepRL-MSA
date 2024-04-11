import tensorflow as tf
from tf_explain.core.grad_cam import GradCAM
import sys
sys.path.append('../src')
from single_agent.agent import Agent
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from matplotlib.colors import LinearSegmentedColormap

dqn_agent = Agent(10, 15)
dqn_agent.model.load_weights('../src/single_agent/recent_weights.hdf5')
dqn_agent.model_target.load_weights('../src/single_agent/recent_weights.hdf5')
dqn_agent.epsilon = 0


state = np.asarray([[20, 7, 7, 1, 20, 3, 1, 3, 20, 20, 3, 1, 1, 1, 3],
 [7, 3, 1, 7, 20, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0],
 [20, 7, 1, 1, 20, 3, 1, 20, 20, 3, 20, 1, 1, 3, 0],
 [20, 7, 1, 20, 3, 3, 20, 3, 3, 3, 1, 7, 3, 0, 0],
 [20, 7, 7, 1, 20, 3, 3, 20, 20, 3, 1, 1, 1, 3, 0],
 [20, 7, 7, 1, 20, 3, 3, 20, 3, 1, 3, 0, 0, 0, 0],
 [20, 7, 7, 20, 3, 1, 3, 20, 1, 3, 20, 1, 1, 3, 0],
 [20, 7, 7, 20, 3, 1, 3, 3, 1, 1, 3, 0, 0, 0, 0],
 [7, 7, 20, 20, 3, 1, 3, 20, 20, 1, 1, 1, 3, 0, 0],
 [20, 7, 7, 1, 20, 3, 3, 7, 20, 3, 1, 3, 3, 0, 0]])

print(state.shape)
encoder = OneHotEncoder()
state = encoder.fit_transform(state.reshape(-1, 1)).toarray()

input_data = state.reshape((1,10,15,1))
action = np.argmax(dqn_agent.model.predict(input_data))
print("Chosen action: ", action)

explainer = GradCAM()

for i in [1,2,action]:
    print(i)
    grid = explainer.explain((input_data, None), dqn_agent.model, class_index=i)  
    plt.imshow(grid, cmap='viridis')
    cbar = plt.colorbar()
    cbar.set_label('Attention Level')
    plt.show()
