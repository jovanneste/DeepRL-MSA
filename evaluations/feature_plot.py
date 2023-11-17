import tensorflow as tf
from tf_explain.core.grad_cam import GradCAM
import sys
sys.path.append('../src')
from single_agent.agent import Agent
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

dqn_agent = Agent(20, 20)
dqn_agent.model.load_weights('../src/single_agent/recent_weights.hdf5')
dqn_agent.model_target.load_weights('../src/single_agent/recent_weights.hdf5')
dqn_agent.epsilon = 0


state = np.asarray( [
    [20, 7, 1, 7, 1, 3, 7, 20, 7, 7, 20, 20, 3, 20, 7, 1, 3, 3, 7, 7],
    [20, 7, 1, 7, 1, 3, 7, 20, 7, 7, 20, 20, 3, 20, 7, 1, 3, 3, 7, 7],
    [7, 1, 1, 3, 7, 20, 7, 7, 3, 3, 20, 7, 3, 3, 20, 0, 0, 0, 0, 0],
    [1, 7, 20, 3, 7, 20, 20, 3, 20, 1, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [20, 1, 20, 7, 20, 3, 20, 7, 1, 3, 3, 7, 0, 0, 0, 0, 0, 0, 0, 0],
    [20, 7, 3, 7, 20, 3, 7, 7, 7, 20, 20, 7, 1, 3, 7, 20, 0, 0, 0, 0],
    [20, 7, 1, 7, 1, 3, 7, 7, 20, 7, 20, 7, 1, 3, 3, 0, 0, 0, 0, 0],
    [20, 7, 1, 7, 1, 3, 7, 20, 7, 7, 20, 20, 3, 20, 7, 1, 3, 3, 7, 7],
    [20, 20, 1, 7, 3, 3, 7, 7, 20, 20, 3, 7, 3, 7, 0, 0, 0, 0, 0, 0],
    [7, 1, 7, 20, 1, 7, 20, 7, 7, 20, 20, 3, 20, 1, 1, 7, 7, 0, 0, 0],
    [20, 7, 7, 1, 3, 7, 20, 7, 20, 7, 20, 1, 20, 7, 7, 0, 0, 0, 0, 0],
    [20, 1, 1, 1, 3, 1, 7, 7, 20, 20, 3, 20, 7, 1, 3, 7, 0, 0, 0, 0],
    [7, 1, 7, 1, 3, 7, 20, 7, 7, 20, 3, 20, 1, 1, 3, 7, 0, 0, 0, 0],
    [20, 7, 1, 1, 7, 7, 1, 1, 3, 7, 3, 3, 7, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 7, 1, 3, 20, 7, 7, 20, 20, 3, 1, 3, 3, 7, 0, 0, 0, 0, 0],
    [7, 7, 1, 20, 7, 7, 3, 20, 1, 1, 3, 7, 7, 0, 0, 0, 0, 0, 0, 0],
    [20, 7, 1, 7, 7, 7, 7, 20, 3, 20, 7, 3, 3, 7, 7, 0, 0, 0, 0, 0],
    [1, 3, 7, 20, 1, 7, 20, 20, 20, 1, 3, 3, 7, 1, 0, 0, 0, 0, 0, 0],
    [20, 3, 1, 7, 3, 1, 7, 7, 20, 20, 3, 20, 7, 3, 7, 0, 0, 0, 0, 0],
    [20, 1, 7, 3, 7, 20, 7, 20, 20, 3, 20, 7, 1, 3, 20, 3, 3, 0, 0, 0]
])


input_data = state.reshape((1,20,20,1))
action = dqn_agent.get_action(input_data)
print(action)
explainer = GradCAM()


for i in [100]:
    print(i)
    grid = explainer.explain((input_data, None), dqn_agent.model, class_index=i)  

    # Plot the state and GradCAM heatmap overlay
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the state array
    ax.imshow(state, cmap='viridis')  # Adjust the colormap as needed

    # Overlay the GradCAM heatmap with numbers
    masked_array = np.ma.masked_where(grid < grid.max(), grid)
    im = ax.imshow(masked_array, alpha=0.5, cmap='coolwarm')  # Adjust the colormap and transparency as needed

    # Display numbers from the state array on the plot
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            ax.text(j, i, state[i, j], ha='center', va='center', color='black')

    # Add a color bar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Level')

    plt.title('State with GradCAM Heatmap Overlay')
    plt.show()
    