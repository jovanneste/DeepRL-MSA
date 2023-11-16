import tensorflow as tf
import sys
sys.path.append('../src')
from single_agent.agent import Agent
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

dqn_agent = Agent(6, 6)
#dqn_agent.model.load_weights('../src/single_agent/recent_weights.hdf5')
#dqn_agent.model_target.load_weights('../src/single_agent/recent_weights.hdf5')


state = np.asarray([[ 7, 20 , 1 ,20 , 1,  1],
 [ 1, 20,  1 , 0 , 0 , 0],
 [20 , 1 , 0 , 0 , 0 , 0],
 [ 7, 20,  1, 20 , 1,  1],
 [ 1 ,20 , 1 , 0 , 0 , 0],
 [ 7,  1 , 1 , 0 , 0 , 0]])


# Forward alignment data through the model to obtain feature maps
feature_map = dqn_agent.model.predict(state.reshape((1,6,6,1))).reshape(6,6)
predicted_action = np.argmax(feature_map)

print(predicted_action)
feature_map.reshape(6,6)



min_val = np.min(feature_map)
max_val = np.max(feature_map)
scaled_feature_map = (feature_map - min_val) / (max_val - min_val)

# Create a custom colormap
colors = [(0, 0, 0), (1, 1, 1)]  # Black to White
cmap_name = 'custom_cmap'
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)

# Thresholding the scaled feature map
scaled_feature_map_thresholded = np.where(scaled_feature_map < 0.5, 0, 1)

# Plotting the scaled and thresholded feature map with the custom colormap
plt.imshow(scaled_feature_map_thresholded, cmap=custom_cmap)
plt.colorbar()  # Add a color bar to indicate values
plt.title('Scaled and Thresholded Feature Map Heatmap')
plt.xlabel('Width')
plt.ylabel('Height')
plt.show()