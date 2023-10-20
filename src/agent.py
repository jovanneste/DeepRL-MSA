from agent_memory import Memory
from generator import *
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np

class Agent():
    def __init__(self):
        self.memory = Memory()
        self.epsilon = 0.1
        self.epsilon_decay = 0.9/100000
        self.gamma = 0.95
        self.learning_rate = 1e-4
        self.model = self._build_model()
        self.model_target = clone_model(self.model)
        self.total_timesteps = 0
        
    def _build_model(self):
        model = Sequential()
        model.add(Input((5, 5, 1)))
        model.add(Conv2D(filters=4, kernel_size=(3, 3), activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Dense(25, activation='linear'))
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer, loss=tf.keras.losses.Huber())
        model.summary()
        print('\nAgent Initialised\n')
        return model
    
    def index_to_coords(self, index):
        return (index%5, index//5)
                  
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return (np.random.randint(0, 5), np.random.randint(0, 5))
        
        action_values = self.model.predict(state)
        print(np.argmax(action_values))
        return self.index_to_coords(np.argmax(action_values))
                  
    
a = Agent()
s = generate_sequence(5, 5, 0.2, 0.4)
action = a.get_action(s.reshape(1,5,5,1))

print(s)
print(action)
                  