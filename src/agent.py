from agent_memory import Memory
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np

class Agent():
    def __init__(self):
        self.memory = Memory()
        self.epsilon = 0.99
        self.epsilon_decay = 0.9/100000
        self.gamma = 0.95
        self.learning_rate = 1e-4
        self.model = self._build_model()
        self.model_target = clone_model(self.model)
        self.total_timesteps = 0
        
    def _build_model(self):
        model = Sequential()
        model.add(Input((5,5)))
        model.add(Conv2D(filters = 32,kernel_size = (8,8),strides = 4,data_format="channels_last", activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2))
        model.add(Flatten())
        model.add(Dense(512,activation = 'relu', kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Dense(25, activation = 'linear'))
        optimizer = Adam(self.learn_rate)
        model.compile(optimizer, loss=tf.keras.losses.Huber())
        model.summary()
        print('\nAgent Initialized\n')
        return model
                  
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return (np.random.randint(0, 4), np.random.randint(0, 4))
        return np.argmax(self.model.predict(state))
                  
    
                  