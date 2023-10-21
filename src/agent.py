from agent_memory import Memory
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np

class Agent():
    def __init__(self):
        self.memory = Memory(100)
        self.epsilon = 0.99
        self.epsilon_decay = 0.9/100000
        self.gamma = 0.95
        self.learning_rate = 1e-4
        self.model = self._build_model()
        self.model_target = clone_model(self.model)
        self.total_timesteps = 0
        self.memory_threshold = 50
        
        
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
        
        action_values = self.model.predict(state.reshape(1,5,5,1))
        return self.index_to_coords(np.argmax(action_values))
    
    
    def score(self, old_state, new_state):
        if (old_state==new_state).all():
            return -4
        n = new_state.shape[0]  # Assuming it's a square array
        score = 0.0

        for i in range(n):
            for j in range(n):
                for k in range(i, n):
                    for l in range(j, n):
                        if new_state[i, j] == 0 or new_state[k, l] == 0:
                            # Apply a gap penalty for any element being 0
                            pair_score = -0.5
                        elif new_state[i, j] == new_state[k, l]:
                            # Pair score for matching elements
                            pair_score = 1
                        else:
                            # A different score for non-matching non-zero elements
                            pair_score = 0 
                        score += pair_score

        return score/(n*n)
    
    
    def step(self, state, coords):
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
#        needs to return done too
        return new_state, self.score(state, new_state), False
                  

                  