from agent_memory import Memory
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
tf.keras.utils.disable_interactive_logging()
import numpy as np

class Agent():
    def __init__(self):
        self.memory = Memory(2500)
        self.epsilon = 0.9
        self.epsilon_min = 0.2
        self.epsilon_decay = 0.8/1000
        self.gamma = 0.95
        self.learning_rate = 1e-4
        self.model = self._build_model()
        self.model_target = clone_model(self.model)
        self.total_timesteps = 0
        self.memory_threshold = 100
        self.batch_size = 32
        self.learns = 0
        
        
    def _build_model(self):
        model = Sequential()
        model.add(Input((5, 5, 1)))
        model.add(Conv2D(filters=4, kernel_size=(3, 3), activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Dense(25, activation='linear'))
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer, loss=tf.keras.losses.Huber())
        model.summary()
        print('\nAgent Initialised\n')
        return model
    
    
    def index_to_coords(self, index):
        return (index%5, index//5)
    
    def coords_to_index(self, coords):
        x,y = coords
        return (y*5)+x
        
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
                            pair_score = -0.5
                        elif new_state[i, j] == new_state[k, l]:
                            pair_score = 1
                        else:
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


    def learn(self):
        states, next_states, actions, rewards = self.memory.sample(self.batch_size)
        labels = self.model.predict(np.array(states).reshape(self.batch_size,5,5,1))
        next_state_values = self.model_target.predict(np.array(next_states).reshape(self.batch_size,5,5,1))
        
        for i in range(self.batch_size):
            labels[i][self.coords_to_index(actions[i])] = rewards[i] + (self.gamma * max(next_state_values[i]))
        
        self.model.fit(np.array(states), labels, batch_size=self.batch_size, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        self.learns += 1
        
        if self.learns % 100 == 0:
            self.model_target.set_weights(self.model.get_weights())
            print('\nTarget model updated')
                  