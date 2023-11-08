from single_agent.agent_memory import Memory
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Input, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
tf.keras.utils.disable_interactive_logging()
import numpy as np
from pairwise_layer import PairwiseConv1D

class Agent():
    def __init__(self, no_seq, length):
        self.memory = Memory(1000)
        self.no_sequences = no_seq
        self.seq_length = length
        self.epsilon = 0.99
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.95/1000
        self.gamma = 0.99
        self.learning_rate = 1e-4
        self.model = self._build_model()
        self.model_target = clone_model(self.model)
        self.total_timesteps = 0
        self.memory_threshold = 256
        self.batch_size = 32
        self.learns = 0
        
        
    def _build_model(self):
        model = Sequential()
        model.add(Input((self.no_sequences, self.seq_length, 1)))
        
        model.add(PairwiseConv1D(filters=8, kernel_size=5))
        
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv1D(filters=self.no_sequences*self.seq_length, kernel_size=1, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        
        model.add(GlobalAveragePooling2D())
            
        
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer, loss=tf.keras.losses.Huber())
        model.summary()
        print('\nAgent Initialised\n')
        return model
    
    
    def index_to_coords(self, index):
        return (index%self.no_sequences, index//self.seq_length)
    
    def coords_to_index(self, coords):
        x,y = coords
        return (y*self.seq_length)+x
        
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return (np.random.randint(0, self.seq_length), np.random.randint(0, self.no_sequences))
        
        action_values = self.model.predict(state.reshape(1,self.no_sequences,self.seq_length,1))
        return self.index_to_coords(np.argmax(action_values))
    
    
    def score(self, old_state, new_state):
        if (old_state == new_state).all():
            return -4

        n_rows_old, n_cols_old = old_state.shape
        n_rows_new, n_cols_new = new_state.shape

        score = 0.0

        for i in range(n_rows_old):
            for j in range(n_cols_old):
                for k in range(n_rows_new):
                    for l in range(n_cols_new):
                        if new_state[k, l] == 0 or old_state[i, j] == 0:
                            pair_score = -0.5
                        elif old_state[i, j] == new_state[k, l]:
                            pair_score = 1
                        else:
                            pair_score = 0
                        score += pair_score

        return score / (n_rows_old * n_cols_old * n_rows_new * n_cols_new)

    
    
    def step(self, state, coords):
#        split state - sliding window heuristic 
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
        labels = self.model.predict(np.array(states).reshape(self.batch_size,self.no_sequences,self.seq_length,1))
        next_state_values = self.model_target.predict(np.array(next_states).reshape(self.batch_size,self.no_sequences,self.seq_length,1))
        
        for i in range(self.batch_size):
            labels[i][self.coords_to_index(actions[i])] = rewards[i] + (self.gamma * max(next_state_values[i]))
        
        self.model.fit(np.array(states), labels, batch_size=self.batch_size, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        self.learns += 1
        
        if self.learns % 350 == 0:
            self.model_target.set_weights(self.model.get_weights())
            print('\nTarget model updated')
                  