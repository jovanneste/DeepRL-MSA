import numpy as np
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, Conv1D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import seq_generator
from pairwise_layer import PairwiseConv1D


class Model():
    def __init__(self):
        self.learning_rate = 1e-4
        self.model = self.build_model()
        
    def build_model(self):
        model = Sequential()
        model.add(Input((9,7,1)))
        model.add(PairwiseConv1D(filters=8, kernel_size=5))
        model.add(Conv1D(filters=16, kernel_size=3, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Conv1D(filters=32, kernel_size=2, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer, loss=tf.keras.losses.Huber())
        return model
    
    def forward(self, state):
        out = self.model.predict(state)
        print(out.shape)
        
    def summary(self):
        model.summary()
    


        
        
model = Model()

sequences = seq_generator.generate(9,7,4,0.2,0.4)
print(sequences)
model.forward(sequences.reshape(1,9,7,1))

        