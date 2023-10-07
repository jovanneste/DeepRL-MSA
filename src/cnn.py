import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
np.random.seed(42)

class MyCNN:
    def __init__(self, input_shape):
        self.model = self.build_model(input_shape)
        
    def build_model(self, input_shape):
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
            # no max pooling layer for now
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax') 
        ])
        return model
    
    def extract_features(self, input_data):
        input_data = np.expand_dims(input_data, axis=0) 
        features = self.model.predict(input_data)
        return features


if __name__ == "__main__":
    input_data = np.array([['1', '1', '2', '4'],
                           ['5', '6', '7', '8'],
                           ['9', '1', '0', '1']], dtype=np.float32)  
    
    input_data = np.reshape(input_data, (1, input_data.shape[0], input_data.shape[1], 1))


    cnn = MyCNN(input_shape=input_data.shape)
    features = cnn.extract_features(input_data)
    print("Features:", features)
