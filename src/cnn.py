import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

global pretrained_model
pretrained_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
with open("resnet.pkl", "wb") as file:
    pickle.dump(pretrained_model, file)


def get_features(state):
    with open("resnet.pkl", "rb") as file:
        pretrained_model = pickle.load(file)
    state = np.stack((state,)*3, axis=-1)  
    state = tf.image.resize(state, (224, 224))  
    state = preprocess_input(state)
    state = np.expand_dims(state, axis=0)
    return pretrained_model.predict(state)
