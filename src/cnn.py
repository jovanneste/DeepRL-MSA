import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

global pretrained_model
pretrained_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
with open("resnet.pkl", "wb") as file:
    print("Saving ResNet model...")
    pickle.dump(pretrained_model, file)


def process(data):
    state = np.stack((data,)*3, axis=-1)  
    state = tf.image.resize(state, (224, 224))  
    state = preprocess_input(state)
    return np.expand_dims(state, axis=0)



def get_features(state):
    with open("resnet.pkl", "rb") as file:
        pretrained_model = pickle.load(file)
    if pretrained_model:
        return pretrained_model.predict(process(state))
    else:
        print("Model not loaded...")
        return 0
