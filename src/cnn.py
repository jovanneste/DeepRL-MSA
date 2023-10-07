import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.applications import ResNet50

global pretrained_model
pretrained_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
with open("resnet.pkl", "wb") as file:
    print("Saving ResNet model...")
    pickle.dump(pretrained_model, file)