"""
Author: Joachim Vanneste
Date: 10 Apr 2024
Description: Novel pairwise layer
"""

import tensorflow as tf
import numpy as np

class PairwiseConv1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(PairwiseConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight("kernel",
                                      shape=(self.kernel_size, input_dim, self.filters),
                                      initializer="glorot_uniform",
                                      trainable=True)

    def call(self, inputs):
        output = []
        for i in range(inputs.shape[1] - self.kernel_size + 1):
            pairs = inputs[:, i:i + self.kernel_size, :, :]
            convolution = tf.reduce_sum(pairs * self.kernel, axis=[1, 2])
            output.append(convolution)
        output = tf.stack(output, axis=1)
        # Add a channel dimension
        output = tf.expand_dims(output, -1)
        return output
    
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
        })
        return config


if __name__ == '__main__':
    print("yep")
    pairwise_conv_layer = PairwiseConv1D(filters=2, kernel_size=2)

    # Example input matrix
    input_matrix = np.array([[[1, 2], [3, 4], [5, 6], [7, 8]]], dtype=np.float32)
    input_tensor = tf.constant(input_matrix)

    # Reshape input tensor to add channel dimension
    input_tensor = tf.expand_dims(input_tensor, axis=-1)

    # Apply the pairwise_conv_layer to the input
    output_tensor = pairwise_conv_layer(input_tensor)

    print(output_tensor)