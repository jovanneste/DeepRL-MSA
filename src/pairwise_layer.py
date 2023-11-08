import tensorflow as tf

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
            "arg1": self.arg1,
            "arg2": self.arg2,
        })
        return config