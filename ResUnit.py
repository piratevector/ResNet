import keras
import tensorflow as tf
from keras.layers import Layer, Activation, BatchNormalization, Conv2D
from functools import partial

DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1, padding="SAME", use_bias=False)

@tf.keras.utils.register_keras_serializable()
class ResUnit(keras.layers.Layer):
    import keras
    import tensorflow as tf
    from keras.layers import Layer, Activation, BatchNormalization, Conv2D
    from functools import partial
    DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1, padding="SAME", use_bias=False)
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                keras.layers.BatchNormalization()]

    def get_config(self):
      config = super().get_config()
      config.update({
          'filters': self.filters,
          'strides': self.strides,
          'activation': self.activation
      })
      return config

    def call(self, inputs):
      Z = inputs
      for layer in self.main_layers:
          Z = layer(Z)
      skip_Z = inputs
      for layer in self.skip_layers:
          skip_Z = layer(skip_Z)
      return self.activation(Z + skip_Z)
