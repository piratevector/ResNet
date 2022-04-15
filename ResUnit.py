from tensorflow.keras.layers import BatchNormalization, Dense, Activation, LeakyReLU, Flatten, Dropout, MaxPool2D, Conv2D, AveragePooling2D, InputLayer, GlobalAvgPool2D, Layer
import keras

def addResBlock(filters,strides,num):
  model.add(ResUnit(filters=filters,strides=strides))
  for i in range(num-1):
    model.add(ResUnit(filters=filters))
  return

@tf.keras.utils.register_keras_serializable()
class ResUnit(Layer):
  
    def __init__(self, filters, kernel_size=3, strides=1, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = keras.activations.get(activation)
        #self.kernel_initializer = kernel_initializer

        self.layers = [
            Conv2D(filters=self.filters,
                   kernel_size=self.kernel_size,
                   strides=self.strides,
                   padding='SAME',
                   use_bias=False),
            BatchNormalization(),
            self.activation,
            Conv2D(filters=self.filters,
                   kernel_size=self.kernel_size,
                   padding='SAME',
                   use_bias=False),
            BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                Conv2D(filters=self.filters, 
                       kernel_size=1,
                       strides=self.strides,
                       padding='SAME',
                       use_bias=False),
                BatchNormalization()]

    def call(self, inputs):
        Z = inputs
        for layer in self.layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

        # z, skip = inputs, inputs
        # for layer in self.layers:
        #   z = layer(z)
        # skip = self.skip_layers[0](skip)
        # skip = BatchNormalization()(skip)
        # return self.activation(z+skip)
        
    def get_config(self):
      config = super().get_config()
      config.update({
          'filters': self.filters,
          'kernel_size': self.kernel_size,
          'strides': self.strides,
          'activation': self.activation,
          #'kernel_initializer': self.kernel_initializer
      })
      return config


# import keras
# import tensorflow as tf
# from keras.layers import Layer, Activation, BatchNormalization, Conv2D
# from functools import partial

# DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1, padding="SAME", use_bias=False)

# @tf.keras.utils.register_keras_serializable()
# class ResUnit(keras.layers.Layer):
#     DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1, padding="SAME", use_bias=False)
#     def __init__(self, filters, strides=1, activation="relu", **kwargs):
#         super().__init__(**kwargs)
#         self.filters = filters
#         self.strides = strides
#         self.activation = keras.activations.get(activation)
#         self.main_layers = [
#             DefaultConv2D(filters, strides=strides),
#             keras.layers.BatchNormalization(),
#             self.activation,
#             DefaultConv2D(filters),
#             keras.layers.BatchNormalization()]
#         self.skip_layers = []
#         if strides > 1:
#             self.skip_layers = [
#                 DefaultConv2D(filters, kernel_size=1, strides=strides),
#                 keras.layers.BatchNormalization()]

#     def get_config(self):
#       config = super().get_config()
#       config.update({
#           'filters': self.filters,
#           'strides': self.strides,
#           'activation': self.activation
#       })
#       return config

#     def call(self, inputs):
#       Z = inputs
#       for layer in self.main_layers:
#           Z = layer(Z)
#       skip_Z = inputs
#       for layer in self.skip_layers:
#           skip_Z = layer(skip_Z)
#       return self.activation(Z + skip_Z)
