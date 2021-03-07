import tensorflow as tf
from tensorflow.keras import Model, layers

class Upscaler(Model):

    def __init__(self):
        super(Upscaler, self).__init__()

    def build_details(self, input, features = 3, kernels = 1, activation = 'tanh'):
        feature = layers.SeparableConv2D(features, kernels, strides = 1, padding = 'same', activation = activation)(input)
        feature = layers.BatchNormalization()(feature)
        feature = layers.PReLU(shared_axes = [1, 2])(feature)

        return feature

    def call(self, fi2, fi1, fi0, name = 'dlss'):
        fi2 = layers.Input(shape = fi2, name = 'fi2')
        fi1 = layers.Input(shape = fi1, name = 'fi1')
        fi0 = layers.Input(shape = fi0, name = 'fi0')

        # large features
        fi2_lf = self.build_details(fi2, 6, 3, 'relu')
        fi1_lf = self.build_details(fi1, 6, 3, 'relu')
        fi0_lf = self.build_details(fi0, 6, 3, 'relu')

        # medium features
        fi2_mf = self.build_details(fi2, 3, 3, 'relu')
        fi1_mf = self.build_details(fi1, 3, 3, 'relu')
        fi0_mf = self.build_details(fi0, 3, 3, 'relu')

        # concatenate features
        fi2_cf = self.build_details(fi2, 3, 1, 'tanh')
        fi1_cf = self.build_details(fi1, 3, 1, 'tanh')
        fi0_cf = self.build_details(fi0, 3, 1, 'tanh')

        fi2f = layers.concatenate([ fi2_lf, fi2_mf, fi2_cf ], name = 'fi2f_concatenate')
        fi1f = layers.concatenate([ fi1_lf, fi1_mf, fi1_cf ], name = 'fi1f_concatenate')
        fi0f = layers.concatenate([ fi0_lf, fi0_mf, fi0_cf ], name = 'fi0f_concatenate')

        base_net = layers.SeparableConv2D(24, 3, strides = 1, padding = 'same')(fi0f)
        base_net = layers.BatchNormalization()(base_net)
        base_net = layers.LeakyReLU(1)(base_net)

        net = layers.concatenate([ fi2f, fi1f, base_net ])
        net = tf.nn.depth_to_space(net, 4)

        return Model(inputs = [ fi2, fi1, fi0 ], outputs = net, name = name)