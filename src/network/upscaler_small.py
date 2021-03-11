import tensorflow as tf
from tensorflow.keras import Model, layers

class UpscalerSmall(Model):

    def __init__(self):
        super(UpscalerSmall, self).__init__()
    
    def feature_map_frame(self, input, features, min_kernels, max_kernels):
        feature = layers.SeparableConv2D(features, min_kernels, strides = 1, padding = 'same')(input)
        feature = layers.SeparableConv2D(features, max_kernels, strides = 1, padding = 'same')(feature)

        feature = layers.BatchNormalization()(feature)
        feature = layers.PReLU(shared_axes = [1, 2])(feature)

        return feature

    def call(self, fi2, fi1, fi0, name = 'dlss'):
        fi2 = layers.Input(shape = fi2, name = 'fi2')
        fi1 = layers.Input(shape = fi1, name = 'fi1')
        fi0 = layers.Input(shape = fi0, name = 'fi0')

        # feature maps
        fi2fm = self.feature_map_frame(fi2, 3, 1, 3)
        fi1fm = self.feature_map_frame(fi1, 3, 1, 3)
        fi0fm = self.feature_map_frame(fi0, 3, 1, 3)

        net = layers.concatenate([ fi2fm, fi1fm, fi0fm ])

        net = layers.SeparableConv2D(3, 1, strides = 1, padding = 'same')(net)
        net = layers.LeakyReLU(0.8)(net)
        
        net_1 = layers.BatchNormalization()(net)
        net_1 = layers.SeparableConv2D(16, 1, strides = 1, padding = 'same')(net_1)
        net_1 = layers.PReLU(shared_axes = [1, 2])(net_1)

        net_2 = layers.BatchNormalization()(net)
        net_2 = layers.SeparableConv2D(16, 1, strides = 1, padding = 'same')(net_2)
        net_2 = layers.PReLU(shared_axes = [1, 2])(net_2)

        net_3 = layers.BatchNormalization()(net)
        net_3 = layers.SeparableConv2D(16, 1, strides = 1, padding = 'same')(net_3)
        net_3 = layers.PReLU(shared_axes = [1, 2])(net_3)

        net = layers.concatenate([ net_1, net_2, net_3 ])
        net = tf.nn.depth_to_space(net, 4)

        return Model(inputs = [ fi2, fi1, fi0 ], outputs = net, name = name)

