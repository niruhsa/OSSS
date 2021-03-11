import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model, layers
from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph

from network.upscaler import Upscaler
from network.upscaler_small import UpscalerSmall

def create_pre_gen_model(version = 2):
    gen = Generator(version)
    gen = gen((None, None, 3))

    gen.compile(loss = l2_norm, optimizer = tf.keras.optimizers.Adam(lr = 1e-3))
    #gen.load_weights('data/model/weights.h5')

    gen.summary()

    return gen

def create_upscaler_model(input_width = None, input_height  = None, compile = True, small = False):
    if small: upscaler = UpscalerSmall()
    else: upscaler = Upscaler()

    upscaler = upscaler((input_height, input_width, 3), (input_height, input_width, 3), (input_height, input_width, 3))
    upscaler.summary()

    if compile:
        upscaler.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(lr = 1e-3))
        flops = get_flops(upscaler)
        print('Number of flops for model: {:,}'.format(flops))
    
    if small:
        pass #upscaler.load_weights('data/model/upscaler_weights.h5')
    else:
        pass #upscaler.load_weights('data/model/upscaler_weights.h5')

    tf.keras.utils.plot_model(upscaler, 'upscaler.png')

    return upscaler

def create_model(dis_input_size, version = 2):
    gen = Generator()
    denoiser = Denoiser()

    gen = gen((None, None, 3))
    den = denoiser((None, None, 3))

    gen.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(lr = 1e-3))
    den.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(lr = 1e-3))
    gen.summary()
    den.summary()

    model = tf.keras.models.Sequential()
    model.add(gen)
    model.add(den)

    model.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(lr = 1e-3))
    model.summary()

    gen.load_weights('data/model/generator_weights.h5')
    den.load_weights('data/model/denoiser_weights.h5')
    model.load_weights('data/model/model_weights.h5')

    return gen, den, model

def mse(y_true, y_pred):
    return tf.keras.losses.MSE(y_true, y_pred)

def l2_norm(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def stack_loss(y_true, y_pred):
    return (mse(y_true, y_pred) + l2_norm(y_true, y_pred))

def get_flops(model):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops