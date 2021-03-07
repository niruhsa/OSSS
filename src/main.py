import tensorflow as tf
import argparse
import numpy as np
import os, time, cv2, random

from lib.model import create_model, create_pre_gen_model, create_upscaler_model, create_denoiser_model

physical_devices = tf.config.list_physical_devices('GPU')
try: tf.config.experimental.set_memory_growth(physical_devices[0], True)
except: pass

class OSSS:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.scale = self.kwargs['scale']
        self.data = self.kwargs['data']
        self.input_size = self.calculate_input_size()
        self.dataset_size = 1024
        self.batch_size = self.kwargs['batch_size']
        self.small = self.kwargs['small']

        if self.small: self.train_small()
        else: self.train()

    def calculate_input_size(self):
        size = str(self.kwargs['input_size']).lower().split('x')
        return (int(size[0]), int(size[1]))

    def get_dataset_frame_batch(self, batch_size = 4):
        s = time.time()
        fi4_arr = []
        fi3_arr = []
        fi2_arr = []
        fi1_arr = []
        x = []
        y = []
        for _, _, files in os.walk(self.data):
            for i in range(batch_size):
                try:
                    file = random.choice(files)
                    filename = int(str(file).split(".png")[0])
                    if filename >= 3:
                        fi1 = "%09d.png" % (filename - 1)
                        fi2 = "%09d.png" % (filename - 2)
                        fi3 = "%09d.png" % (filename - 3)
                        fi4 = "%09d.png" % (filename - 4)

                        fi1 = os.path.join(self.data, fi1)
                        fi2 = os.path.join(self.data, fi2)
                        fi3 = os.path.join(self.data, fi3)
                        fi4 = os.path.join(self.data, fi4)
                        file = os.path.join(self.data, file)

                        fi1 = cv2.imread(fi1)
                        fi1 = cv2.resize(fi1, (self.input_size[0] // int(self.scale), self.input_size[1] // int(self.scale)))

                        fi2 = cv2.imread(fi2)
                        fi2 = cv2.resize(fi2, (self.input_size[0] // int(self.scale), self.input_size[1] // int(self.scale)))

                        fi3 = cv2.imread(fi3)
                        fi3 = cv2.resize(fi3, (self.input_size[0] // int(self.scale), self.input_size[1] // int(self.scale)))

                        fi4 = cv2.imread(fi4)
                        fi4 = cv2.resize(fi4, (self.input_size[0] // int(self.scale), self.input_size[1] // int(self.scale)))

                        _file = cv2.imread(file)
                        _file = cv2.resize(_file, (self.input_size[0] // int(self.scale), self.input_size[1] // int(self.scale)))

                        __file = cv2.imread(file)
                        __file = cv2.resize(__file, (self.input_size[0], self.input_size[1]))

                        fi4_arr.append(fi4 / 255.)
                        fi3_arr.append(fi3 / 255.)
                        fi2_arr.append(fi2 / 255.)
                        fi1_arr.append(fi1 / 255.)

                        x.append(_file / 255.)
                        y.append(__file / 255.)
                    else:
                        i = i - 1
                except: i = i - 1

        return np.asarray(fi4_arr), np.asarray(fi3_arr), np.asarray(fi2_arr), np.asarray(fi1_arr), np.asarray(x), np.asarray(y), time.time() - s

    def create_training_samples(self, epoch, loss, small = False):
        try:
            fi4, fi3, fi2, fi1, x, y, total_time = self.get_dataset_frame_batch(batch_size = 1)
            train_x = { 'fi2': fi2, 'fi1': fi1, 'fi0': x }
            sample = y[0]
            sample_predict = cv2.resize(sample, (sample.shape[1] // self.scale, sample.shape[0] // self.scale))
            s = time.time()
            predictions = self.upscaler.predict(train_x)
            e = time.time()

            if small:
                scale = str(self.scale) + "x_SMOL"
            else:
                scale = str(self.scale) + "x"
            
            random_sample = predictions[0] * 255
            os.makedirs('data/training_images/{}'.format(scale), exist_ok = True)
            cv2.imwrite('data/training_images/{}/EPOCH_{}__LOSS_{:,.5f}_GEN.png'.format(scale, epoch, loss), random_sample)
            cv2.imwrite('data/training_images/{}/EPOCH_{}__LOSS_{:,.5f}_ORI.png'.format(scale, epoch, loss), sample * 255)

            return e - s
        except:
            return 0

    def train(self, epochs = 1024 * 1024):
        self.upscaler = create_upscaler_model(self.input_size[0] // int(self.scale), self.input_size[1] // int(self.scale))
        for i in range(epochs):
            epoch = i + 1
            fi4, fi3, fi2, fi1, x, y, total_time = self.get_dataset_frame_batch(batch_size = self.batch_size)
            train_x = { 'fi2': fi2, 'fi1': fi1, 'fi0': x }
            train_y = { 'tf.nn.depth_to_space': y }
            
            s = time.time()
            upscaler_loss = []
            for step in range(self.dataset_size // self.batch_size):
               loss = self.upscaler.train_on_batch(train_x, train_y)
               upscaler_loss.append(loss)

            loss = np.average(upscaler_loss)
            total_time = (time.time() - s)
            query_time = self.create_training_samples(epoch, loss)
            print('[ EPOCH #{:,}/{:,} ] Upscaler ({}x) Loss: {:,.6f} || Took {:,.3f}s for the epoch'.format(epoch, epochs, self.scale, loss, total_time))

            os.makedirs('data/model/{}x'.format(self.scale), exist_ok = True)
            self.upscaler.save_weights('data/model/{}x/niru_upscaler_epoch_{}.h5'.format(self.scale, epoch))

    def train_small(self, epochs = 1024 * 1024):
        self.upscaler = create_upscaler_model(self.input_size[0] // int(self.scale), self.input_size[1] // int(self.scale), small = True)
        for i in range(epochs):
            epoch = i + 1
            fi4, fi3, fi2, fi1, x, y, total_time = self.get_dataset_frame_batch(batch_size = self.batch_size)
            train_x = { 'fi2': fi2, 'fi1': fi1, 'fi0': x }
            train_y = { 'tf.nn.depth_to_space': y }
            
            s = time.time()
            upscaler_loss = []
            for step in range(self.dataset_size // self.batch_size):
               loss = self.upscaler.train_on_batch(train_x, train_y)
               upscaler_loss.append(loss)

            loss = np.average(upscaler_loss)
            total_time = (time.time() - s)
            query_time = self.create_training_samples(epoch, loss, small = True)
            print('[ EPOCH #{:,}/{:,} ] Upscaler Small ({}x) Loss: {:,.6f} || Took {:,.3f}s for the epoch'.format(epoch, epochs, self.scale, loss, total_time))

            os.makedirs('data/model/{}x'.format(self.scale), exist_ok = True)
            self.upscaler.save_weights('data/model/{}x/niru_upscaler_small_epoch_{}.h5'.format(self.scale, epoch))

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--scale', type=int, default=2, help = 'Scale to use, either 2, 4, or 8 is accepted')
    args.add_argument('--data', type=str, default='data/dataset', help = 'Training data to use')
    args.add_argument('--input_size', type=str, default='1920x1080', help = 'Input size in the format of WIDTHxHEIGHT, e.g 1920x1080')
    args.add_argument('--batch_size', type=int, default=32, help = 'Batch Size')
    args.add_argument('--small', type=bool, default=False, help = 'Train small network instead of regular size')
    args = args.parse_args()

    OSSS(**vars(args))