import tensorflow as tf
import argparse
import numpy as np
import os, time, cv2, random

from lib.model import create_model, create_pre_gen_model, create_upscaler_model

physical_devices = tf.config.list_physical_devices('GPU')
try: tf.config.experimental.set_memory_growth(physical_devices[0], True)
except: pass

class OSSS:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.image = self.kwargs['image']
        self.output = self.kwargs['output']

        self.model = create_upscaler_model(compile = False, small = False)
        self.model.load_weights('data/model/weights.h5')

        self.upscale()

    def prepare_image(self):
        fi3 = None
        fi2 = None
        fi1 = None
        x = None

        image = cv2.imread(self.image)
        image = image / 255.
        
        image = np.reshape(image, (1, image.shape[0], image.shape[1], 3))


        print(image.shape)

        fi3 = [image]
        fi2 = [image]
        fi1 = [image]
        x = [image]

        return { 'fi2': fi2, 'fi1': fi1, 'fi0': x }

    def upscale(self):
        if self.image is not None:
            try:
                image = self.prepare_image()
                
                times = []
                for i in range(100):
                    s = time.time()
                    preds = self.model([ image ])
                    e = time.time()
                    times.append(e - s)

                times = times[1:]
                
                print('[ OK ] Took {:,.2f}ms to upscale'.format((e - s) * 1000))
                print('[ OK ] Took {:,.2f}ms on average'.format(np.average(times) * 1000))
                print(preds[0].shape)
                image = np.array(preds[0])
                image = image * 255
                cv2.imwrite(self.output, image)
            except Exception as e:
                print('An error occured: %s' % str(e))
        else: print('Please provide a valid image!')

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--image', type=str, default=None, help = 'Image to upscale')
    args.add_argument('--output', type=str, default='upscaled.png', help = 'Output file for upscaled image')
    args = args.parse_args()

    OSSS(**vars(args))