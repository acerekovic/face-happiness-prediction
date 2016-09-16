from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.misc import imread

import _old.data_loader as data_loader
from googlenet_model import GoogLeNet_FC
from vgg_model import VGG16

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('img_src','./data/fake_HAPPEI/images', """input images""")
tf.app.flags.DEFINE_string('weights','./data/models/gnet-fc.ckpt-6744', """Model which restores the weights.""") #Leave empty if training from scratch
tf.app.flags.DEFINE_string('model','', """TF model""")

# Global constants describing the data set.
NUM_CLASSES = 6
NUM_CHANNELS = 3

def demo():

    # List files in directory
    imagefiles = [f for f in os.listdir(FLAGS.img_src) if os.path.isfile(os.path.join(FLAGS.img_src, f))]
    # Load model
    # Build an initialization operation to run below.
    if FLAGS.model == 'GNET':
        global IMAGE_SIZE
        IMAGE_SIZE = 256  # global variable for GoogLeNet
        model = GoogLeNet_FC(IMAGE_SIZE, NUM_CHANNELS, NUM_CLASSES)
    elif FLAGS.model == 'VGG16':
        global IMAGE_SIZE
        IMAGE_SIZE = 224  # change global variable for VGG
        FLAGS.weights = './data/models/vgg16.ckpt-6601'
        model = VGG16(IMAGE_SIZE, NUM_CHANNELS, NUM_CLASSES)
    else:
        print('Please provide a proper model, --model "VGG16" or "GNET"')
        exit(1)


    # Start running operations on the Graph.
    sess = tf.InteractiveSession()

    # initialize Saver
    saver = tf.train.Saver(tf.trainable_variables())
    tf.initialize_all_variables().run()
    print('Initialized!')

    fig, ax = plt.subplots()
    plt.ion()
    labels = ['neutral','small smile','big smile','small laugh','big laugh','thrilled']
    predictions = {}

    if FLAGS.weights:
        print('Restoring the weights from the model...' +  FLAGS.weights)
        saver.restore(sess, FLAGS.weights)

        for imagefile in imagefiles:
            try:
                imagedata = imread(os.path.join(FLAGS.img_src,imagefile))
                # Compute probabilities for each happiness label
                happ_prob = model.infer(sess,imagedata)
                # Compute predicted label
                happ_idx = int(np.argmax(happ_prob, 1))
                length, width, ch = np.shape(imagedata)
                ax.imshow(imagedata, extent=[-width / 2, width / 2, -length / 2, length / 2])
                ax.set_title('Predicted hsappiness intensity: ' + labels[happ_idx])
                plt.show()
                plt.pause(3)
                predictions[imagefile] = labels[happ_idx]
            except IOError:
                print('Can not open the image')

    json.dump(predictions, open('./data/results.json', 'w'))


def main(**kwargs):
    # Override default
    FLAGS.model = kwargs["model"]
    demo()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get a program and run it with input')
    parser.add_argument('--model', type=str, default='', help='Which model to use')
    args = parser.parse_args()
    main(**vars(args))

