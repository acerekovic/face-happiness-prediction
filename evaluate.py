from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from googlenet_model import GoogLeNet_FC
import argparse
from vgg_model import VGG16
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from utils.misc import plot_confusion_matrix

import utils.data_loader as data_loader

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir','', """input dataset""")
tf.app.flags.DEFINE_string('weights','', """Model which restores the weights.""") #Leave empty if training from scratch
tf.app.flags.DEFINE_string('model','', """TF model""")

# Global constants describing the data set.
NUM_CLASSES = 6
NUM_CHANNELS = 3
BATCH_SIZE = 32
def eval():


    # Build an initialization operation to run below.
    # Uncomment the model one wants to use

    # Build an initialization operation to run below.
    if FLAGS.model == 'GNET':
        global IMAGE_SIZE
        IMAGE_SIZE = 256  # global variable for GoogLeNet
        model = GoogLeNet_FC(IMAGE_SIZE, NUM_CHANNELS, NUM_CLASSES, BATCH_SIZE)
    elif FLAGS.model == 'VGG16':
        global IMAGE_SIZE
        IMAGE_SIZE = 224  # global variable for VGG
        model = VGG16(IMAGE_SIZE, NUM_CHANNELS, NUM_CLASSES, BATCH_SIZE)
    else:
        print('Please provide a proper model')
        exit(1)

    # Generate a validation set
    val_loader = data_loader.Loader(FLAGS.data_dir, IMAGE_SIZE, NUM_CHANNELS, BATCH_SIZE, 'validation', 0)

    # Start running operations on the Graph.
    sess = tf.InteractiveSession()


    # initialize Saver
    saver = tf.train.Saver(tf.trainable_variables())
    tf.initialize_all_variables().run()
    print('Initialized!')

    # If model specified evaluate it
    if FLAGS.weights:
        print('Restoring the weights from the model...' +  FLAGS.weights)
        saver.restore(sess, FLAGS.weights)

        val_acc = 0
        y_test = []
        y_pred = []

        for bv in range(val_loader.num_batches):
            val_x, val_y = val_loader.next_batch()
            y_test = y_test + list(val_y)
            feed_dict = {model.x: val_x,
                         model.y: val_y,
                         model.dropout_keep_prob: 1}
            # Run the graph and fetch some of the nodes.
            vpred, val_batch_acc = sess.run(
                [model.predictions, model.acc],
                feed_dict=feed_dict)
            vpred_2 = np.argmax(vpred,1)
            y_pred = y_pred + list(vpred_2)
            val_acc += val_batch_acc  # accumulate acc for every setp

        print('Validation accuracy: %.6f' % (val_acc/val_loader.num_batches*100))
        #evalm.evaluate_linear_regression(FLAGS.weights,val_pred_flatten,val_labels_flatten)
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)
        print('Confusion matrix, without normalization')
        print(cm)
        plt.figure()
        plot_confusion_matrix(cm)

        # Normalize the confusion matrix by row (i.e by the number of samples
        # in each class)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
        print(cm_normalized)
        plt.figure()
        plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

        plt.show()


def fatal_error(msg):
    print(msg)
    exit(-1)

def validate_arguments(args):
    if (args.val_src is None):
        fatal_error('No input paths provided.')
    if (args.weight is None):
        fatal_error('No input model provided.')

def main(**kwargs):

    for key, value in list(kwargs.items()):
        print(key, value)

    #Override default
    FLAGS.model = kwargs["model"]
    FLAGS.data_dir = kwargs["data_dir"]
    FLAGS.weights = kwargs["weights"]
    eval()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get a program and run it with input')
    parser.add_argument('--model', type=str, default='', help='Which model to use')
    parser.add_argument('--data_dir', type=str, default='', help='name of data dir file')
    parser.add_argument('--weights', type=str, default='', help='name of model file')
    args = parser.parse_args()
    main(**vars(args))
