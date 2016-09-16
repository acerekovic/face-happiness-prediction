from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import time
from datetime import datetime
from googlenet_model import GoogLeNet_FC
from vgg_model import VGG16
import argparse
import tensorflow as tf
import utils.data_loader as data_loader
from utils.data_loader import Loader

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir','./data/HAPPEI/', """input dataset""")
#tf.app.flags.DEFINE_string('weights','./train_dir/HAPPEI_googlenet_2016_08_18_19.41/model.ckpt-6501', """Model which restores the weights.""") #Leave empty if training from scratch
tf.app.flags.DEFINE_string('weights','', """Model which restores the weights.""")
tf.app.flags.DEFINE_string('train_dir', './train_dir',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('num_epochs', 5, """Number of bepochs to train.""")
tf.app.flags.DEFINE_string('model','', """TF model""")

# Global constants describing the data set.
NUM_CLASSES = 6
NUM_CHANNELS = 3

# Basic model parameters.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 32
EVAL_FREQUENCY = 200 # Number of steps between evaluations.


def train():


    # Build an initialization operation to run below.
    if FLAGS.model == 'GNET':
        global IMAGE_SIZE
        IMAGE_SIZE = 256  # global variable for GoogLeNet-FC
        model = GoogLeNet_FC(IMAGE_SIZE, NUM_CHANNELS, NUM_CLASSES, BATCH_SIZE, SEED)
    elif FLAGS.model == 'VGG16':
        global IMAGE_SIZE
        IMAGE_SIZE = 224  # global variable for VGG
        model = VGG16(IMAGE_SIZE, NUM_CHANNELS, NUM_CLASSES, BATCH_SIZE, SEED)
    else:
        print('Please provide a proper model')
        exit(1)


    # Generate batches for training and validation
    train_loader = Loader(FLAGS.data_dir,IMAGE_SIZE, NUM_CHANNELS,  BATCH_SIZE,'training',10,SEED)
    val_loader = Loader(FLAGS.data_dir,IMAGE_SIZE, NUM_CHANNELS, BATCH_SIZE,'validation',0,SEED)

    # Start running operations on the Graph.
    sess = tf.InteractiveSession()
    merged = tf.merge_all_summaries()

    # Initialize Saver (we save only trainable variables)
    saver = tf.train.Saver(tf.trainable_variables())
    tf.initialize_all_variables().run()

    #load initial weights for VGG
    if FLAGS.model == 'VGG16':
        model.load_weights('./data/prereq/vgg16_weights.npz',sess)

    # Restore weights from pretrained model, if specified
    if FLAGS.weights:
        print('Restoring the weights from the model...' +  FLAGS.weights)
        saver.restore(sess, FLAGS.weights)

    print('Model initialized!')
    # Generate save dir based on current date and time
    save_dir = FLAGS.train_dir + '/' + FLAGS.model + '_' + datetime.now().strftime('%Y_%m_%d_%H.%M')
    train_writer = tf.train.SummaryWriter(save_dir, sess.graph)

    # Loop through training batches for a number of epochs
    for e in range(FLAGS.num_epochs):

        # save the epoch start time
        e_start_time = time.time()
        acc_train_acc = 0

        # Loop through training steps.
        for b in range(train_loader.num_batches):

            # compute number of steps so far
            s = e*train_loader.num_batches +b
            # Compute the offset of the current minibatch in the data.

            batch_x, batch_y = train_loader.next_batch()

            start_time = time.time()

            if b % EVAL_FREQUENCY == 0:
                print('Training with batch %d out of %d, epoch %d' %
                      (b+1, train_loader.num_batches, e+1))
            # Define training parameters
            feed_dict = {model.x: batch_x,
                         model.y: batch_y,
                         model.dropout_keep_prob: 0.5}

            # Run the graph and fetch some of the nodes.
            _, l, lr, pred, b_acc, sum = sess.run(
                [model.optimizer, model.loss, model.learning_rate, model.predictions, model.acc, merged],
                feed_dict=feed_dict)

            # Accumulate accuracy
            acc_train_acc += b_acc

            # Duration of this batch phase
            duration = time.time() - start_time

            # Evaluate the model
            if s % EVAL_FREQUENCY == 0:

                # Compute total training accuracy
                train_acc = acc_train_acc/(b+1) # normalize with number of batches in this phase

                # Add summary for Tensorboard
                train_writer.add_summary(sum, s)
                start_time = time.time()

                acc_val_acc = 0

                for bv in range(val_loader.num_batches):

                    val_x, val_y = val_loader.next_batch()
                    feed_dict = {model.x: val_x,
                                 model.y: val_y,
                                 model.dropout_keep_prob: 1}
                    # Run the graph and fetch some of the nodes.
                    vpred, val_batch_acc = sess.run(
                        [model.predictions, model.acc],
                        feed_dict=feed_dict)
                    # Accumulate acc for every bacth
                    acc_val_acc += val_batch_acc

                # Compute total validation accuracy
                val_acc = acc_val_acc / val_loader.num_batches

                duration += time.time() - start_time
                print('Epoch %d, step %d,  learning rate: %.6f, minibatch loss: %.6f, minibatch acc: %.6f, epoch train accuracy: %.6f, '
                      'validation accuracy: %.6f' % (e+1, s+1, lr,l, b_acc*100, train_acc*100.0,  val_acc*100))

            # Save model periodically
            if s % (EVAL_FREQUENCY*2) == 0:
                saver.save(sess, save_dir + '/' + 'model.ckpt',
                               global_step=s + 1)
        print('Epoch %d finished in %d steps, %.4f minutes!' %(e+1, s+1, (time.time() - e_start_time)/60))

    #at the end, save the model
    saver.save(sess, save_dir + '/' + 'model.ckpt',
               global_step=s + 1)

def fatal_error(msg):
    print(msg)
    exit(-1)

def validate_arguments(args):
    if (args.val_src is None) or (args.train_src is None):
        fatal_error('No input paths provided.')


def main(**kwargs):

    #for key, value in list(kwargs.items()):
    #    print(key, value)

    # Override default
    FLAGS.model = kwargs["model"]
    FLAGS.data_dir = kwargs["data_dir"]
    if kwargs["weights"] is not '':
        FLAGS.weights = kwargs["weights"]
    if kwargs["num_epochs"] is not '':
        FLAGS.num_epochs = int(kwargs["num_epochs"])

    train()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get a program and run it with input')
    parser.add_argument('--model', type=str, default='', help='Which model to use')
    parser.add_argument('--data_dir', type=str, default='', help='Path to dataset')
    parser.add_argument('--weights', type=str, default='', help='Path to pretrained model')
    parser.add_argument('--num_epochs', type=str, default='', help='How many epochs to train')
    args = parser.parse_args()
    main(**vars(args))
