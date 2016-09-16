from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import utils.misc as pd
import utils.googlenet_load as googlenet_load
import numpy as np
import tensorflow as tf
import cv2
from itertools import chain

class GoogLeNet_FC():

    def __init__(self, image_size, num_channels,  num_classes, batch_size=1, seed=66478):

        #Store important features for the graph
        self.NUM_CLASSES = num_classes
        self.SEED = seed
        self.IMAGE_SIZE = image_size
        self.NUM_CHANNELS = num_channels

        # This is where training samples and labels are fed to the graph.
        # These placeholder nodes will be fed a batch of training data at each
        # training step using the {feed_dict} argument to the Run() call below.
        with tf.name_scope('input'):
            self.x = tf.placeholder(
                tf.float32,
                shape=(batch_size, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_CHANNELS))
            self.y = tf.placeholder(tf.int64, shape=(batch_size,))
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")


        with tf.name_scope('input_reshape'):
            image_shaped_input = tf.reshape(self.x, [-1, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_CHANNELS])
            tf.image_summary('input', image_shaped_input, 6)

        def variable_summaries(var, name):
            """Attach a lot of summaries to a Tensor."""
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.scalar_summary('mean/' + name, mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
                tf.scalar_summary('sttdev/' + name, stddev)
                tf.scalar_summary('max/' + name, tf.reduce_max(var))
                tf.scalar_summary('min/' + name, tf.reduce_min(var))
                tf.histogram_summary(name, var)


        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options)

        # Init stored model
        google_net = googlenet_load.init(config)

        with tf.variable_scope('hidden1') and tf.device(
                '/cpu:0'):  # initialize this part on cpu because it's super large
            input_size = 65536

            fc1_weights = tf.get_variable("fc1_w",  # fully connected 1
                                          shape=[input_size, 2024],
                                          initializer=tf.contrib.layers.xavier_initializer(seed=self.SEED))
            variable_summaries(fc1_weights,"fc1_w")

            fc1_biases = tf.Variable(tf.constant(0., shape=[2024]))

        with tf.variable_scope('hidden2'):
            fc2_weights = tf.get_variable("fc2_w",  # fully connected 2
                                          shape=[2024, 1024],
                                          initializer=tf.contrib.layers.xavier_initializer(seed=self.SEED))
            variable_summaries(fc1_weights, "fc2_w")
            fc2_biases = tf.Variable(tf.constant(0., shape=[1024]))

        with tf.variable_scope('hidden3'):
            fc3_weights = tf.get_variable("fc3_w",  # fully connected 3
                                          shape=[1024, 512],
                                          initializer=tf.contrib.layers.xavier_initializer(seed=self.SEED))
            variable_summaries(fc1_weights, "fc3_w")
            fc3_biases = tf.Variable(tf.constant(0., shape=[512]))

        with tf.variable_scope('final_layer'):
            fc4_weights = tf.get_variable("fc4_w",  # fully connected 4
                                          shape=[512, self.NUM_CLASSES],
                                          initializer=tf.contrib.layers.xavier_initializer(seed=self.SEED))
            variable_summaries(fc1_weights, "fc4_w")
            fc4_biases = tf.Variable(tf.constant(0., shape=[self.NUM_CLASSES]))

        def cnn_model(data):
            Z = googlenet_load.model(data, google_net)
            # Reshape the feature map cuboid into a 2D matrix to feed it to the
            # fully connected layers.
            pool_shape = Z.get_shape().as_list()
            reshape = tf.reshape(
                Z,
                [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            preactivate1 = tf.matmul(reshape, fc1_weights) + fc1_biases
            hidden1 = tf.nn.relu(preactivate1)
            tf.histogram_summary('hidden1/pre_activations', preactivate1)
            tf.histogram_summary('hidden1/activations', hidden1)
            # Add a 50% dropout during training only. Dropout also scales
            # activations such that no rescaling is needed at evaluation time.
            hidden1 = tf.nn.dropout(hidden1, self.dropout_keep_prob)

            preactivate2 = tf.matmul(hidden1, fc2_weights) + fc2_biases
            hidden2 = tf.nn.relu(preactivate2)
            tf.histogram_summary('hidden2/pre_activations', preactivate2)
            tf.histogram_summary('hidden2/activations', hidden2)
            hidden2 = tf.nn.dropout(hidden2, self.dropout_keep_prob)

            preactivate3 = tf.matmul(hidden2, fc3_weights) + fc3_biases
            hidden3 = tf.nn.relu(preactivate3)
            tf.histogram_summary('hidden3/pre_activations', preactivate3)
            tf.histogram_summary('hidden3/activations', hidden3)

            hidden3 = tf.nn.dropout(hidden3, self.dropout_keep_prob)
            output = tf.matmul(hidden3, fc4_weights) + fc4_biases

            return output

        # Training computation: logits + cross-entropy loss.
        logits = cnn_model(self.x)

        with tf.name_scope('loss'):
            with tf.name_scope('total'):
                self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.y))
                # L2 regularization for the fully connected parameters.
                regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                                tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases) +
                                tf.nn.l2_loss(fc3_weights) + tf.nn.l2_loss(fc3_biases) +
                                tf.nn.l2_loss(fc4_weights) + tf.nn.l2_loss(fc4_biases))
                # Add the regularization term to the loss.
                self.loss += 1e-3 * regularizers
            tf.scalar_summary('loss', self.loss)

        with tf.name_scope('train'):
            self.learning_rate = tf.Variable(1e-5)
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                # Predictions (probabilities)
                self.predictions = tf.nn.softmax(logits)
                # Compute number of correct predictions
                correct_prediction = tf.equal(tf.argmax(self.predictions, 1), self.y)
            with tf.name_scope('accuracy'):
                self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary('accuracy',self.acc)


    def infer(self, sess, input_imagedata):
        PIXEL_DEPTH = 255.
        # resize the image, use opencv
        gray = cv2.cvtColor(input_imagedata, cv2.COLOR_BGR2GRAY)
        image_data_rs = cv2.resize(gray, (self.IMAGE_SIZE, self.IMAGE_SIZE))
        # standardize the pixels
        # image_data = image_data_rs
        image_data = (image_data_rs -
                      PIXEL_DEPTH / 2.0) / PIXEL_DEPTH
        image_data = pd.to_rgb1a(image_data)
        image_reshaped = image_data.reshape((-1, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_CHANNELS)).astype(np.float32)
        feed = {self.x: image_reshaped, self.dropout_keep_prob: 1}

        pred = sess.run([self.predictions], feed_dict=feed)
        pred = list(chain.from_iterable(pred))
        return pred