import math
import os
import random

import cv2

import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
from scipy.misc import imread, imresize
import utils.misc as misc
import glob


PIXEL_DEPTH = 255.


class Loader():

    def __init__(self, data_dir,image_size, num_channels, batch_size,mode,aug_size=0,seed=66478):

        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.aug_size = aug_size
        self.seed = seed
        self.mode = mode
        self.data_dir = data_dir

        # Parameters for training augmentation
        self.data_gen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.01,
        height_shift_range=0.01,
        shear_range=0.1,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest')

        print("Loading files...")

        # Load training/validation/testing images from dest_dir folder
        self.load()

        #Create batches
        self.create_batches()



    def load(self):

        #find csv file (either training or validation)
        label_filename = ''
        for name in glob.glob(os.path.join(self.data_dir,'*.csv')):
            if self.mode in name:
                label_filename = name
                break

        if label_filename is '': #if there is no filename print error
            print('Please provide a proper path to data_dir:'+str(self.mode)+', or a prer *.csv file in folder ' +self.data_dir)
            exit(1)

        #loal all filenames and labels
        imagefiles, labels = misc.load_labels(label_filename)

        #balance training dataset
        if self.mode == 'training':
            imagefiles, labels, _, _ = misc.balance_dataset(imagefiles,labels,SEED=self.seed)

        #shuffle randomly
        random.seed(self.seed)
        index_shuf = list(range(len(imagefiles)))
        random.shuffle(index_shuf)
        imagefiles_sh = []
        labels_sh = []

        for i in index_shuf:
            imagefiles_sh.append(imagefiles[i])
            labels_sh.append(labels[i])

        #store list of images and labels
        self.input_imagefiles = imagefiles_sh
        self.input_labels = labels_sh
        self.input_size = len(self.input_imagefiles)


    def create_batches(self):

        # Compute number of batches.
        # Note that this is not perfect - if the batch_size is not a divisor of an input_size, some images will be discarded in this epoch
        self.num_batches = int(self.input_size / (self.batch_size))

        # # When the data (tensor) is too small, let's give them a better error message
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        xdata = np.array(self.input_imagefiles[:self.num_batches * self.batch_size])
        ydata = np.array(self.input_labels[:self.num_batches * self.batch_size],dtype=np.int32)

        self.x_batches = np.split(xdata, self.num_batches)
        self.y_batches = np.split(ydata, self.num_batches)
        self.pointer = 0

        #update number of batches according to aug_size
        self.orig_num_batches = self.num_batches
        self.num_batches = int(self.input_size / (self.batch_size)) * (self.aug_size + 1)
        self.curr_batch = 0

    def next_batch(self):
        #if came to an end of list of existing images, start from beginning
        if (self.pointer == self.orig_num_batches):
            self.pointer = 0

        # If an epoch is finished, reset batch pointer (This also randomizes images accross batches)
        if (self.curr_batch == self.num_batches):
            self.reset_batch_pointer()

        x_imagefiles, y = self.x_batches[self.pointer], self.y_batches[self.pointer].reshape(self.batch_size, )
        #Create an input np array
        x = np.zeros(
            (self.batch_size, self.image_size, self.image_size, self.num_channels), dtype=np.float32)

        #Now read images, convert them to grayscale, normalize it, and convert back to 3-channel image
        for image_idx, imagefile in enumerate(x_imagefiles):
            try:
                #read image from data_dir directory
                image_file = os.path.join(self.data_dir+'/images',imagefile)

                image_data_o = imread(image_file)
                # resize the image, use opencv
                image_data_rs = imresize(image_data_o, (self.image_size, self.image_size), interp='cubic')
                # convert to grayscale
                gray = cv2.cvtColor(image_data_rs, cv2.COLOR_BGR2GRAY)

                #augment image, if current batch goes beyond int(self.input_size / (self.batch_size)
                if self.aug_size != 0 and self.curr_batch > self.orig_num_batches:
                    #print('batch #' + str(self.curr_batch) + '...augmenting image')
                    gray = self.augment_image(gray)
                # normalize image
                aug_img_norm = (gray -
                             PIXEL_DEPTH / 2.0) / PIXEL_DEPTH  # normalize it
                # duplicate grayscale channel 2 times to create 3 channel image
                image_data = misc.to_rgb1a(aug_img_norm)
                #preview augmented image
                # fig, ax = plt.subplots()
                # plt.ion()
                # length, width, ch = np.shape(image_data)
                # ax.imshow(image_data, extent=[-width / 2, width / 2, -length / 2, length / 2])
                # plt.show()

                #check image shape, is valid
                if image_data.shape != (self.image_size, self.image_size, self.num_channels):
                    raise Exception('Unexpected image shape: %s' % str(image_data.shape))
                if np.isnan(image_data).any():
                    raise Exception('Unexpected image value')
                x[image_idx, :, :, :] = image_data

            except IOError as e:
                print('Could not read:', imagefile, ':', e, '- it\'s ok, skipping.')

        self.pointer += 1
        self.curr_batch += 1
        return x, y

    def reset_batch_pointer(self):
        #reshuffle images and create new batches
        self.load()
        #Create batches
        self.create_batches()

    #To augment data, ImageGenerator from Keras is used.
    def augment_image(self, imagedata):

        img_arr = img_to_array(imagedata)  # this is a Numpy array with shape (3, 150, 150)
        img_arr = img_arr.reshape((1,) + img_arr.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

        # the .flow() command below generates batches of randomly transformed images
        itr = self.data_gen.flow(img_arr, batch_size=1)
        image = itr.X[0]
        image = np.array(array_to_img(image))

        return image

