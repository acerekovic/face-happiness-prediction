import random
import math
import os
import random
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np

#Group images sequentially by the given labels
def group_images_by_labels(imagefiles,labels):

    num_labels = len(set(labels))
    image_files_list = [[] for _ in range(num_labels)]


    for image_idx, image_file in enumerate(imagefiles):
        image_files_list[labels[image_idx]].append(image_file)

    return image_files_list

def balance_dataset(imagefiles, labels, SEED=10100, num_samples=500):

    image_files_list = group_images_by_labels(imagefiles, labels)
    classes_len = [len(images) for images in image_files_list]
    # gets number of images in category/label with least number of images
    min_num = min(map(len, image_files_list))
    # max_num = max(map(len, image_files_list))
    # mean_num = sum(classes_len) / float(len(classes_len))
    # total_samples_per_class = round(mean_num, -3) # number of samples for training, now we will oversample the minor classes
    # total_samples_per_class = num_samples
    total_samples_per_class = min_num

    random.seed(SEED)
    r = random.random()

    train_list = []
    for _ in range(0, len(classes_len)):
        train_list.append([])

    rest_of_data = []
    rest_of_labels = []

    for list_idx, list_mem in enumerate(image_files_list):
        index_shuf = list(range(len(list_mem)))
        random.shuffle(index_shuf, lambda: r)
        for idx, i in enumerate(index_shuf):
            if idx < total_samples_per_class:
                train_list[list_idx].append(image_files_list[list_idx][i])
            else:
                rest_of_data.append(image_files_list[list_idx][i])
                rest_of_labels.append(list_idx)

    # now update oversampled classes from train_data
    for idx, imagelist in enumerate(train_list):
        while len(imagelist) < total_samples_per_class:
            samples = image_files_list[idx]
            index_shuf = list(range(len(samples)))
            random.shuffle(index_shuf)
            for i in index_shuf:
                if len(imagelist) < total_samples_per_class:
                    train_list[idx].append(samples[i])

    train_data = []
    train_labels = []

    # rewrite into proper structure
    for list_idx, list_mem in enumerate(train_list):
        for sample in list_mem:
            train_data.append(sample)
            train_labels.append(list_idx)



    return train_data, train_labels, rest_of_data, rest_of_labels


def to_rgb1a(im):
    # This should be fsater than 1, as we only
    # truncate to uint8 once (?)
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.float32)
    ret[:, :, 2] =  ret[:, :, 1] =  ret[:, :, 0] =  im
    return ret


def load_labels(csv_file):
    with open(csv_file, "rt", encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter=',')
        next(reader, None)  # skip the headers
        imagefiles = []
        labels = []
        for row in reader:
            imagefiles.append(row[0])
            labels.append(int(row[1]))

    return imagefiles, labels


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(6)
    target_names = ['neutral', 'small_smile', 'large_smile','small_laugh','large_laugh','thrilled']
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')