import re
import numpy as np
import pandas as pd
import json
import os
import random
import config
import cv2
from collections import defaultdict
from tensorflow.python.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical

# Mod Bounding Boxes
def load_train_csv_bb(train_csv):
    df = pd.read_csv(train_csv, sep=';')

    filenames = df['File_name'].dropna().tolist()
    box = df['Bounding_boxes'].dropna().tolist()
    #axis = df['Measurement_coordinates'].tolist()

    return filenames, box

# Mod Labels
def load_train_csv(train_csv):
    df = pd.read_csv(train_csv, sep=';')

    return df

def load_test_csv(test_csv):
    df = pd.read_csv(test_csv, sep=';')

    return df

def load_val_csv(val_csv):
    df = pd.read_csv(val_csv, sep=';')

    return df

def load_labels(labels_file):
    labels = []

    data = json.load(open(labels_file))
    label = data.get('term_list')

    for i in label:
        labels.append(i.strip())

    return labels


def read_image_as_array(image_path, target_shape=config.input_shape):

    height = config.input_shape[0]
    width = config.input_shape[1]

    # load with Pillow
    img = image.load_img(image_path, target_size=target_shape)
    #img.show()
    image_array = image.img_to_array(img)

    # load with opencv
#    img_o = cv2.imread(image_path)
#    img_o = cv2.resize(img_o, (width, height))
#    image_array = cv2.cvtColor(img_o,cv2.COLOR_BGR2RGB)
#    cv2.imshow('image', img_o)
#    k = cv2.waitKey(0)

    return image_array * (1. / 255)


def load_dataset_from_dir(d):
    images = []

    with os.scandir(d) as files:
       for f in files:
           if f.is_file():
               images.append(f.name)

    #random.shuffle(images)

    return images


def load_train_dataset(train_dir):
    return load_dataset_from_dir(train_dir)

def load_test_dataset(test_dir):
    return load_dataset_from_dir(test_dir)

def load_val_dataset(val_dir):
    return load_dataset_from_dir(val_dir)



# def split_train_test_val(dataset):
#     for c in dataset.keys():
#         random.shuffle(dataset[c])
#     train_images = []
#     test_images = []
#     val_images = []
#     for c in dataset.keys():
#         train_images = train_images + dataset[c][int(len(dataset[c]) * 0.0) : int(len(dataset[c]) * 0.7)]
#         test_images = test_images + dataset[c][int(len(dataset[c]) * 0.7) : int(len(dataset[c]) * 0.85)]
#         val_images = val_images + dataset[c][int(len(dataset[c]) * 0.85) : int(len(dataset[c]) * 1.0)]


#     random.shuffle(train_images)
#     random.shuffle(test_images)
#     random.shuffle(val_images)

#     return train_images, test_images, val_images



def decode_label(labels, label):
    labels.sort()
    return labels[np.argmax(label)]


def data_generator(dataset_dir, labels, dataset_list, bath_size):
    x_image, y_class = list(), list()
    labels.sort()
    labels_int_dict = {labels[i]: i for i in range(0, len(labels))}

    n = 0
    while True:

        for image_name in dataset_list:
            n += 1
#            img = image.load_img(dataset_dir + image_name, target_size=config.input_shape)
#            img = image.img_to_array(img)
            image_array = read_image_as_array(dataset_dir + image_name)
            c = re.split(r'[0-9]', image_name)[0]

            encoded_label = to_categorical(labels_int_dict[c], num_classes=len(labels))
            x_image.append(image_array)
            y_class.append(encoded_label)

            if n == bath_size:
                yield (np.array(x_image),  np.array(y_class))
                x_image, y_class = list(), list()
                n = 0
