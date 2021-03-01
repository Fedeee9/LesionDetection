import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def data_generator(dir_bb, csv, bath_size):
    df = pd.read_csv(csv, sep=';').dropna()

    images = []
    boxes = []
    count = 0

    filenames = df['File_name'].tolist()
    filenames.sort()
    bb = df['Bounding_boxes'].tolist()
    # axis = df['Measurement_coordinates'].tolist()

    n = 0
    for row in bb:
        name = filenames[count]
        row = row.split(',')
        (startX, startY, endX, endY) = row

        imagePath = os.path.sep.join([dir_bb, name])
        image = cv2.imread(imagePath)

        if image is None:
            #print('NON PRESENTE')
            count += 1
            continue
        (h, w) = image.shape[:2]

        # scale the bounding box coordinates relative to the spatial
        startX = float(startX) / w
        startY = float(startY) / h
        endX = float(endX) / w
        endY = float(endY) / h

        # load the image and preprocess it
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)

        image_x = np.array(image, dtype='float32') / 255.0
        box_y = np.array((startX, startY, endX, endY), dtype='float32')

        images.append(image_x)
        boxes.append(box_y)

        count += 1
        n += 1

        if n == bath_size:
            yield np.array(images), np.array(boxes)
            images, boxes = list(), list()
            n = 0
