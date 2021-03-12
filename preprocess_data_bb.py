import os
import cv2
import config
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def data_generator(dir_bb, csv, batch_size):
    df = pd.read_csv(csv, sep=';').dropna()

    file_name = []
    bounding_box = []

    for i, line in df.iterrows():
        name = line['File_name']
        bb = line['Bounding_boxes']

        if os.path.isfile(os.path.join(dir_bb, name)):
            file_name.append(name)
            bounding_box.append(bb)

    images = []
    boxes = []
    count = 0

    n = 0
    while True:
        for name, bb in zip(file_name, bounding_box):
            bb = bb.split(',')
            (startX, startY, endX, endY) = bb

            imagePath = os.path.sep.join([dir_bb, name])
            image = cv2.imread(imagePath)

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

            if n == batch_size:
                yield np.array(images), np.array(boxes)
                images, boxes = list(), list()
                n = 0


if __name__ == "__main__":
    data_generator(config.train_dir_bb, config.train_csv, config.batch_size)
