import config
from preprocess_data import load_train_csv_bb, load_train_dataset, load_valid_csv_bb, load_val_dataset
from preprocess_data import read_image_as_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np


def load_img_train():
    img = []

    filenames_dataset = load_train_dataset(config.train_dir)
    for file in filenames_dataset:
        image = read_image_as_array(config.train_dir + file)
        img.append(image)

    return img

def load_img_validation():
    img = []

    filenames_dataset = load_val_dataset(config.val_dir)
    for file in filenames_dataset:
        image = read_image_as_array(config.val_dir + file)
        img.append(image)

    return img

def train():
    print("[INFO] loading dataset...")

    # load filenames and bounding boxes
    filenames_t, bounding_box_t = load_train_csv_bb(config.train_csv)
    filenames_v, bounding_box_v = load_valid_csv_bb(config.val_csv)

    # load images
    images_train = load_img_train()
    images_validation = load_img_validation()

    print("[INFO] saving testing filenames...")
    f = open(config.test_image, 'w')
    f.write('\n'.join(filenames_v))
    f.close()

    # load the VGG16 network
    vgg = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 244, 3)))

    vgg.trainable = False

    flatten = vgg.output
    flatten = Flatten()(flatten)

    bboxHead = Dense(128, activation='relu')(flatten)
    bboxHead = Dense(64, activation='relu')(bboxHead)
    bboxHead = Dense(32, activation='relu')(bboxHead)
    bboxHead = Dense(4, activation='sigmoid')(bboxHead)

    model = Model(inputs=vgg.input, outputs=bboxHead)

    # initialize the optimizer, compile the model and show the model
    opt = Adam(lr=1e-4)
    model.compile(loss='mse', optimizer=opt)
    print(model.summary())

    # train the network for bounding box regression
    print("[INFO] training bounding box regressor...")
    history = model.fit(images_train, bounding_box_t, validation_data=(images_validation, bounding_box_v),
                  batch_size=32, epochs=25, verbose=1)

    print("[INFO] saving object detector model...")
    model.save(config.model_detector, save_format='h5')

    return history

if __name__ == "__main__":

    H = train()

    N = 25
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0,N), H.history['loss'], label='train_loss')
    plt.plot(np.arange(0,N), H.history['val_loss'], label='val_loss')
    plt.title('Bounding Box Regression Loss on Training Set')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.legend(loc='lower left')
    plt.savefig(config.plot_path)


