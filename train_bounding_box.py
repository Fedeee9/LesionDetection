import os
import config
import numpy as np
import matplotlib.pyplot as plt
from preprocess_data_bb import data_generator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, CSVLogger, ReduceLROnPlateau


def train():
    # callbacks
    save_model_callback = ModelCheckpoint(config.model_detector, monitor='val_accuracy', save_best_only=True,
                                          mode='auto', verbose=1, period=1)
    early_stopping_callback = EarlyStopping(monitor='val_accuracy', mode='max', restore_best_weights=True,
                                            verbose=1, patience=15)
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_accuracy', mode='auto', factor=0.5, patience=5,
                                           min_lr=0.00001, verbose=1)

    print("[INFO] loading dataset...")
    print("Batch size =", config.batch_size)
    train_data_gen = data_generator(config.train_dir_bb, config.train_csv, config.batch_size)
    val_data_gen = data_generator(config.val_dir_bb, config.val_csv, config.batch_size)

    files_train = [f for f in os.listdir(config.train_dir_bb) if os.path.isfile(os.path.join(config.train_dir_bb, f))]
    train_steps = (len(files_train) // config.batch_size) + 1

    files_val = [f for f in os.listdir(config.val_dir_bb) if os.path.isfile(os.path.join(config.val_dir_bb, f))]
    val_steps = (len(files_val) // config.batch_size) + 1

    print("Train steps =", train_steps)
    print("Validation steps =", val_steps)

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
    opt = Adam(lr=0.0001)
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    print(model.summary())

    # train the network for bounding box regression
    print("[INFO] training bounding box regress...")
    history = model.fit(x=train_data_gen, validation_data=val_data_gen, epochs=config.total_epochs,
                        steps_per_epoch=train_steps, validation_steps=val_steps, verbose=1,
                        callbacks=[save_model_callback, early_stopping_callback, reduce_lr_callback])

    return history


if __name__ == "__main__":
    H = train()

    N = len(H.history['loss'])

    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
    plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss', color='blue')
    plt.title('Bounding Box Regression Loss on Training Set')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.legend(loc='lower left')
    plt.savefig(config.plot_path_loss)

    plt.figure()
    plt.plot(np.arange(0, N), H.history['accuracy'], label='accuracy')
    plt.plot(np.arange(0, N), H.history['val_accuracy'], label='val_accuracy', color='blue')
    plt.title('Bounding Box Accuracy on Training Set')
    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower left')
    plt.savefig(config.plot_path_acc)
