"""
Trains a model for entropy estimation of 3D objects.

Parses prevoxelization.py output as x-data and generate_entropy_dataset.py
as y-data to train a model that estimates 60 entropy values from (56,56,56)
occupancy voxel grids.
"""
import os
import sys
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import utility
import kerastuner as kt
from kerastuner.tuners import Hyperband
from tqdm import tqdm

print(f"Tensorflow v{tf.__version__}\n")

parser = argparse.ArgumentParser()
parser.add_argument('--voxel_data', required=True)
parser.add_argument('--entropy_dataset', required=True)
parser.add_argument('-b', '--batch_size', type=int, default=8)
parser.add_argument('-e', '--epochs', type=int, default=5)
parser.add_argument('--load_model')
parser.add_argument('--out', default="./")
args = parser.parse_args()

TIMESTAMP = datetime.now().strftime('%d-%m-%H%M')
BASE_DIR = sys.path[0]
DATA_DIR = os.path.join(BASE_DIR, args.voxel_data)
MODEL_DIR = os.path.join(BASE_DIR, args.out, f"entropy-model")

METRICS = [
    keras.metrics.AUC(name='auc'),
    keras.metrics.MeanSquaredError(name='mse')
]

CALLBACKS = [
    # tf.keras.callbacks.EarlyStopping(patience=3),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'entropy_model.h5'),
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_freq='epoch'),
    tf.keras.callbacks.TensorBoard(log_dir=os.path.join(MODEL_DIR, 'logs/')),
    tf.keras.callbacks.CSVLogger(os.path.join(MODEL_DIR, 'logs/training_log.csv')),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.3,
                                         patience=10,
                                         verbose=1,
                                         mode='min',
                                         min_lr=3e-7),
]
CLASSES = utility.CLASSES


def load_data(data, csv):
    x_train, y_train, x_test, y_test = [], [], [], []
    csv = pd.read_csv(csv)
    print('[INFO] Loading Data...')
    for lab in CLASSES:
        # print(f"[DEBUG] Loading {lab}\n")
        for file in tqdm(os.listdir(os.path.join(data, lab, 'train'))):
            if '.npy' in file:
                _data = np.load(os.path.join(data, lab, 'train', file))
                padded_data = np.pad(_data, 3, 'constant')
                x_train.append(padded_data)
                filename = file.split(".")[0]
                index = int(filename.split("_")[-1])
                # print(f"[DEBUG] label, index : {lab}, {index}")
                subcsv = csv[csv['label'] == lab]
                entropies = np.array(subcsv[subcsv['object_index'] == index].sort_values(by=['view_code']).entropy)
                # print(f"[DEBUG] Entropies of {file} : {entropies}")
                y_train.append(entropies)

        for file in tqdm(os.listdir(os.path.join(data, lab, 'test'))):
            if '.npy' in file:
                _data = np.load(os.path.join(data, lab, 'test', file))
                padded_data = np.pad(_data, 3, 'constant')
                x_test.append(padded_data)
                filename = file.split(".")[0]
                index = int(filename.split("_")[-1])
                # print(f"[DEBUG] label, index : {lab}, {index}")
                subcsv = csv[csv['label'] == lab]
                entropies = np.array(subcsv[subcsv['object_index'] == index].sort_values(by=['view_code']).entropy)
                # print(f"[DEBUG] Entropies of {file} : {entropies}")
                y_test.append(entropies)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    xy_train = list(zip(x_train, y_train))
    np.random.shuffle(xy_train)
    x_train, y_train = zip(*xy_train)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    return x_train, y_train, x_test, y_test


def generate_cnn():
    """
    Function to generate a Convolutional Neural Network
    to estimate entropy from (56,56,56) voxel occupancy grids.

    :return: Keras.Model
    """
    inputs = keras.Input(shape=(56, 56, 56))
    base = layers.Reshape(target_shape=(56, 56, 56, 1))(inputs)

    # cnn_a_filters = hp.Int('cnn1_filters', min_value=4, max_value=16, step=4)
    a = layers.Conv3D(8, (5, 5, 5), activation='relu', padding='same')(base)
    a = layers.AveragePooling3D(pool_size=(2, 2, 2))(a)
    a = layers.BatchNormalization()(a)
    a = layers.Dropout(0.25)(a)
    a = layers.Flatten()(a)

    # cnn_b_filters = hp.Int('cnn2_filters', min_value=4, max_value=16, step=4)
    b = layers.Conv3D(8, (3, 3, 3), activation='relu', padding='same')(base)
    b = layers.AveragePooling3D(pool_size=(2, 2, 2))(b)
    b = layers.BatchNormalization()(b)
    b = layers.Dropout(0.25)(b)
    b = layers.Flatten()(b)

    x = layers.Concatenate(axis=1)([a, b])
    # dense_units = hp.Int('dense_units', min_value=256, max_value=512, step=64)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(60, activation='linear')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='entronet')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5), loss='mae', metrics=['mse'])
    model.summary()
    return model


def main():
    os.mkdir(MODEL_DIR)
    x_train, y_train, x_test, y_test = load_data(args.voxel_data, args.entropy_dataset)

    ## Uncomment following to perform hyperparameters training.
    # tuner = Hyperband(generate_cnn,
    #                   objective=kt.Objective("val_loss", direction="min"),
    #                   max_epochs=20,
    #                   factor=3,
    #                   directory='../../../../data/s3866033/fyp',  # Only admits relative path, for some reason.
    #                   project_name=f'hyperband_optimization{TIMESTAMP}')
    # tuner.search(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    # best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    #
    # model = tuner.hypermodel.build(best_hps)

    model = generate_cnn()
    if args.load_model is not None:
        model.load_weights(args.load_model)
        print(f"[INFO] Model {args.load_model} correctly loaded.")
    history = model.fit(x_train, y_train,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        validation_data=(x_test, y_test),
                        callbacks=CALLBACKS,
                        shuffle=True)


if __name__ == '__main__':
    main()
