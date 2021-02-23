import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import pandas as pd
import numpy as np
import cv2
import utility
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument("--train_data")
parser.add_argument("--test_data")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--train_sample_ratio", type=float, default=10)
parser.add_argument("--test_sample_ratio", type=float, default=10)
parser.add_argument("-a", "--architecture", default="vgg",
                    choices=['efficientnet', 'vgg', 'mobilenet', 'mobilenetv2', 'vggm'])
parser.add_argument("-o", "--out", default="./")
parser.add_argument("--load_model")
parser.add_argument("--lr")
args = parser.parse_args()

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
TIMESTAMP = utility.get_datastamp()
MODEL_DIR = os.path.join(args.out, f"{args.architecture}-{TIMESTAMP}")

print("[INFO] Processing training data..")
TRAIN_DATA_PATH = args.train_data
TRAIN_FILES = os.listdir(TRAIN_DATA_PATH)
for filename in TRAIN_FILES:  # Removes file without .png extension
    if not filename.endswith('png'):
        TRAIN_FILES.remove(filename)
np.random.shuffle(TRAIN_FILES)
NUM_OBJECTS_TRAIN = len(TRAIN_FILES)
TRAIN_FILTER = args.train_sample_ratio

print("[INFO] Processing validation data..")
TEST_DATA_PATH = args.test_data
TEST_FILES = os.listdir(TEST_DATA_PATH)
for filename in TEST_FILES:
    if not filename.endswith('png'):
        TEST_FILES.remove(filename)
np.random.shuffle(TEST_FILES)
NUM_OBJECTS_TEST = len(TEST_FILES)
TEST_FILTER = args.test_sample_ratio

os.mkdir(MODEL_DIR)

METRICS = [
    keras.metrics.CategoricalAccuracy(name='accuracy'),
    # keras.metrics.BinaryAccuracy(name='binary_accuracy'),
    # keras.metrics.Precision(name='precision'),
    # keras.metrics.Recall(name='recall'),
    # keras.metrics.AUC(name='auc')
]


# def scheduler(epoch, lr):
#     if epoch <= 20:
#         return 1e-3
#     elif 20 < epoch <= 50:
#         return 1e-4
#     else:
#         return 1e-5


CALLBACKS = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, f'classification_model.h5'),
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_freq='epoch'),
    tf.keras.callbacks.TensorBoard(log_dir=os.path.join(MODEL_DIR, 'logs')),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.5,
                                         patience=3,
                                         verbose=1,
                                         mode='min',
                                         min_lr=1e-8),
    tf.keras.callbacks.EarlyStopping(patience=5),
    # tf.keras.callbacks.LearningRateScheduler(scheduler)
]


def data_loader_train():
    labels_dict = utility.get_label_dict()
    for i in range(NUM_OBJECTS_TRAIN):
        if i % TRAIN_FILTER == 0:
            idx = np.random.randint(0, NUM_OBJECTS_TRAIN)
            file_path = os.path.join(TRAIN_DATA_PATH, TRAIN_FILES[idx])
            x = cv2.imread(file_path)
            x = x / 255.0
            # x = x[:, :, 0]
            label_class = TRAIN_FILES[idx].split("_")[0]
            if label_class == 'night':
                label_class = 'night_stand'  # Quick fix for label parsing
            label_class = utility.int_to_1hot(labels_dict[label_class], 10)
            label_view = utility.int_to_1hot(int(TRAIN_FILES[idx].split("_")[-1].split(".")[0]), 60)
            yield np.resize(x, (224, 224, 3)), (label_class, label_view)


def data_loader_test():
    labels_dict = utility.get_label_dict()
    for i in range(NUM_OBJECTS_TEST):
        if i % TEST_FILTER == 0:
            # idx = np.random.randint(0, NUM_OBJECTS_TEST)  # Remove randomization in sampling to stabilize validation
            file_path = os.path.join(TEST_DATA_PATH, TEST_FILES[i])
            # x = keras.preprocessing.image.load_img(file_path,
            #                                        color_mode='grayscale',
            #                                        target_size=(224, 224),
            #                                        interpolation='nearest')
            # x = keras.preprocessing.image.img_to_array(x)
            x = cv2.imread(file_path)
            x = x / 255.0
            # x = x[:, :, 0]
            label_class = TEST_FILES[i].split("_")[0]
            if label_class == 'night':
                label_class = 'night_stand'  # Quick fix for label parsing
            label_class = utility.int_to_1hot(labels_dict[label_class], 10)
            label_view = utility.int_to_1hot(int(TEST_FILES[i].split("_")[-1].split(".")[0]), 60)
            yield np.resize(x, (224, 224, 3)), (label_class, label_view)


def dataset_generator_train():
    dataset = tf.data.Dataset.from_generator(data_loader_train,
                                             output_types=(tf.float32, (tf.int16, tf.int16)),
                                             output_shapes=(tf.TensorShape([224, 224, 3]),
                                                            (tf.TensorShape([10]), tf.TensorShape([60]))))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat(EPOCHS)
    return dataset


def dataset_generator_test():
    dataset = tf.data.Dataset.from_generator(data_loader_test,
                                             output_types=(tf.float32, (tf.int16, tf.int16)),
                                             output_shapes=(tf.TensorShape([224, 224, 3]),
                                                            (tf.TensorShape([10]), tf.TensorShape([60]))))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat(EPOCHS)
    return dataset


def generate_cnn(app="vgg"):
    inputs = keras.Input(shape=(224, 224, 3))

    if app == "vgg":
        net = keras.applications.VGG16(include_top=False,
                                       weights='imagenet',
                                       input_tensor=inputs)
        net.trainable = False
        # preprocessed = keras.applications.vgg16.preprocess_input(inputs)
        x = net(inputs)

    elif app == "efficientnet":
        net = keras.applications.EfficientNetB0(include_top=False,
                                                weights='imagenet')
        # net.trainable = False
        # preprocessed = keras.applications.efficientnet.preprocess_input(inputs)
        x = net(inputs)

    elif app == "mobilenet":
        net = keras.applications.MobileNet(include_top=False,
                                           weights='imagenet',
                                           )
        net.trainable = False
        # preprocessed = keras.applications.mobilenet.preprocess_input(inputs)
        x = net(inputs)

    elif app == "mobilenetv2":
        net = keras.applications.MobileNetV2(include_top=False,
                                             weights='imagenet',
                                             )
        net.trainable = False
        # preprocessed = keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = net(inputs)

    elif app == "vggm":
        x = keras.layers.Conv2D(96, kernel_size=7, strides=2, padding='same', kernel_regularizer='l2')(inputs)
        x = layers.LeakyReLU()(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPool2D(pool_size=3, strides=2)(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Conv2D(256, kernel_size=5, strides=2, padding='same', kernel_regularizer='l2')(x)
        x = layers.LeakyReLU()(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPool2D(pool_size=3, strides=2)(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Conv2D(512, kernel_size=3, strides=1, padding='same', kernel_regularizer='l2')(x)
        x = layers.LeakyReLU()(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Conv2D(512, kernel_size=3, strides=1, padding='same', kernel_regularizer='l2')(x)
        x = layers.LeakyReLU()(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Conv2D(512, kernel_size=3, strides=1, padding='same', kernel_regularizer='l2')(x)
        x = layers.LeakyReLU()(x)
        x = keras.layers.MaxPool2D(pool_size=3, strides=2)(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(4096)(x)
        x = layers.LeakyReLU()(x)
        x = keras.layers.Dropout(0.5)(x)

    x = layers.Flatten()(x)

    # x_class = keras.layers.Dense(4096)(x)
    # x_class = layers.ReLU()(x_class)
    # x_class = keras.layers.Dropout(0.5)(x_class)
    # x_class = keras.layers.Dense(220)(x_class)
    # x_class = layers.ReLU()(x_class)
    # x_class = layers.Dropout(0.5)(x_class)
    #
    # x_view = keras.layers.Dense(4096)(x)
    # x_view = layers.ReLU()(x_view)
    # x_view = keras.layers.Dropout(0.5)(x_view)
    # x_view = keras.layers.Dense(220)(x_view)
    # x_view = layers.ReLU()(x_view)
    # x_view = keras.layers.Dropout(0.5)(x_view)

    out_class = layers.Dense(10, activation='softmax', name="class")(x)
    out_view = layers.Dense(60, activation='softmax', name="view")(x)
    model = keras.Model(inputs=inputs, outputs=[out_class, out_view])
    model.summary()
    losses = {"class": 'categorical_crossentropy',
              "view": 'categorical_crossentropy'}
    model.compile(keras.optimizers.Adam(learning_rate=float(args.lr)), loss=losses, metrics=METRICS)
    # keras.utils.plot_model(model, "net_structure.png", show_shapes=True, expand_nested=True)
    return model


def main():
    mirrored_strategy = tf.distribute.MirroredStrategy()
    model = generate_cnn(app=args.architecture)
    num_batches = int((NUM_OBJECTS_TRAIN / TRAIN_FILTER) / BATCH_SIZE)
    train_data_gen = dataset_generator_train()
    test_data = dataset_generator_test()
    if args.load_model is not None:
        model.load_weights(args.load_model)
        print(f"[INFO] Model {args.load_model} correctly loaded.")
    history = model.fit(train_data_gen,
                        shuffle=True,
                        steps_per_epoch=num_batches,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=CALLBACKS,
                        validation_data=test_data)
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(MODEL_DIR, f"{TIMESTAMP}_training_history.csv"))


if __name__ == '__main__':
    main()
