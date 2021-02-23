import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence warnings
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import utility

parser = argparse.ArgumentParser()
parser.add_argument("data")
# parser.add_argument("--single_picture", action='store_true')
args = parser.parse_args()

DATA_PATH = args.data

FILES = os.listdir(DATA_PATH)
for filename in FILES:  # Removes file without .png extension
    if not filename.endswith('png'):
        FILES.remove(filename)
NUM_OBJECTS = len(FILES)


def data_loader():
    labels_dict = utility.get_label_dict()
    for i in range(NUM_OBJECTS):
        file_path = os.path.join(DATA_PATH, FILES[i])
        x = keras.preprocessing.image.load_img(file_path,
                                               color_mode='rgb',
                                               target_size=(240, 320),
                                               interpolation='nearest')
        x = keras.preprocessing.image.img_to_array(x)
        label_class = FILES[i].split("_")[0]
        if label_class == 'night':
            label_class = 'night_stand'  # Quick fix for label parsing
        label_class = utility.int_to_1hot(labels_dict[label_class], 10)
        label_view = utility.int_to_1hot(int(FILES[i].split("_")[-1].split(".")[0]), 60)
        yield np.reshape(x, newshape=(1,240,320,3)), (np.reshape(label_class, newshape=(1,10)), np.reshape(label_view, newshape=(1,60)))


def dataset_generator():
    dataset = tf.data.Dataset.from_generator(data_loader,
                                             output_types=(tf.float32, (tf.int16, tf.int16)),
                                             output_shapes=(tf.TensorShape([1, 240, 320, 3]),
                                                            (tf.TensorShape([1,10]), tf.TensorShape([1,60]))))
    return dataset


def generate_cnn():
    inputs = keras.Input(shape=(240, 320, 3))
    vgg = keras.applications.VGG16(include_top=False,
                                   weights='imagenet',
                                   input_tensor=inputs,
                                   input_shape=(240, 320, 3))
    preprocessed = keras.applications.vgg16.preprocess_input(inputs)
    x = vgg(preprocessed)
    x = layers.Flatten()(x)
    out_class = layers.Dense(10, activation='softmax', name="class")(x)
    out_view = layers.Dense(60, activation='softmax', name="view")(x)
    model = keras.Model(inputs=inputs, outputs=[out_class, out_view])
    model.summary()
    losses = {"class": "categorical_crossentropy",
              "view": "categorical_crossentropy"}
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss=losses, metrics='accuracy')
    # keras.utils.plot_model(model, "net_structure.png", show_shapes=True, expand_nested=True)
    return model


model = generate_cnn()
model.load_weights('single_view_3_epochs.h5')
test_dataset = dataset_generator()
results = model.evaluate(test_dataset)

# file_path = args.picture
# x = keras.preprocessing.image.load_img(file_path,
#                                        color_mode='rgb',
#                                        target_size=(240, 320),
#                                        interpolation='nearest')
# x = keras.preprocessing.image.img_to_array(x)
# x = x.reshape((1, 240, 320, 3))
#
# print("[INFO] Computing prediction...")
# int2lab = utility.get_label_dict(inverse=True)
# y1pred, y2pred = model.predict(x)
# y1out, y2out = np.argmax(y1pred), np.argmax(y2pred)
# print(int2lab[y1out])
# print(y2out)
