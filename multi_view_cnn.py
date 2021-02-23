import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import utility
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--views', nargs='+')
parser.add_argument('--model')
args = parser.parse_args()

idx2label = {}
count = 0
for phi in range(30, 151, 30):
    for theta in range(0, 331, 30):
        idx2label[count] = (theta, phi)
        count += 1




def most_common(lst):
    return max(set(lst), key=lst.count)

def main():
    print("[INFO] Prediction results:")
    images = []
    true_labels = []
    true_views = []
    for file in args.views:
        images.append(cv2.imread(file))
        true_labels.append(os.path.split(file.split("_")[-8])[-1])
        true_views.append(file.split("_")[-1].split(".")[0])
    model = keras.models.load_model(args.model)
    images = np.array(images)
    results = model.predict(images)
    labels = results[0]
    views = results[1]
    dic = utility.get_label_dict(inverse=True)
    for i in range(len(views)):
        print(f"Predicted: {dic[np.argmax(labels[i])]}, {idx2label[int(np.argmax(views[i]))]} - True: {true_labels[i]}, {idx2label[int(true_views[i])]}")

    print(f"[INFO] Majority vote:")
    labint = []
    for el in labels:
        labint.append(np.argmax(el))
    print(f"    class: {dic[most_common(labint)]}")
    view_offset = []
    for i in range(len(views)):
        view_offset.append(int(true_views[i]) - int(np.argmax(views[i])))
    offset = np.array(idx2label[most_common(view_offset)])
    print(f"    offset: theta={offset[0]} phi={offset[1]}")


if __name__ == "__main__":
    main()
