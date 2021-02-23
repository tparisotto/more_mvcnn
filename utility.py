import numpy as np
from skimage.feature import peak_local_max
import os
import shutil
import datetime
import open3d as o3d

CLASSES = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
           'monitor', 'night_stand', 'sofa', 'table', 'toilet']


def normalize3d(vector):
    np_arr = np.asarray(vector)
    max_val = np.max(np_arr)
    np_normalized = np_arr / max_val
    return o3d.utility.Vector3dVector(np_normalized)


def int_to_1hot(n, dim):
    vec = np.zeros(dim)
    vec[n] = 1
    return vec


def view_vector(data, dim):
    res = np.zeros(dim)
    for n in data:
        res[n] = 1
    return res


def make_dir(path, delete=False):
    if os.path.exists(path) and delete is True:
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.mkdir(path)


def extract_labels(data):
    mat = np.zeros((60, 1))
    for i in range(0, 60):
        entropy = float(data[data['view_code'] == i].entropy)
        mat[i] = entropy
    mat.resize((5, 12))
    coords = peak_local_max(mat, min_distance=1, exclude_border=False)
    labels = []
    for (y, x) in coords:
        labels.append((y * 12) + x)
    # fig, ax = plt.subplots(1)
    # ax.imshow(mat, cmap='rainbow')
    # for i in range(len(coords)):
    #     circle = plt.Circle((coords[i][1], coords[i][0]), radius=0.2, color='black')
    #     ax.add_patch(circle)
    #
    # plt.xticks([i for i in range(12)], [i*30 for i in range(12)])
    # plt.yticks([i for i in range(5)], [(i+1) * 30 for i in range(5)])
    # plt.show()

    return labels


def get_labels_from_object_views(data):
    subset_labels = extract_labels(data)
    # subset_idx = []
    # for lab in subset_labels:
    #     subset_idx.append(label2idx[lab])
    subset_labels.sort()
    return subset_labels


def get_label_dict(inverse=False):
    label2int = {'bathtub': 0,
                 'bed': 1,
                 'chair': 2,
                 'desk': 3,
                 'dresser': 4,
                 'monitor': 5,
                 'night_stand': 6,
                 'sofa': 7,
                 'table': 8,
                 'toilet': 9}

    int2label = {0: 'bathtub',
                 1: 'bed',
                 2: 'chair',
                 3: 'desk',
                 4: 'dresser',
                 5: 'monitor',
                 6: 'night_stand',
                 7: 'sofa',
                 8: 'table',
                 9: 'toilet'}
    if inverse:
        return int2label
    else:
        return label2int


def get_datastamp():
    time = datetime.datetime.now()
    return time.strftime("%d-%b-%H%M%S")
