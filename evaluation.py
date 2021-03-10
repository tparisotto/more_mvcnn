"""
Performs prediction of the model on a test set.
"""

import os
import sys
import glob
import argparse
import numpy as np
import open3d
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from utility import normalize3d, get_datastamp, get_label_dict
from skimage.feature import peak_local_max
from time import time
from tqdm import tqdm
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--modelnet10')
parser.add_argument("--entropy_model")
parser.add_argument("--classifier_model")
parser.add_argument("--name")
parser.add_argument("--topk")
parser.add_argument("--view_dataset")
args = parser.parse_args()
BASE_DIR = sys.path[0]
DATA_PATH = os.path.join(BASE_DIR, args.modelnet10)
TIMESTAMP = get_datastamp()
tmp = tempfile.mkdtemp()
TMP_DIR = os.path.join(BASE_DIR, tmp)


class ViewData:
    """ Class to keep track of attributes of the views. """
    obj_label = ''
    obj_index = 1
    view_index = 0
    phi = 0
    theta = 0
    voxel_size = float(1 / 50)
    n_voxel = 50


CLASSES = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
           'monitor', 'night_stand', 'sofa', 'table', 'toilet']

idx2rot = {}
count = 0
for _phi in range(30, 151, 30):
    for _theta in range(0, 331, 30):
        idx2rot[count] = (_phi, _theta)
        count += 1


def nonblocking_custom_capture(mesh, rot_xyz, last_rot):
    ViewData.phi = -round(np.rad2deg(rot_xyz[0]))
    ViewData.theta = round(np.rad2deg(rot_xyz[2]))
    vis = open3d.visualization.Visualizer()
    vis.create_window(width=224, height=224, visible=False)
    # Rotate back from last rotation
    R_0 = mesh.get_rotation_matrix_from_xyz(last_rot)
    mesh.rotate(np.linalg.inv(R_0), center=mesh.get_center())
    # Then rotate to the next rotation
    R = mesh.get_rotation_matrix_from_xyz(rot_xyz)
    mesh.rotate(R, center=mesh.get_center())
    vis.add_geometry(mesh)
    vis.poll_events()
    path = f"{TMP_DIR}/view_theta_{int(ViewData.theta)}_phi_{int(ViewData.phi)}.png"
    vis.capture_screen_image(path)
    vis.destroy_window()


def classify(off_file, entropy_model, classifier):
    FILENAME = off_file
    mesh = open3d.io.read_triangle_mesh(FILENAME)
    mesh.vertices = normalize3d(mesh.vertices)
    mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())
    center = (mesh.get_max_bound() + mesh.get_min_bound()) / 2
    mesh = mesh.translate((-center[0], -center[1], -center[2]))
    voxel_grid = open3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(input=mesh,
                                                                                   voxel_size=1 / 50,
                                                                                   min_bound=np.array(
                                                                                       [-0.5, -0.5, -0.5]),
                                                                                   max_bound=np.array([0.5, 0.5, 0.5]))
    voxels = voxel_grid.get_voxels()
    grid_size = 50
    mask = np.zeros((grid_size, grid_size, grid_size))
    for vox in voxels:
        mask[vox.grid_index[0], vox.grid_index[1], vox.grid_index[2]] = 1
    mask = np.pad(mask, 3, 'constant')
    mask = np.resize(mask, (1, mask.shape[0], mask.shape[1], mask.shape[2], 1))
    pred_entropies = entropy_model.predict(mask)
    pred_entropies = np.resize(pred_entropies, (5, 12))
    coords = peak_local_max(pred_entropies, min_distance=1, exclude_border=False)
    peak_views = []
    for (y, x) in coords:
        peak_views.append((y * 12) + x)
    peak_views = sorted(peak_views)
    mesh = open3d.io.read_triangle_mesh(FILENAME)
    mesh.vertices = normalize3d(mesh.vertices)
    mesh.compute_vertex_normals()
    rotations = []
    for j in range(5):
        for i in range(12):
            if ((j * 12) + i) in peak_views:
                rotations.append((-(j + 1) * np.pi / 6, 0, i * 2 * np.pi / 12))
    last_rotation = (0, 0, 0)
    for rot in rotations:
        nonblocking_custom_capture(mesh, rot, last_rotation)
        last_rotation = rot
    views = []
    views_images = []
    views_images_dir = os.listdir(TMP_DIR)
    for file in views_images_dir:
        if '.png' in file:
            im = plt.imread(os.path.join(TMP_DIR, file))
            views_images.append(im)
            phi = int(file.split(".")[0].split("_")[-1])
            theta = int(file.split(".")[0].split("_")[-3])
            views.append((theta, phi))
    views_images = np.array(views_images)
    results = classifier.predict(views_images)
    labels = results[0]
    pred_views = results[1]
    for im in os.listdir(TMP_DIR):
        os.remove(os.path.join(TMP_DIR, im))
    return labels, pred_views, views


def classify_topk(off_file, entropy_model, classifier, k):
    FILENAME = off_file
    mesh = open3d.io.read_triangle_mesh(FILENAME)
    mesh.vertices = normalize3d(mesh.vertices)
    mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())
    center = (mesh.get_max_bound() + mesh.get_min_bound()) / 2
    mesh = mesh.translate((-center[0], -center[1], -center[2]))
    # print("[DEBUG] Start voxelization")
    voxel_grid = open3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(input=mesh,
                                                                                   voxel_size=1 / 50,
                                                                                   min_bound=np.array(
                                                                                       [-0.5, -0.5, -0.5]),
                                                                                   max_bound=np.array([0.5, 0.5, 0.5]))
    voxels = voxel_grid.get_voxels()
    grid_size = 50
    mask = np.zeros((grid_size, grid_size, grid_size))
    for vox in voxels:
        mask[vox.grid_index[0], vox.grid_index[1], vox.grid_index[2]] = 1
    mask = np.pad(mask, 3, 'constant')
    mask = np.resize(mask, (1, mask.shape[0], mask.shape[1], mask.shape[2], 1))
    # print("[DEBUG] Start 1st prediction")
    pred_entropies = entropy_model.predict(mask)
    pred_entropies = np.resize(pred_entropies, 60)
    topk_views = np.array(pred_entropies).argsort()[-k:][::-1]


    views = []
    views_images = []

    for viewpoint in topk_views:
        name = os.path.split(off_file)[-1].rstrip('.off')
        view_dir = os.path.join(BASE_DIR, args.view_dataset, 'image')
        file = glob.glob(os.path.join(view_dir, f"{name}*vc_{viewpoint}.png"))[0]
        # print(f"[DEBUG] vc: {viewpoint}")
        im = plt.imread(file)
        views_images.append(im)
        file = os.path.split(file)[-1]
        phi = int(file.split(".")[0].split("_")[-3])
        # print(f"[DEBUG] phi: {phi}")
        theta = int(file.split(".")[0].split("_")[-5])
        # print(f"[DEBUG] theta: {theta}")
        views.append((theta, phi))
    views_images = np.array(views_images)
    # print("[DEBUG] Start 2nd prediction")
    results = classifier.predict(views_images)
    labels = results[0]
    pred_views = results[1]
    for im in os.listdir(TMP_DIR):
        os.remove(os.path.join(TMP_DIR, im))
    return labels, pred_views, views


def most_common(lst):
    return max(set(lst), key=lst.count)


def mode_rows(a):
    a = np.ascontiguousarray(a)
    void_dt = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[1:])))
    _, ids, _count = np.unique(a.view(void_dt).ravel(), return_index=True, return_counts=True)
    largest_count_id = ids[_count.argmax()]
    most_frequent_row = a[largest_count_id]
    return most_frequent_row


def main():
    print(f"[INFO] Loading models...")
    entropy_model = keras.models.load_model(args.entropy_model)
    classifier = keras.models.load_model(args.classifier_model)
    print(f"[INFO] Models loaded.")
    vec2lab = get_label_dict(inverse=True)
    FIRST_OBJECT = True
    for lab in CLASSES:
        test_files = sorted(os.listdir(os.path.join(BASE_DIR, DATA_PATH, lab, 'test')))
        object_index, labels_true, labels_pred, offset_phi, offset_theta, n_views = [], [], [], [], [], []
        for x in tqdm(test_files):
            if '.off' in x:
                x = os.path.join(BASE_DIR, DATA_PATH, lab, 'test', x)
                if args.topk:
                    labels, pred_views, views = classify_topk(x, entropy_model, classifier, int(args.topk))
                else:
                    labels, pred_views, views = classify(x, entropy_model, classifier)
                # for i in range(len(labels)):
                #     cl = vec2lab[np.argmax(labels[i])]
                #     pv = idx2rot[int(np.argmax(pred_views[i]))]
                #     tv = views[i]
                labint = []
                for el in labels:
                    labint.append(np.argmax(el))
                pred_class = vec2lab[most_common(labint)]
                angles = []
                pred_angles = []
                for i in range(len(labels)):
                    angles.append(views[i])
                    pred_angles.append(idx2rot[int(np.argmax(pred_views[i]))])
                angles = np.array(angles)
                pred_angles = np.array(pred_angles)
                offset = mode_rows(pred_angles - angles)

                object_index.append(x.rstrip(".off").split("_")[-1])
                labels_true.append(lab)
                labels_pred.append(pred_class)
                offset_theta.append(offset[0])
                offset_phi.append(offset[1])
                n_views.append(len(views))

        csv = pd.DataFrame({"true_label": labels_true,
                            "object_index": object_index,
                            "pred_label": labels_pred,
                            "offset_theta": offset_theta,
                            "offset_phi": offset_phi,
                            "n_views": n_views})

        if FIRST_OBJECT:  # Create the main DataFrame and csv, next ones will be appended
            FIRST_OBJECT = False
            csv.to_csv(os.path.join(BASE_DIR, f"evaluation_results_{args.name}.csv"), index=False)
        else:
            csv.to_csv(os.path.join(BASE_DIR, f"evaluation_results_{args.name}.csv"), index=False, mode='a',
                       header=False)

    os.rmdir(TMP_DIR)


if __name__ == '__main__':
    main()
