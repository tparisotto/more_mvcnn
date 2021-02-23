"""
Generates a dataset in CSV format of depth-views entropy values.

This script register the entropy of a set of viewpoints regularly distributed on a sphere.
The number of viewpoints is determined by the -x and -y values, which determine
respectively the yaw and pitch values density.

Example: -x=12 --> yaw values: phi=[0, 30, 60, 90, 120, ... , 330]
         -y=5 --> pitch values: theta=[30, 60, 90, 120, 150]

The entropy values are stored in the entropy_dataset.csv file.
"""

import os
import sys
import argparse
import shutil
from open3d import *
import open3d as o3d
import cv2
from utility import normalize3d
import numpy as np
import pandas as pd
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Generates a dataset in CSV format of depth-views entropy values.")
parser.add_argument("--modelnet10", help="Specify root directory to the ModelNet10 dataset.", required=True)
parser.add_argument("--out", help="Select a desired output directory.", default="./")
parser.add_argument("-v", "--verbose", help="Prints current state of the program while executing.", action='store_true')
parser.add_argument("-x", "--horizontal_split", help="Number of views from a single ring. Each ring is divided in x "
                                                     "splits so each viewpoint is at an angle of multiple of 360/x. "
                                                     "Example: -x=12 --> phi=[0, 30, 60, 90, 120, ... , 330].",
                    default=12,
                    metavar='VALUE',
                    type=int
                    )
parser.add_argument("-y", "--vertical_split", help="Number of horizontal rings. Each ring of viewpoints is an "
                                                   "horizontal section of a sphere, looking at the center at an angle "
                                                   "180/(y+1), skipping 0 and 180 degrees (since the diameter of the "
                                                   "ring would be zero). Example: -y=5 --> theta=[30, 60, 90, 120, "
                                                   "150] ; -y=11 --> theta=[15, 30, 45, ... , 150, 165]. ",
                    default=5,
                    metavar='VALUE',
                    type=int
                    )
parser.add_argument("--debug", help="Prints debug statements during runtime.", action='store_true')
args = parser.parse_args()

BASE_DIR = sys.path[0]
OUT_DIR = os.path.join(BASE_DIR, args.out)
DATA_PATH = os.path.join(BASE_DIR, args.data)
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
N_VIEWS_W = args.horizontal_split
N_VIEWS_H = args.vertical_split
VIEW_INDEX = 0
FIRST_OBJECT = 1


def nonblocking_custom_capture(tr_mesh, rot_xyz, last_rot=(0, 0, 0)):
    """
    Custom function for Open3D to allow non-blocking capturing on a headless server.

    The function renders a triangle mesh file in a 224x224 window capturing a depth-image
    from the rot_xyz rotation of the object. Stores the image in the ./tmp/ folder.

    :param tr_mesh: open3d.geometry.TriangleMesh object to render.
    :param rot_xyz: (x, y, z)-tuple with rotation values.
    :param last_rot: if the function is being called within a loop of rotations,
     specify the previous rot_xyz to reposition the object to thee original rotation before
     applying rot_xyz
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=224, height=224, visible=False)
    vis.add_geometry(tr_mesh)
    # Rotate back from last rotation
    R_0 = tr_mesh.get_rotation_matrix_from_xyz(last_rot)
    tr_mesh.rotate(np.linalg.inv(R_0), center=tr_mesh.get_center())
    # Then rotate to the next rotation
    R = tr_mesh.get_rotation_matrix_from_xyz(rot_xyz)
    tr_mesh.rotate(R, center=tr_mesh.get_center())
    vis.update_geometry(tr_mesh)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_depth_image(
        "{}/tmp/{}_{}_x_{}_y_{}.png".format(BASE_DIR, label, VIEW_INDEX, -round(np.rad2deg(rot_xyz[0])),
                                            round(np.rad2deg(rot_xyz[2]))), depth_scale=10000)
    vis.destroy_window()
    depth = cv2.imread("{}/tmp/{}_{}_x_{}_y_{}.png".format(BASE_DIR, label, VIEW_INDEX, -round(np.rad2deg(rot_xyz[0])),
                                                           round(np.rad2deg(rot_xyz[2]))))
    result = cv2.normalize(depth, depth, 0, 255, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite("{}/tmp/{}_{}_x_{}_y_{}.png".format(BASE_DIR, label, VIEW_INDEX, -round(np.rad2deg(rot_xyz[0])),
                                                    round(np.rad2deg(rot_xyz[2]))), result)


labels = []
for cur in os.listdir(DATA_PATH):
    if os.path.isdir(os.path.join(DATA_PATH, cur)):
        labels.append(cur)

TMP_DIR = os.path.join(BASE_DIR, "tmp")
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR)
else:
    os.makedirs(TMP_DIR)

for split_set in ['train', 'test']:
    for label in labels:
        files = os.listdir(os.path.join(DATA_PATH, label, split_set))
        files.sort()
        for file in files:  # Removes file without .off extension
            if not file.endswith('off'):
                files.remove(file)

        for file in files:
            OBJECT_INDEX = file.split('.')[0].split('_')[-1]
            VIEW_INDEX = 0
            filepath = os.path.join(DATA_PATH, label, split_set, file)

            mesh = io.read_triangle_mesh(filepath)
            mesh.vertices = normalize3d(mesh.vertices)
            mesh.compute_vertex_normals()

            if args.debug:
                print(f"[DEBUG] Current Object: {file}")

            rotations = []
            for j in range(0, N_VIEWS_H):
                for i in range(N_VIEWS_W):
                    # Excluding 'rings' on 0 and 180 degrees since it would be the same projection but rotated
                    rotations.append((-(j + 1) * np.pi / (N_VIEWS_H + 1), 0, i * 2 * np.pi / N_VIEWS_W))
            last_rotation = (0, 0, 0)
            for rot in rotations:
                nonblocking_custom_capture(mesh, rot, last_rotation)
                VIEW_INDEX = VIEW_INDEX + 1
                last_rotation = rot

            data_label = []
            data_code = []
            data_x = []
            data_y = []
            entropy = []
            data_index = []

            for filename in os.listdir(TMP_DIR):
                if "png" in filename:  # skip auto-generated .DS_Store
                    if "night_stand" in filename:
                        data_label.append("night_stand")
                        data_y.append(float((filename.split("_")[-1]).split(".")[0]))
                        data_x.append(float((filename.split("_")[-3]).split(".")[0]))
                        data_code.append(int((filename.split("_")[2])))
                        data_index.append(int(OBJECT_INDEX))
                        image = plt.imread(os.path.join(TMP_DIR, filename))
                        entropy.append(shannon_entropy(image))
                    else:
                        data_label.append(filename.split("_")[0])
                        data_y.append(float((filename.split("_")[-1]).split(".")[0]))
                        data_x.append(float((filename.split("_")[-3]).split(".")[0]))
                        data_code.append(int((filename.split("_")[1])))
                        data_index.append(int(OBJECT_INDEX))
                        image = plt.imread(os.path.join(TMP_DIR, filename))
                        entropy.append(shannon_entropy(image))

            data = pd.DataFrame({"label": data_label,
                                 "obj_ind": data_index,
                                 "code": data_code,
                                 "rot_x": data_x,
                                 "rot_y": data_y,
                                 "entropy": entropy})
            if FIRST_OBJECT == 1:  # Create the main DataFrame and csv, next ones will be appended
                FIRST_OBJECT = 0
                data.to_csv(os.path.join(OUT_DIR, "entropy_dataset.csv"), index=False)
            else:
                data.to_csv(os.path.join(OUT_DIR, "entropy_dataset.csv"), index=False, mode='a', header=False)

            for im in os.listdir(TMP_DIR):
                os.remove(os.path.join(TMP_DIR, im))


os.rmdir(TMP_DIR)
