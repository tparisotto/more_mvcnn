"""
Demonstration of classification and pose estimation of
a 3D object with entropy estimated multi-views based on depth-views.

--data: 3D object file
--entropy_model: Keras Model file for entropy estimation
--classifier_model: Keras Model file for single-view classification
--pose_theta: add a custom rotation to the object to simulate a different pose [EXPERIMENTAL]
--pose_phi: add a custom rotation to the object to simulate a different pose [EXPERIMENTAL]
"""
import os
import sys
import argparse
import numpy as np
import open3d
from tensorflow import keras
import utility
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--data")
parser.add_argument("--entropy_model")
parser.add_argument("--classifier_model")
parser.add_argument("--pose_theta", default=0)
parser.add_argument("--pose_phi", default=0)
args = parser.parse_args()
TMP_DIR = os.path.join(sys.path[0], "tmp")

pose_theta = int(args.pose_theta)
pose_phi = int(args.pose_phi)


class ViewData:
    obj_label = ''
    obj_index = 1
    view_index = 0
    phi = 0
    theta = 0
    voxel_size = float(1 / 50)
    n_voxel = 50


idx2rot = {}
count = 0
for _phi in range(30, 151, 30):
    for _theta in range(0, 331, 30):
        idx2rot[count] = (_theta, _phi)
        count += 1


def normalize3d(vector):
    np_arr = np.asarray(vector)
    max_val = np.max(np_arr)
    np_normalized = np_arr / max_val
    return open3d.utility.Vector3dVector(np_normalized)


def custom_parser(string):
    number = int(string.split("_")[0])
    return number


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
    os.mkdir(TMP_DIR)
    FILENAME = os.path.join(sys.path[0], off_file)
    mesh = open3d.io.read_triangle_mesh(FILENAME)
    mesh.vertices = normalize3d(mesh.vertices)
    mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())
    center = (mesh.get_max_bound() + mesh.get_min_bound()) / 2
    mesh = mesh.translate((-center[0], -center[1], -center[2]))
    # Add custom rotation for symulating different pose
    R = mesh.get_rotation_matrix_from_xyz((np.deg2rad(pose_phi), 0, np.deg2rad(pose_theta)))
    mesh.rotate(R, center=mesh.get_center())

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
    fig, ax = plt.subplots(1)
    image = ax.imshow(pred_entropies, cmap='rainbow')
    fig.colorbar(image, orientation='horizontal')
    for i in range(len(coords)):
        circle = plt.Circle((coords[i][1], coords[i][0]), radius=0.2, color='black')
        ax.add_patch(circle)

    plt.xticks([i for i in range(12)], [i * 30 for i in range(12)])
    plt.yticks([i for i in range(5)], [(i + 1) * 30 for i in range(5)])
    plt.title(f'Entropy Map of {os.path.split(off_file)[-1].rstrip(".off")}')
    plt.show()

    # print(f"[DEBUG] peak_views : {np.shape(peak_views)}")
    print(f"[DEBUG] peak_views : {peak_views}")
    # print(f"[DEBUG] _views-argwhere : {_views}")

    mesh = open3d.io.read_triangle_mesh(FILENAME)
    mesh.vertices = normalize3d(mesh.vertices)
    mesh.compute_vertex_normals()

    # Add custom rotation for symulating different pose
    R = mesh.get_rotation_matrix_from_xyz((np.deg2rad(pose_phi), 0, np.deg2rad(pose_theta)))
    mesh.rotate(R, center=mesh.get_center())

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
    i = 0
    for file in views_images_dir:
        if '.png' in file:
            i = i + 1
            plt.subplot(int(np.ceil(len(rotations) / 3)), 3, i)
            im = plt.imread(os.path.join(TMP_DIR, file))
            views_images.append(im)
            phi = int(file.split(".")[0].split("_")[-1])
            theta = int(file.split(".")[0].split("_")[-3])
            plt.imshow(im, cmap='gray')
            plt.title(label=f'({theta:.2f}, {phi:.2f})')
            plt.xticks([])
            plt.yticks([])

            views.append((theta, phi))

    views_images = np.array(views_images)
    plt.show()

    results = classifier.predict(views_images)
    labels = results[0]
    pred_views = results[1]
    for im in os.listdir(TMP_DIR):
        os.remove(os.path.join(TMP_DIR, im))
    os.rmdir(TMP_DIR)
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
    x = args.data
    labels, pred_views, views = classify(x, entropy_model, classifier)
    vec2lab = utility.get_label_dict(inverse=True)
    for i in range(len(labels)):
        cl = vec2lab[np.argmax(labels[i])]
        pv = idx2rot[int(np.argmax(pred_views[i]))]
        tv = views[i]
        print(
            f"[INFO] Predicted: {cl:<11} - {str(pv):<10} from {str(tv):<10} --> Offset: ({(np.array(pv) - np.array(tv))[0]}, "
            f"{(np.array(pv) - np.array(tv))[1]})")
    print(f"[INFO] Majority vote:")
    labint = []
    for el in labels:
        labint.append(np.argmax(el))
    print(f"    class: {vec2lab[most_common(labint)]}")
    angles = []
    pred_angles = []
    for i in range(len(labels)):
        angles.append(views[i])
        pred_angles.append(idx2rot[int(np.argmax(pred_views[i]))])
    angles = np.array(angles)
    pred_angles = np.array(pred_angles)
    offset = mode_rows(pred_angles - angles)
    print(f"    pose: theta={offset[0]} phi={offset[1]}")


if __name__ == '__main__':
    main()
