import os
import sys
import argparse
import numpy as np
import pandas as pd
import open3d
import cv2
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn import preprocessing, metrics
from skimage.measure import shannon_entropy
from skimage.feature import peak_local_max
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('data')
args = parser.parse_args()

CLASSES = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
           'monitor', 'night_stand', 'sofa', 'table', 'toilet']
TMP_DIR = os.path.join(sys.path[0], 'tmp')


def normalize3d(vector):
    np_arr = np.asarray(vector)
    max_val = np.max(np_arr)
    np_normalized = np_arr / max_val
    return open3d.utility.Vector3dVector(np_normalized)


def nonblocking_custom_capture(pcd, rot_xyz, last_rot, view_index, capture='depth'):
    vis = open3d.visualization.Visualizer()
    vis.create_window(width=224, height=224, visible=False)
    vis.add_geometry(pcd)
    # Rotate back from last rotation
    R_0 = pcd.get_rotation_matrix_from_xyz(last_rot)
    pcd.rotate(np.linalg.inv(R_0), center=pcd.get_center())
    # Then rotate to the next rotation
    R = pcd.get_rotation_matrix_from_xyz(rot_xyz)
    pcd.rotate(R, center=pcd.get_center())
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    if capture == 'depth':
        vis.capture_depth_image(
            "{}/{}_x_{}_y_{}.png".format(TMP_DIR, view_index, -round(np.rad2deg(rot_xyz[0])),
                                         round(np.rad2deg(rot_xyz[2]))), depth_scale=10000)
    elif capture == 'image':
        vis.capture_screen_image(
            "{}/{}_x_{}_y_{}.png".format(TMP_DIR, view_index, -round(np.rad2deg(rot_xyz[0])),
                                         round(np.rad2deg(rot_xyz[2]))))
    else:
        print(f"[ERROR] capture: '{capture}' not recognized. Nothing was saved.")
    vis.destroy_window()


def entropy_distribution(label):
    data = pd.read_csv(args.data)
    chairs = data[data['label'] == label]
    distrib = np.zeros(60)
    for i in range(int(len(chairs) / 60)):
        obj = chairs[chairs['object_index'] == i + 1].sort_values(by=['view_code'])
        for j in range(60):
            distrib[j] = distrib[j] + obj[obj['view_code'] == j].entropy
    distrib = np.array(distrib)
    distrib = distrib / (len(chairs) / 60)
    # plt.bar(x=np.arange(0, 60), height=distrib)
    entropies = np.resize(distrib, (5, 12))

    fig, ax = plt.subplots(1, figsize=(12, 8))
    image = ax.imshow(entropies, cmap='rainbow')
    ax.set_title(f"Mean Entropy Map - {label.capitalize()}")
    for i in range(5):
        for j in range(12):
            value = entropies[i, j]
            ax.annotate(f'{value:.2f}', xy=(j - 0.2, i + 0.1))

    plt.xticks([i for i in range(12)], [i * 30 for i in range(12)])
    plt.yticks([i for i in range(5)], [(i + 1) * 30 for i in range(5)])
    plt.savefig(f"/Users/tommaso/Downloads/menmap_{label}.png")


def custom_parser(string):
    number = int(string.split("_")[0])
    return number


def entropy_plot2():
    os.mkdir(TMP_DIR)
    VIEW_INDEX = 0
    FILENAME = os.path.join(sys.path[0], args.data)
    label = os.path.split(FILENAME)[-1].split(".")[0].split("_")[0]
    mesh = open3d.io.read_triangle_mesh(FILENAME)
    mesh.vertices = normalize3d(mesh.vertices)
    mesh.compute_vertex_normals()
    rotations = []
    for j in range(0, 5):
        for i in range(0, 12):
            # Excluding 'rings' on 0 and 180 degrees since it would be the same projection but rotated
            rotations.append((-(j + 1) * np.pi / (5 + 1), 0, i * 2 * np.pi / 12))
    last_rotation = (0, 0, 0)
    for rot in rotations:
        nonblocking_custom_capture(mesh, rot, last_rotation, VIEW_INDEX)
        VIEW_INDEX = VIEW_INDEX + 1
        last_rotation = rot

    true_entropies = []
    for filename in sorted(os.listdir(TMP_DIR), key=custom_parser):
        if "png" in filename:
            image = cv2.imread(os.path.join(TMP_DIR, filename))
            image = cv2.normalize(image, image, 0, 255, norm_type=cv2.NORM_MINMAX)
            true_entropies.append(shannon_entropy(image))

    entropy_model = keras.models.load_model(
        '/Users/tommaso/Documents/RUG/ResearchProject/data/voxnet-157_val_loss-0.133.h5')
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

    true_entropies = np.resize(true_entropies, (5, 12))
    pred_entropies = np.resize(pred_entropies, (5, 12))

    fig, ax = plt.subplots(2, 1)
    image = ax[0].imshow(true_entropies, cmap='rainbow')
    ax[0].set_xlabel("Theta (\u03B8)", fontsize='large')
    ax[0].set_ylabel("Phi (\u03A6)", fontsize='large')
    ax[0].set_title(f"Original Entropy Map - {label.capitalize()}")
    for i in range(5):
        for j in range(12):
            value = true_entropies[i, j]
            ax[0].annotate(f'{value:.2f}', xy=(j - 0.2, i + 0.1))
    # coords = peak_local_max(true_entropies, min_distance=1, exclude_border=False)
    # peak_views = []
    # for (y, x) in coords:
    #     peak_views.append((y * 12) + x)
    # peak_views = sorted(peak_views)
    # for i in range(len(coords)):
    #     circle = plt.Circle((coords[i][1], coords[i][0]), radius=0.2, color='black')
    #     ax[0].add_patch(circle)

    image = ax[1].imshow(pred_entropies, cmap='rainbow')
    ax[1].set_xlabel("Theta (\u03B8)", fontsize='large')
    ax[1].set_ylabel("Phi (\u03A6)", fontsize='large')
    ax[1].set_title(f"Predicted Entropy Map - {label.capitalize()}")
    for i in range(5):
        for j in range(12):
            value = pred_entropies[i, j]
            ax[1].annotate(f'{value:.2f}', xy=(j - 0.2, i + 0.1))
    # coords = peak_local_max(pred_entropies, min_distance=1, exclude_border=False)
    # peak_views = []
    # for (y, x) in coords:
    #     peak_views.append((y * 12) + x)
    # peak_views = sorted(peak_views)
    # for i in range(len(coords)):
    #     circle = plt.Circle((coords[i][1], coords[i][0]), radius=0.2, color='black')
    #     ax[1].add_patch(circle)

    ax[0].set_xticks([i for i in range(12)])
    ax[0].set_xticklabels([i * 30 for i in range(12)])
    ax[0].set_yticks([i for i in range(5)])
    ax[0].set_yticklabels([(i + 1) * 30 for i in range(5)])
    ax[1].set_xticks([i for i in range(12)])
    ax[1].set_xticklabels([i * 30 for i in range(12)])
    ax[1].set_yticks([i for i in range(5)])
    ax[1].set_yticklabels([(i + 1) * 30 for i in range(5)])
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    for im in os.listdir(TMP_DIR):
        os.remove(os.path.join(TMP_DIR, im))
    os.rmdir(TMP_DIR)


def entropy_plot3():
    os.mkdir(TMP_DIR)
    VIEW_INDEX = 0
    FILENAME = os.path.join(sys.path[0], args.data)
    label = os.path.split(FILENAME)[-1].split(".")[0].split("_")[0]
    mesh = open3d.io.read_triangle_mesh(FILENAME)
    mesh.vertices = normalize3d(mesh.vertices)
    mesh.compute_vertex_normals()
    rotations = []
    for j in range(0, 5):
        for i in range(0, 12):
            rotations.append((-(j + 1) * np.pi / (5 + 1), 0, i * 2 * np.pi / 12))
    last_rotation = (0, 0, 0)
    for rot in rotations:
        nonblocking_custom_capture(mesh, rot, last_rotation, VIEW_INDEX)
        VIEW_INDEX = VIEW_INDEX + 1
        last_rotation = rot

    true_entropies = []
    depth_images = []
    for filename in sorted(os.listdir(TMP_DIR), key=custom_parser):
        if "png" in filename:
            image = cv2.imread(os.path.join(TMP_DIR, filename))
            image = cv2.normalize(image, image, 0, 255, norm_type=cv2.NORM_MINMAX)
            depth_images.append(image)
            true_entropies.append(shannon_entropy(image))

    entropy_model = keras.models.load_model(
        '/Users/tommaso/Documents/RUG/ResearchProject/data/voxnet-157_val_loss-0.133.h5')
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

    true_entropies = np.resize(true_entropies, (5, 12))
    pred_entropies = np.resize(pred_entropies, (5, 12))

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(nrows=6, ncols=6, wspace=0.2, hspace=0.2)
    ax0 = fig.add_subplot(gs[:3, :5])
    ax1 = fig.add_subplot(gs[3:, :5])

    ax0.imshow(true_entropies, cmap='rainbow')
    ax0.set_xlabel("Theta (\u03B8)", fontsize='large')
    ax0.set_ylabel("Phi (\u03A6)", fontsize='large')
    ax0.set_title(f"Original Entropy Map")
    for i in range(5):
        for j in range(12):
            value = true_entropies[i, j]
            ax0.annotate(f'{value:.2f}', xy=(j - 0.2, i + 0.1))

    ax1.imshow(pred_entropies, cmap='rainbow')
    ax1.set_xlabel("Theta (\u03B8)", fontsize='large')
    ax1.set_ylabel("Phi (\u03A6)", fontsize='large')
    ax1.set_title(f"Predicted Entropy Map")
    for i in range(5):
        for j in range(12):
            value = pred_entropies[i, j]
            ax1.annotate(f'{value:.2f}', xy=(j - 0.2, i + 0.1))

    ax0.set_xticks([i for i in range(12)])
    ax0.set_xticklabels([i * 30 for i in range(12)])
    ax0.set_yticks([i for i in range(5)])
    ax0.set_yticklabels([(i + 1) * 30 for i in range(5)])
    ax1.set_xticks([i for i in range(12)])
    ax1.set_xticklabels([i * 30 for i in range(12)])
    ax1.set_yticks([i for i in range(5)])
    ax1.set_yticklabels([(i + 1) * 30 for i in range(5)])

    peaks = peak_local_max(pred_entropies, min_distance=1, exclude_border=False, indices=True)
    x = peaks[:, 0]
    y = peaks[:, 1]
    for i in range(6):
        ax2 = fig.add_subplot(gs[i, 5])
        _x = 30.0 * (1 + x[i])
        _y = 30.0 * y[i]
        vc = x[i] * 12 + y[i]
        filename = f"{vc}_x_{_x}_y_{_y}.png"
        image = cv2.imread(os.path.join(TMP_DIR, filename))
        ax2.imshow(image)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title(f"({_y}, {_x}) - H={pred_entropies[x[i],y[i]]:.3f}   ")

    plt.show()
    for im in os.listdir(TMP_DIR):
        os.remove(os.path.join(TMP_DIR, im))
    os.rmdir(TMP_DIR)


def entropy_prediction():
    FILENAME = args.data
    entropy_model = keras.models.load_model(
        '/Users/tommaso/Documents/RUG/ResearchProject/data/voxnet-157_val_loss-0.133.h5')
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

    pred_entropies = np.rot90(pred_entropies)

    fig, ax = plt.subplots(1)
    image = ax.imshow(pred_entropies, cmap='rainbow')
    ax.set_ylabel("Theta (\u03B8)", fontsize='large')
    ax.set_xlabel("Phi (\u03A6)", fontsize='large')
    ax.set_title(f"Predicted Entropy Map")

    coords = peak_local_max(pred_entropies, min_distance=1, exclude_border=False)
    peak_views = []
    for (y, x) in coords:
        peak_views.append((y * 12) + x)
    peak_views = sorted(peak_views)
    for i in range(len(coords)):
        circle = plt.Circle((coords[i][1], coords[i][0]), radius=0.2, color='black')
        ax.add_patch(circle)

    ax.set_yticks([i for i in range(12)])
    ax.set_yticklabels([i * 30 for i in range(12, 0, -1)])
    ax.set_xticks([i for i in range(5)])
    ax.set_xticklabels([(i + 1) * 30 for i in range(5)])
    plt.show()


def model_sample(file):
    file = os.path.join(sys.path[0], file)
    mesh = open3d.io.read_triangle_mesh(file)
    label = os.path.split(file)[-1].split(".")[0].split("_")[:-1][0]
    mesh.vertices = normalize3d(mesh.vertices)
    mesh.compute_vertex_normals()
    rot = (-2 * np.pi / 6, 0, 4 * np.pi / 12)
    R = mesh.get_rotation_matrix_from_xyz(rot)
    mesh.rotate(R, center=mesh.get_center())
    vis = open3d.visualization.Visualizer()
    vis.create_window(width=224, height=224, visible=False)
    vis.add_geometry(mesh)
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(f"/Users/tommaso/Downloads/{label}_sample.png")
    vis.destroy_window()


def historyplot():
    sns.set_theme(style='darkgrid')
    history = pd.read_csv(args.data)
    history = history.rename(columns={'Unnamed: 0': 'epochs'})
    history = history[['epochs', 'class_recall', 'val_class_recall']]
    print(history.head())
    data_preproc = pd.melt(history, ['epochs'])
    sns.lineplot(x='epochs', y='value', hue='variable', data=data_preproc)
    plt.show()


def conf_mat():
    data = pd.read_csv(args.data)
    lab_enc = preprocessing.LabelEncoder()
    lab_enc.fit(CLASSES)
    true_labels = lab_enc.transform(data['true_label'])
    pred_labels = lab_enc.transform(data['pred_label'])
    offset_theta = data['offset_theta']
    offset_phi = data['offset_phi']
    conf_matrix = metrics.confusion_matrix(y_true=true_labels, y_pred=pred_labels, normalize='true')
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=CLASSES)
    disp.plot(cmap='Blues', xticks_rotation='vertical')
    plt.show()


def custom_draw_geometry(pcd):
    # The following code achieves the same effect as:
    # o3d.visualization.draw_geometries([pcd])
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    for el in pcd:
        vis.add_geometry(el)
    vis.run()
    vis.destroy_window()


def object_show():
    mesh = open3d.io.read_triangle_mesh(args.data)
    mesh.vertices = normalize3d(mesh.vertices)
    mesh.scale(0.95 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())
    # center = (mesh.get_max_bound() + mesh.get_min_bound()) / 2
    mesh = mesh.translate((0.5, 0.5, 0.5))
    mesh.compute_vertex_normals()
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points),
        lines=open3d.utility.Vector2iVector(lines),
    )
    line_set.colors = open3d.utility.Vector3dVector(colors)
    custom_draw_geometry([line_set, mesh])


def plot_best_views():
    os.mkdir(TMP_DIR)
    VIEW_INDEX = 0
    FILENAME = os.path.join(sys.path[0], args.data)
    mesh = open3d.io.read_triangle_mesh(FILENAME)
    mesh.vertices = normalize3d(mesh.vertices)
    mesh.compute_vertex_normals()
    rotations = []
    for j in range(0, 5):
        for i in range(0, 12):
            rotations.append((-(j + 1) * np.pi / (5 + 1), 0, i * 2 * np.pi / 12))
    last_rotation = (0, 0, 0)
    for rot in rotations:
        nonblocking_custom_capture(mesh, rot, last_rotation, VIEW_INDEX, capture='image')
        VIEW_INDEX = VIEW_INDEX + 1
        last_rotation = rot

    views_images = []
    for filename in sorted(os.listdir(TMP_DIR), key=custom_parser):
        if "png" in filename:
            image = cv2.imread(os.path.join(TMP_DIR, filename))
            image = cv2.normalize(image, image, 0, 255, norm_type=cv2.NORM_MINMAX)
            views_images.append(image)

    entropy_model = keras.models.load_model(
        '/Users/tommaso/Documents/RUG/ResearchProject/data/voxnet-157_val_loss-0.133.h5')
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

    fig, ax = plt.subplots(1, 5)
    peaks = peak_local_max(pred_entropies, min_distance=1, exclude_border=False, indices=True)
    x = peaks[:, 0]
    y = peaks[:, 1]
    for i in range(5):
        _x = 30.0 * (1 + x[i])
        _y = 30.0 * y[i]
        vc = x[i] * 12 + y[i]
        filename = f"{vc}_x_{_x}_y_{_y}.png"
        image = cv2.imread(os.path.join(TMP_DIR, filename))
        ax[i].imshow(image)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(f"({_y}, {_x}) - H={pred_entropies[x[i],y[i]]:.3f}")

    plt.show()
    for im in os.listdir(TMP_DIR):
        os.remove(os.path.join(TMP_DIR, im))
    os.rmdir(TMP_DIR)


def acc_score():
    data = pd.read_csv(args.data)
    n_objects = data.shape[0]
    lab_enc = preprocessing.LabelEncoder()
    lab_enc.fit(CLASSES)
    true_labels = lab_enc.transform(data['true_label'])
    pred_labels = lab_enc.transform(data['pred_label'])
    class_accuracy = metrics.accuracy_score(true_labels, pred_labels)

    correct = 0
    for index, row in data.iterrows():
        if row['offset_theta'] == 0 and row['offset_phi'] == 0:
            correct = correct + 1

    print(f'Class Acc.: {class_accuracy} - View Acc.: {correct / n_objects}')


def main():
    plot_best_views()


if __name__ == '__main__':
    main()
