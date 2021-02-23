"""
Script to post-process a set of depth-images.
"""

import os
import sys
import argparse
import numpy as np
import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
parser.add_argument('-v', '--verbose', action='store_true')
args = parser.parse_args()

BASE_DIR = sys.path[0]
DATA_DIR = os.path.join(BASE_DIR, args.data_dir)

files = os.listdir(DATA_DIR)
for filename in tqdm(files):
    if filename.endswith('png'):
        if args.verbose:
            print(f"[INFO] Current file: {filename}")
        path = os.path.join(DATA_DIR, filename)
        image = cv2.imread(path)
        result = cv2.normalize(image, image, 0, 255, norm_type=cv2.NORM_MINMAX)
        cv2.imwrite(path, result)


