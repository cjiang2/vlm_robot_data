import os
import sys
import argparse
import json
import glob
from collections import OrderedDict

import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_part_of_object_annot_v2(
    dataset_path: str,
    video_name: str,
    ):
    """Get frame-by-frame constraint and event annotations.
    """
    # Load part-of-object annotation as ground-truth
    part_of_object_json = os.path.join(dataset_path, video_name, "constraints", "part_of_object.json")
    with open(part_of_object_json, "r") as f:
        annots = json.load(f)

    annots_reformed = OrderedDict()

    for annot in annots:
        video_name = annot["video_name"]
        frame_id = annot["i"]

        n_constraints = 0

        for k in annot.keys():
            if "constraint" in k:
                n_constraints += 1
        
        # One frame -> Many constraints
        img_path = os.path.join(dataset_path, video_name, "imgs", "{}.png".format(frame_id))

        # Constraints are listed as constraint_1, constraint_2, ...
        constraints = [annot["constraint_{}".format(i + 1)] for i in range(n_constraints)]

        annots_reformed[img_path] = constraints

    return annots_reformed


# ####################
# Image Geometry
# ####################

def get_object_orientation(
    mask,
    scale_axis1: float = 0.01,
    scale_axis2: float = 0.5,
    ):
    """Perform PCA over binary image.
    """
    points = cv2.findNonZero(mask).sum(axis=1).astype(np.float32)

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(points, mean)

    # Store the center of the object
    cntr = np.array([int(mean[0, 0]), int(mean[0, 1])])

    # Major
    p1 = (cntr[0] + scale_axis1 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + scale_axis1 * eigenvectors[0,1] * eigenvalues[0,0])

    # Minor
    p2 = (cntr[0] - scale_axis2 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - scale_axis2 * eigenvectors[1,1] * eigenvalues[1,0])
    
    return cntr.astype(np.float32), np.array(p1).astype(np.float32), np.array(p2).astype(np.float32)

def construct_image_midline(
    width: int = 640,
    height: int = 480,
    direction: str = "vertical",
    ):
    """Grab a perfect vertical or horizontal line by image size.
    """
    if direction == "vertical":
        x = width // 2
        y_step = height // 8
        y1 = y_step * 5
        y2 = y_step * 6
        p1 = [x, y1]
        p2 = [x, y2]
    else:
        x_step = width // 8
        y = height // 2
        x1 = x_step * 5
        x2 = x_step * 6
        p1 = [x1, y]
        p2 = [x2, y]
    return np.array(p1).astype(np.float32), np.array(p2).astype(np.float32)

def get_in_hand_camera_center_point(
    width: int = 640,
    height: int = 480,
    step: int = 3,
    offset_x: int = 25,
    ):
    """Grab a good point used to represent end-effector pos
    for eye-in-hand setup.
    """
    x = (width // 2) + offset_x
    y_step = height // 5
    y = y_step * step
    return np.array([x, y]).astype(np.float32)


# ####################
# Visualization
# ####################

def visualize_constraint(
    img,
    mask,
    constraint_name,
    ):
    center, component_1, component_2 = get_object_orientation(mask)
    center = center.astype(int)
    component_1 = component_1.astype(int)
    component_2 = component_2.astype(int)

    ax = plt.gca()

    if "par" in constraint_name:
        color = (255/255, 144/255, 30/255)
        p1, p2 = construct_image_midline()
        p1 = p1.astype(int)
        p2 = p2.astype(int)

        ax.scatter([p1[0], p2[0], center[0], component_1[0]], [p1[1], p2[1], center[1], component_1[1]], color=color, marker='.', s=100, edgecolor=color, linewidth=1.25, label="Parallel-line")
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color)
        ax.plot([center[0], component_1[0]], [center[1], component_1[1]], color=color)

    elif "p2p" in constraint_name:
        color = (50/255,205/255,50/255)
        p1 = get_in_hand_camera_center_point().astype(int)
        x2, y2 = int(center[0]), int(center[1])
        ax.scatter([x2, p1[0]], [y2, p1[1]], color=color, marker='.', s=100, edgecolor=color, linewidth=1.25, label="Point-to-point")

    elif "p2l" in constraint_name:
        color = (0,1,1)
        p1 = get_in_hand_camera_center_point().astype(int)

        ax.scatter([center[0], p1[0], component_1[0]], [center[1], p1[1], component_1[1]], color=color, marker='.', s=100, edgecolor=color, linewidth=1.25, label="Point-to-line")
        ax.plot([center[0], component_1[0]], [center[1], component_1[1]], color=color)

    elif "l2l" in constraint_name:
        color = (168/255, 50/255, 111/255)
        p1, p2 = construct_image_midline()
        p1 = p1.astype(int)
        p2 = p2.astype(int)

        ax.scatter([p1[0], p2[0], center[0], component_1[0]], [p1[1], p2[1], center[1], component_1[1]], color=color, marker='.', s=100, edgecolor=color, linewidth=1.25, label="Parallel-line")
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color)
        ax.plot([center[0], component_1[0]], [center[1], component_1[1]], color=color)

def main(
    imgdir,
    ):
    if imgdir[-1] == os.sep:
        imgdir = imgdir[:-1]
    dataset_path = os.path.join(*imgdir.split(os.sep)[:-1])
    video_name = imgdir.split(os.sep)[-1]

    # Load Part-of-object constraints, we will use this to annotate videoe
    annot_vid = load_part_of_object_annot_v2(dataset_path, video_name)

    for img_path, annot_frame in annot_vid.items():
        print("-"*10)
        print(img_path)
        print(annot_frame)

        img = cv2.imread(img_path)
        vis = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        constraints_to_text = []
        plt.imshow(vis)

        for constraint in annot_frame:
            constraint_type = constraint["type"]
            parts = constraint["parts"]
            mask = np.zeros(img.shape[:2], np.uint8)

            for part in parts:
                part_path = img_path.replace("imgs", "annots{}{}".format(os.sep, part))
                mask_part = cv2.imread(part_path, 0)
                mask = cv2.bitwise_or(mask, mask_part)
            
            visualize_constraint(vis, mask, constraint_type)

        plt.legend(fontsize="16")
        plt.axis('off')
        # plt.savefig('out.png', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.pause(0.1)
        # plt.show()
        plt.clf()

    return


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Quickly visualize data annotation.')
    parser.add_argument('imgdir', help='Image directory.')
    args = parser.parse_args()

    main(args.imgdir)