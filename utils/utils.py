
__author__ = 'Majd Jamal'

import numpy as np
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def intersection_over_union(boxes_preds, boxes_labels):
    """ Compuites Intersection Over Union for two bounding boxes.
    :params boxes_preds: Predicted bounding box, shape = (Npts, S, S, 4)
    :params boxes_labels: Predicted bounding box, shape = (Npts, S, S, 4)
    :return: intersection over union value
    #Inspiration from https://youtu.be/n9_XyCGr-MI
    """
    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    x1 = tf.maximum(box1_x1, box2_x1)
    y1 = tf.maximum(box1_y1, box2_y1)
    x2 = tf.minimum(box1_x2, box2_x2)
    y2 = tf.minimum(box1_y2, box2_y2)

    t_x = tf.cast((x2 - x1), 'float32')
    t_y = tf.cast((y2 - y1), 'float32')

    intersection = tf.clip_by_value(t_x, -1, 1) * tf.clip_by_value(t_y, -1, 1)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def get_image(x):
    """ Read and return image in a format ready for prediction.
    """
    dim = (448, 448)

    #read
    x = cv2.imread(x)
    #y = cv2.imread(y)

    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    #y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)

    #re-size
    x = cv2.resize(x, dim, interpolation = cv2.INTER_AREA) / 255

    #x /= 255

    x = x.reshape((1,448,448,3))

    x = tf.cast(x, tf.float32)

    return x

def plot_image(im, box):
    """Plots predicted bounding boxes on the image"""
    #Inspiration from https://youtu.be/n9_XyCGr-MI
    
    im = im[0]

    height, width, _ = im.shape

    fig, ax = plt.subplots(1)

    ax.imshow(im)

    upper_left_x = box[0] - box[2] / 2
    upper_left_y = box[1] - box[3] / 2
    rect = patches.Rectangle(
    (upper_left_x * width, upper_left_y * height),
    box[2] * width,
    box[3] * height,
    linewidth=2,
    edgecolor="r",
    facecolor="none",
    )

    ax.add_patch(rect)

    plt.show()
