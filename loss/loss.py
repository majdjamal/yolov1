
__author__ = 'Majd Jamal'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from utils.utils import intersection_over_union

def yolo_loss(out, tar, lmd_coord = 5, lmd_noobj = 0.5):
    """ Computes the loss function intended for YoloV1.
    :params out: Predicted values, such as bounding box and class probability.
    :params tar: Target values
    :params lmd_coord: Penalty constant for coordinates loss
    :params lmd_noobj: Penalty constant for non objects loss
    :return loss: Scalar, loss value
    #Inspiration from https://youtu.be/n9_XyCGr-MI
    """

    mse = lambda a, b: tf.math.reduce_sum(tf.math.square(tf.math.subtract(a,b)))


    pc_1 = out[...,2]
    bb1 = out[...,3:7]

    pc_2 = out[...,7]
    bb2 = out[...,8:]

    tar_bb = tar[..., 3:]

    iou_bb1 = intersection_over_union(bb1, tar_bb) # 1, 7, 7, 1
    iou_bb2 = intersection_over_union(bb2, tar_bb) # 1, 7, 7, 1

    iou_bb1 = tf.expand_dims(iou_bb1, 0)

    iou_bb2 = tf.expand_dims(iou_bb2, 0)
    ious = tf.concat((iou_bb1, iou_bb2), 0)

    best_box = tf.cast(tf.math.argmax(ious, axis=0), 'float32') # Most accurate bounding box

    exist_box = tf.cast(tf.expand_dims(tar[..., 2],3), 'float32')

    ##############
    ## Box Loss
    ##############

    box_predictions = exist_box * ( best_box * bb2 + (1 - best_box) * bb1 )

    box_targets = exist_box * tar_bb

    box_predictions_processed = tf.math.sqrt(
    	tf.math.abs(box_predictions[..., 2:4]) + 1e-6)

    box_predictions = tf.concat((box_predictions[..., :2], box_predictions_processed), -1)

    box_targets_processed = tf.math.sqrt(
    	tf.math.abs(box_targets[..., 2:4]) + 1e-6)

    box_targets = tf.concat((box_targets[..., :2], box_targets_processed), -1)


    N, S, S, pos = box_predictions.shape

    box_predictions = tf.reshape(box_predictions, (N*S*S, pos))

    box_targets = tf.reshape(box_targets, (N*S*S, pos))


    box_loss = mse(box_predictions, box_targets)

    ##############
    ## Object Loss
    ##############

    pred_box = (best_box * tf.expand_dims(pc_2, -1) + (1 - best_box) * tf.expand_dims(pc_1, -1))

    obj_loss_pred = pred_box * exist_box
    obj_loss_pred = tf.transpose(layers.Flatten()(obj_loss_pred))[:, 0]

    obj_loss_tar = exist_box * tf.expand_dims(tar[..., 2], -1)
    obj_loss_tar = tf.transpose(layers.Flatten()(obj_loss_tar))[:, 0]

    obj_loss = mse(obj_loss_pred, obj_loss_tar)


    ##################
    ## No Object Loss
    ##################

    tar_pc = tf.expand_dims(tar[..., 2], -1)

    no_obj_loss_pred = (1 - exist_box) * tf.expand_dims(pc_1, -1)
    no_obj_loss_pred = tf.transpose(layers.Flatten()(no_obj_loss_pred))[:, 0]

    no_obj_loss_tar = (1 - exist_box) * tar_pc
    no_obj_loss_tar = tf.transpose(layers.Flatten()(no_obj_loss_tar))[:, 0]

    no_obj_loss_first = mse(no_obj_loss_pred, no_obj_loss_tar)

    no_obj_loss_pred = (1 - exist_box) * tf.expand_dims(pc_2, -1)
    no_obj_loss_pred = tf.transpose(layers.Flatten()(no_obj_loss_pred))[:, 0]

    no_obj_loss_tar = (1 - exist_box) * tar_pc
    no_obj_loss_tar = tf.transpose(layers.Flatten()(no_obj_loss_tar))[:, 0]

    no_obj_loss_second = mse(no_obj_loss_pred, no_obj_loss_tar)

    no_obj_loss = no_obj_loss_first + no_obj_loss_second

    ##################
    ## Class Loss
    ##################

    classes = out[..., :2]

    N, S, S, C = classes.shape

    class_loss_pred = exist_box * classes
    class_loss_pred = tf.reshape(class_loss_pred, (N*S*S, C))

    class_loss_tar = exist_box * tar[..., :2]
    class_loss_tar = tf.reshape(class_loss_tar, (N*S*S, C))

    class_loss = mse(class_loss_pred, class_loss_tar)

    ##################
    ## Total Loss
    ##################

    loss = lmd_coord * box_loss + obj_loss + lmd_noobj * no_obj_loss + class_loss

    return loss
