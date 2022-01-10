
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset

def getData(batch_size = 16):
    """ Load and returns the dataset for training
    :params batch size: Number of data point for each training step.
    :return train_dataset: Dataset in Tensorflow format
    """
    X = np.load('data/X.npy')
    Y = np.load('data/Y.npy')

    X = tf.cast(X, tf.float32)
    Y = tf.cast(Y, tf.float32)

    train_dataset = Dataset.from_tensor_slices((X, Y))
    train_dataset.shuffle(400)
    train_dataset = train_dataset.batch(batch_size)

    return train_dataset
