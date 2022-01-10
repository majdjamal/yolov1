
__author__ = 'Majd Jamal'

from loss.loss import yolo_loss
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from colorama import Fore
from tensorflow.keras import optimizers

class TrainYolo:

    def __init__(self, model):
        self.yolo = model
        self.YOLOOptimizer = optimizers.Adam(2e-5, beta_1 = 0.5)

    @tf.function
    def train_step(self, inp, tar, step) -> None:
        """ Updates network weights.
        :params inp: Input data point, shape = (Batch_size, 448, 448, 3)
        :params tar: Target values, shape = (Batch_size, 448, 448, 3)
        """

        with tf.GradientTape() as yolo_tape:

            yolo_output = self.yolo(inp, training = True)

            total_loss = yolo_loss(yolo_output, tar)

        yolo_gradients = yolo_tape.gradient(
            total_loss, self.yolo.trainable_variables)

        self.YOLOOptimizer.apply_gradients(
            zip(yolo_gradients, self.yolo.trainable_variables)
        )

    def getModel(self):
        """ Return model
        """
        return self.yolo

    def fit(self, train_dataset, steps = 20000 * 1):
        """This function trains the network
        :params model: YoloV1 Architecture
        :params train_dataset: Training data points in Tensorflow format
        :params steps: Number of iterations in the training process. Scalar,
        default is set as 20 000 * 1, which is approx. 1 hour of training.
        """

        with tqdm(train_dataset.repeat().take(steps).enumerate(), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)) as prg:
            milestone = 1000

            for step, (input_image, target) in prg:
                #i = tf.cast(i, tf.int64)
                yolo = self.train_step(input_image, target, step)

                if step % 10 == 0:
                    pred = self.yolo(input_image, training = True)
                    total_loss = yolo_loss(pred, target)
                    num_loss = total_loss.numpy()
                    prg.set_postfix(loss=num_loss)

                    if num_loss < milestone:

                        self.yolo.save('data/checkpoints/yolov1.h5')

                        milestone = num_loss/10

        self.yolo.save('data/checkpoints/yolov1_FINAL.h5')
