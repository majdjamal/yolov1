
__author__ = 'Majd Jamal'

import numpy as np
import matplotlib.pyplot as plt
from model.yolov1 import YoloV1
from utils.params import args

if __name__ == "__main__":

    yolo_object = YoloV1()
    yolo = yolo_object.MakeYOLO()

    if args.train:
        from train.train import TrainYolo
        from data.getData import getData

        train_ds = getData()
        solver = TrainYolo(yolo)
        solver.fit(train_ds)

        yolo = solver.getModel()

    elif args.predict:
        from tensorflow.keras.models import load_model
        from predict.predict import Detect

        yolo_path = 'data/checkpoints/yolo.h5'
        yolo = load_model(yolo_path)
        Detect(yolo)



    #if params.predict:
    #    continue
