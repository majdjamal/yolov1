
__author__ = 'Majd Jamal'

import numpy as np
from utils.utils import get_image, plot_image

def Detect(yolo):
    """ Detect object in an image.
    :params yolo: YoloV1 Architecture
    """
    img_path = 'data/klopp_imgs/X0.jpg' #Modify this path to your custom image.
    img = get_image(img_path)
    pred = yolo(img)

    max_val = 0
    box = None
    pos = ''
    boxes = []
    idx = []
    s = 7

    for i in range(7):
        for j in range(7):

            p1 = pred[0, i, j, 2]

            p2 = pred[0, i, j, 7]

            if p1 > max_val:

                x, y, w, h  = pred[0, i, j, 3:7]

                x, y, w, h = x.numpy(), y.numpy(), w.numpy(), h.numpy()

                x = (x + j) / s
                y = (y + i) / s

                w = (w / s)

                h = (h / s)

                boxes.append( (x, y, w, h) )
                max_val = p1

            if p2 > max_val:

                x, y, w, h  = pred[0, i, j, 8:]

                x, y, w, h = x.numpy(), y.numpy(), w.numpy(), h.numpy()

                x = (x + j) / s
                y = (y + i) / s

                w = (w / s)

                h = (h / s)

                boxes.append( (x, y, w, h) )
                max_val = p2

    plot_image(img, np.abs(boxes[-1]))
