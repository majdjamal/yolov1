# You Only Look Once Version 1

YoloV1 implemented and trained from scratch using Tensorflow 2.0.

## Result
![ Alt text](data/pred.gif)

(Figure 1. Predicting Jürgen Klopp in a Press Conference with YoloV1. Comment: the predictions were good, considering that the network was trained from scratch, i.e., randomly initialized weights.)

## Test the program
* Download [weights](https://drive.google.com/file/d/1-2pqSLrakkt6-WlctZtvZhS9W3jvnjhF/view?usp=sharing) and place the .h5 file in data/checkpoints/

* Run,
```bash
  python3 main.py --predict
```

Inspiration from,
https://www.youtube.com/watch?v=n9_XyCGr-MI
https://www.youtube.com/watch?v=2hAiJe8ITsE
https://www.youtube.com/watch?v=ANIzQ5G-XPE
https://youtu.be/yo9S3eNtkXc
https://www.youtube.com/watch?v=m6VnxEooIeI
