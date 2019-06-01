from darkflow.net.build import TFNet
import cv2
import tensorflow as tf
import sys 
import os
import numpy as np
import time

absFilePath = os.path.abspath(__file__)
darkflowDir = os.path.dirname(os.path.dirname(absFilePath))

yolov2Path = os.path.join(darkflowDir, 'yolov2.weights')
modelPath = os.path.join(darkflowDir, 'cfg/yolo.cfg')

# Config TF, set True if using GPU
config = tf.ConfigProto(log_device_placement = False)
config.gpu_options.allow_growth = False

with tf.Session(config = config) as sess:
  options = {'model' : 'cfg/yolo.cfg', 
            'load' : 'yolov2.weights',
            'threshold' : 0.3, 
            #'gpu' : 1.0 # uncomment if using GPU
  }
  tfnet = TFNet(options)


def displayResults(results, img):
  for(i, result) in enumerate(results):
    x = result['topleft']['x']
    w = result['bottomright']['x'] - result['topleft']['x']
    y = result['topleft']['y']
    h = result['bottomright']['y'] - result['topleft']['y']
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    label_position = (x + int(w/2)), abs(y-10)
    cv2.putText(img, result['label'], label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
  return img


# Running yolo on an image

def RunOnImage():
  img = cv2.imread('sample_img/sample_horses.jpg')
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  results = tfnet.return_predict(img)
  print(results)

  img = cv2.imread('sample_img/sample_horses.jpg')
  cv2.imshow("object detection yolo", displayResults(results, img))
  cv2.waitKey(0)
  cv2.destroyAllWindows()


# Running yolo on a webcam

def RunWebCam():
  capture = cv2.VideoCapture(0)

  while True:
    ret, frame = capture.read()

    if ret:
      results = tfnet.return_predict(frame)
      image = displayResults(results, frame)
      cv2.imshow('YoloV2', image)
      if cv2.waitKey(0) == 13:
        break

  capture.release()
  cv2.destroyAllWindows()



# Running yolo on a video

cap = cv2.VideoCapture('Shorter.mov')
frame_number = 0
while True:
  ret, frame = cap.read()
  frame_number += 1
  print("frame {}".format(frame_number))
  if ret:
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = tfnet.return_predict(img)

    for(i, result) in enumerate(results):
      x = result['topleft']['x']
      w = result['bottomright']['x'] - result['topleft']['x']
      y = result['topleft']['y']
      h = result['bottomright']['y'] - result['topleft']['y']
      cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
      label_position = (x + int(w/2)), abs(y-10)
      cv2.putText(frame, result['label'], label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    print("Show")
    cv2.imshow("ObjectDetection Yolo", frame)
      

cap.release()
cv2.destroyAllWindows()

