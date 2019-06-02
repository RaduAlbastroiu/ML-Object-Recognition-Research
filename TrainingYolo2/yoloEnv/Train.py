from darkflow.net.build import TFNet
import cv2
import tensorflow as tf

config = tf.ConfigProto(log_device_placement = False)
config.gpu_options.allow_growth = False

with tf.Session(config=config) as sess:
  options = {'model':'cfg/yolo-train.cfg',
          'load':'yolov2-voc.weights', 
          'epoch':15000,
          'train':True,
          'annotation': '/Users/radualbastroiu/Documents/My_projects/Licenta/ML-Object-Recognition/data/partial_soccer_ball_data/annotations/',
          'dataset': '/Users/radualbastroiu/Documents/My_projects/Licenta/ML-Object-Recognition/data/partial_soccer_ball_data/images/',
          'gpu':1.0
          }
  tfnet = TFNet(options)

tfnet.train()
