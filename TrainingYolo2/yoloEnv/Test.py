from darkflow.net.build import TFNet
import cv2
import tensorflow as tf

config = tf.ConfigProto(log_device_placement = False)
config.gpu_options.allow_growth = False

with tf.Session(config=config) as sess:
  options = {'model':'cfg/yolo-train.cfg',
          'load':-1, 
          #'epoch':10,
          #'train':True,
          #'annotation': '/Users/radualbastroiu/Documents/My_projects/Licenta/ML-Object-Recognition/data/CarDownloadImagesLabeled/annotations/',
          #'annotation': '/Users/radualbastroiu/Documents/My_projects/Licenta/ML-Object-Recognition/data/CarDownloadImagesLabeled/images/'
          'threshold':0.5
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


img = cv2.imread('/Users/radualbastroiu/Documents/My_projects/Licenta/ML-Object-Recognition/data/soccer_ball_data/images/scene00681.png')

imgConv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = tfnet.return_predict(imgConv)
print(results)

cv2.imshow("object detection yolo", displayResults(results, img))
cv2.waitKey(0)
cv2.destroyAllWindows()

