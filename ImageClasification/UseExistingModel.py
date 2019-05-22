import tensorflow as tf
import matplotlib.pyplot as plt
from keras.backend.tensorflow_backend import set_session
# gpu setup
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.log_device_placement = True
#sess = tf.Session(config=config)
#set_session(sess)

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet')

img_path = '../data/Images/dog.jpg'
img = image.load_img(img_path, target_size=(224, 224))

plt.figure(figsize=(8,8))
plt.imshow(img)
plt.axis('off')
plt.show()

img = image.img_to_array(img)
img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
img = preprocess_input(img)

pred = model.predict(img)
print('Predicted:', decode_predictions(pred, top=3)[0])
