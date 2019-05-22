import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import numpy as np
from matplotlib import pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.datasets import mnist


(X_train_, y_train_), (X_test_, y_test_) = mnist.load_data()

# Reshape from (60000, 28, 28) => (60000, 28, 28, 1)
img_rows, img_cols = X_train_[0].shape[0], X_train_[0].shape[1]
x_train = X_train_.reshape(X_train_.shape[0], img_rows, img_cols, 1)
x_test = X_test_.reshape(X_test_.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train_std = x_train.astype('float32')/255.
x_test_std = x_train.astype('float32')/255.

n_classes = len(set(y_train_))
y_train = to_categorical(y_train_, n_classes)
y_test = to_categorical(y_test_, n_classes)


# Creating a model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 128
n_epochs = 1

# train 
model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs)

# predict
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy', score[1])


# show clasified images
preds = model.predict(x_test)
n = 10
plt.figure(figsize=(15,15))
for i in range(n):
  plt.subplot(1, n, i + 1)
  plt.imshow(x_test[i, :, :, 0], cmap='gray')
  plt.title("Label: {}\nPredicted:{}".format(np.argmax(y_test[i]), np.argmax(preds[i])))
  plt.axis('off')
plt.show()

# show missclasified images
plt.figure(figsize=(15,15))
misc = 0
for i in range(len(y_test)):
  if(misc==10):
    break
  
  label = np.argmax(y_test[i])
  pred = np.argmax(preds[i])
  if(label != pred):
    plt.subplot(1, n, i + 1)
    plt.imshow(x_test[i, :, :, 0], cmap='gray')
    plt.title("Label: {}\nPredicted:{}".format(np.argmax(y_test[i]), np.argmax(preds[i])))
    plt.axis('off')
    misc += 1
plt.show()

