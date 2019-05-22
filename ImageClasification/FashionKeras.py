import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard

# import data
train = pd.read_csv(r'../data/FashionMnist/fashion-mnist_train.csv')
test = pd.read_csv(r'../data/FashionMnist/fashion-mnist_test.csv')

train_data = np.array(train, dtype='float32')
test_data = np.array(test, dtype='float32')

x_train = train_data[:, 1:] / 255
y_train = train_data[:, 0]

x_test = test_data[:, 1:] / 255
y_test = test_data[:, 0]

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=2019)

# reshape data
im_rows = 28
im_cols = 28
batch_size = 512
im_shape = (im_rows, im_cols, 1)

x_train = x_train.reshape(x_train.shape[0], *im_shape)
x_val = x_val.reshape(x_val.shape[0], *im_shape)
x_test = x_test.reshape(x_test.shape[0], *im_shape)

print('x_train shape: {}'.format(x_train.shape))
print('x_val shape: {}'.format(x_val.shape))
print('x_test shape: {}'.format(x_test.shape))

# model
cnn_model = Sequential()
cnn_model.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=im_shape))
#cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=2))
cnn_model.add(Dropout(0.3))
cnn_model.add(Flatten())
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dense(10, activation='softmax'))

tensorboard = TensorBoard(log_dir=r'../data/FashionMnist/logs/{}'.format('cnn_1layer'), write_graph=True, write_grads=True, histogram_freq=1, write_images=True)

cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# train 
cnn_model.fit(x_train, y_train, batch_size=batch_size, epochs=2, verbose=1, validation_data=(x_val, y_val), callbacks=[tensorboard])

