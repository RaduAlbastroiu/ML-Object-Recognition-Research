import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
sess = tf.Session()
set_session(sess)

from keras.datasets import mnist

(x_train_, y_train), (x_test_, y_test) = mnist.load_data()

input_dim = 784 # 28*28
x_train = x_train_.reshape(60000, input_dim)
x_test = x_test_.reshape(10000, input_dim)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

from keras.utils import np_utils
n_classes = len(set(y_train))
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)

from keras.models import Sequential
from keras.layers import Dense, Activation

output_dim = n_classes = 10
model = Sequential()
model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))
batch_size = 128
n_epoch = 20

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch, verbose=1, validation_data=(x_test, y_test))
model.summary()

score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
