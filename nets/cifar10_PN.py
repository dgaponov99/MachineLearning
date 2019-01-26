from __future__ import print_function
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt

NB_EPOCH = 400
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
OPTIMIZER = SGD()
VALIDATION_SPLIT = 0.2

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
RESHAPED = 3072
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train_line = []
for l in range(len(x_train)):
    x_train_line.append([])
    for i in range(32):
        for j in range(32):
            for k in range(3):
                x_train_line[l].append(x_train[l][i][j][k])

x_train = np.array(x_train_line, np.float32)
print('x_train created')

x_test_line = []
for l in range(len(x_test)):
    x_test_line.append([])
    for i in range(32):
        for j in range(32):
            for k in range(3):
                x_test_line[l].append(x_test[l][i][j][k])

x_test = np.array(x_test_line, np.float32)
print('x_test created')

y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# Сеть
model = Sequential()
model.add(Dense(NB_CLASSES, input_shape=(RESHAPED,)))
model.add(Activation('sigmoid'))
model.summary()

# Компиляция
model.compile(loss='mean_squared_error',
              optimizer=OPTIMIZER,
              metrics=['accuracy'])

# Обучение
history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=NB_EPOCH,
                    verbose=VERBOSE,
                    validation_split=VALIDATION_SPLIT)

score = model.evaluate(x_test, y_test,
                       verbose=VERBOSE)
print('Test accuracy: ', score[1])

# Сохранение
model_json = model.to_json()
open('cifar10_PN.json', 'w').write(model_json)
model.save_weights('cifar10_weights_PN.h5', overwrite=True)

# График изменения точности
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# построить график изменения потери
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
