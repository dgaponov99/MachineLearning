from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

# набор CIFAR_10 содержит 60K изображений 32x32 с 3 каналами
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

# константы
NB_EPOCH = 25
BATCH_SIZE = 128
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()

# загрузить набор данных
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('X_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# преобразовать к категориальному виду
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# преобразовать к формату с плавающей точкой и нормировать
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Сеть
model = Sequential()
model.add(Conv2D(32,
                 (3, 3),
                 padding='valid',
                 input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NB_CLASSES, activation='sigmoid'))
model.summary()

# Компиляция сети
model.compile(loss='mean_squared_error',
              optimizer=OPTIM,
              metrics=['accuracy'])
# Обучение
history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=NB_EPOCH,
                    validation_split=VALIDATION_SPLIT,
                    verbose=VERBOSE)

# Тестирование
score = model.evaluate(x_test, y_test,
                       batch_size=BATCH_SIZE,
                       verbose=VERBOSE)

print('Test accuracy: ', score[1])

# Сохранение модели
model_json = model.to_json()
open('cifar10_CNN.json', 'w').write(model_json)

# Сохранение весов
model.save_weights('cifar10_weights_CNN.h5', overwrite=True)

# График точности
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# График потери
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
