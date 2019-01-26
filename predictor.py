import numpy as np
import scipy.misc
from keras.models import model_from_json
from keras.optimizers import SGD

# загрузить модель
model_architecture = 'nets_and_weights/cifar10_DCNN.json'
model_weights = 'nets_and_weights/cifar10_weights_DCNN.h5'
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)
# загрузить изображения
img_names = ['airs/air0r.jpg', 'airs/air1_4.jpg', 'airs/air2_5.jpg', 'airs/air3r.jpg', 'airs/air4r.jpg',
             'dogs/dog0_1.jpg', 'dogs/dog1r.jpg', 'dogs/dog2_6.jpg', 'dogs/dog3r.jpg', 'dogs/dog4r.jpg']
imgs = [np.transpose(scipy.misc.imresize(scipy.misc.imread(img_name), (32, 32)),
                     (1, 0, 2)).astype('float32') for img_name in img_names]
imgs = np.array(imgs) / 255
# обучить
optim = SGD()
model.compile(loss='mean_squared_error', optimizer=optim,
              metrics=['accuracy'])
# предсказать
predictions = model.predict_classes(imgs)
print(predictions)
