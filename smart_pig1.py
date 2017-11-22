import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Reshape
from keras.utils import np_utils
import scipy
import tensorflow
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
import os
from keras.layers import  BatchNormalization

np.random.seed(123)
x, y = [], []
val_x, Val_y = [] ,[]
img_width, img_height = 64, 64


train_data_dir = './data3/preview'
validation_data_dir = './data3/validation'
nb_train_samples = 3119
nb_validation_samples = 550
epochs = 50
batch_size = 32


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# create model
model = Sequential()
model.add(Conv2D(64, (3, 3), strides=1, padding='valid',input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.75))
#model.add(BatchNormalization())


model.add(Conv2D(32, (3, 3), strides=1, padding='valid'))
model.add(Activation('relu'))
# model.add(Reshape(target_shape=[-1, 4]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#model.add(BatchNormalization())

# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#model.add(BatchNormalization())
model.add(Dense(5))
model.add(Activation('softmax'))
# model.add(Dropout(0.5))
# model.add(BatchNormalization())

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer= 'adadelta',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

print(model.summary())

# Fit the model
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

# save serialize model to JSON
model_json = model.to_json()
with open("smart_pig_model_2.json", "w" ) as json_file:
    json_file.write(model_json)

# save weight to HDF5
model.save_weights('sp_try2.h5')

# evaluate the model
scores = model.evaluate_generator(validation_generator, nb_validation_samples/batch_size, workers=12)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
