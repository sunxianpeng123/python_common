# encoding: utf-8

"""
@author: sunxianpeng
@file: kill_train_res50.py
@time: 2020/6/4 10:53
"""

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from tensorflow.keras.layers import Dense, Add, Activation, Dropout, Flatten, AveragePooling2D, GlobalAveragePooling2D, \
    Input, ZeroPadding2D, BatchNormalization, Conv2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.optimizers import SGD, Adagrad
from tensorflow.keras.initializers import glorot_uniform
import os
import cv2
import numpy as np
import tensorflow as tf

##############################################################
# Log
##############################################################
import time

from blood_recognition import get_this_file_name, get_logger, check_device

log_path = 'logs/'
today_datetime = time.strftime("%Y%m%d%H%M%S", time.localtime())
log_name = log_path + 'logs_' + str(today_datetime) + '.log'
log_file = log_name
this_file_name = get_this_file_name()
logger = get_logger(log_file, this_file_name)
logger.info('start')

check_device()
##############################################################
# train
##############################################################
train_datagen = ImageDataGenerator(
    # rescale=1./255,
    # horizontal_flip=True,
    # rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
    # shear_range=0.2,
)
val_datagen = ImageDataGenerator(
    #  preprocessing_function=preprocess_input,
    # rescale=1./255
    # horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
# img = load_img('./val/4/1245.jpg')  # this is a PIL image
# x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
# x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
# i = 0
# for batch in val_datagen.flow(x, batch_size=1,
#                           save_to_dir='val_data', save_prefix='cat', save_format='jpeg'):
#     i += 1
#     if i > 20:
#         break  # otherwise the generator would loop indefinitely
image_shape = (32, 32, 3)
train_generator = train_datagen.flow_from_directory(directory='./train',
                                                    target_size=image_shape[0:2],
                                                    batch_size=32,
                                                    )
val_generator = val_datagen.flow_from_directory(directory='./val',
                                                target_size=image_shape[0:2],
                                                batch_size=32,
                                                )
# save_to_dir='./img_create' )
print(train_generator.class_indices)

model_res = ResNet50(include_top=False, weights=None, input_shape=(image_shape[0], image_shape[1], 3))
model = Flatten()(model_res.output)
model = Dense(2, activation='softmax', name='prediction')(model)
model = Model(model_res.input, model, name='res50_pretrain')
# model = Sequential()
# weight_decay = 0.01
# model.add(Conv2D(64, (3, 3), input_shape=image_shape, activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(128, (3, 3), activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
# model.add(Conv2D(128, (3, 3), activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
# #model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(256, (3, 3), activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
# model.add(Conv2D(256, (3, 3), activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
# #model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(2304, activation='relu'))
# #model.add(BatchNormalization())
# model.add(Dropout(0.25))
# model.add(Dense(2304, activation='relu'))
# #model.add(BatchNormalization())
# model.add(Dropout(0.25))
# model.add(Dense(2, activation='softmax'))
print(model.summary())
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history_tl = model.fit_generator(generator=train_generator,
                    #steps_per_epoch=800,#800
                    epochs=1,#2
                    validation_data=val_generator,
                    #validation_steps=12,#12
                    class_weight='auto'
                    )
model.save("model/kill_dead_model_res.h5")
#krs.models.save_model(model_vgg_mnist_pretrain, "model.h5")
print("save done")