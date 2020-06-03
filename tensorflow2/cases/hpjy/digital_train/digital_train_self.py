# encoding: utf-8

"""
@author: sunxianpeng
@file: digital_train_res50.py
@time: 2020/6/3 13:19
"""
import time

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten,GlobalAveragePooling2D,Conv2D,MaxPooling2D,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array,array_to_img
from tensorflow.keras.optimizers import SGD,Adagrad,RMSprop
from tensorflow.keras import callbacks
import os
import cv2
import numpy as np
import tensorflow as tf

from digital_train import get_this_file_name, get_logger, check_device
##############################################################
# Log
##############################################################


log_path = 'logs/'
today_datetime = time.strftime("%Y%m%d%H%M%S", time.localtime())
log_name = log_path + 'logs_' + str(today_datetime) + '.log'
log_file = log_name
this_file_name = get_this_file_name()
logger = get_logger(log_file, this_file_name)
logger.info('start')

check_device()

def lap_filter(img, k_max=6):
    if np.sum(img) < 0.1:
        return img_to_array(img)
    kernel_lap = np.array([[0, -1, 0],
                           [-1, k_max, -1],
                           [0, -1, 0]])
    f1 = cv2.filter2D(img, -1, kernel_lap)
    fz = 254
    f1[f1 < fz] = 0
    f1[f1 >= fz] = 1
    return img_to_array(f1)

def get_model():
    model = Sequential()  # 采用贯序模型

    # model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(ishape, ishape, 1)))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    # model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    # model.add(Flatten())
    # model.add(Dense(2048, activation='relu'))
    # model.add(Dropout(0.25))
    # model.add(Dense(2048, activation='relu'))
    # model.add(Dropout(0.25))
    # model.add(Dense(13, activation='softmax'))

    weight_decay = 0.01

    model.add(Conv2D(64, (3, 3), input_shape=(ishape, ishape, 1), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(2304, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(2304, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))

    print(model.summary())
    return model

train_datagen = ImageDataGenerator(
    # preprocessing_function=lap_filter,# ((x/255)-0.5)*2  归一化到±1之间
    # rescale=1./255,
    # rotation_range=10,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    zoom_range=0.2,
    # fill_mode='constant',
    # cval=0
)
val_datagen = ImageDataGenerator(
    # preprocessing_function=lap_filter,# ((x/255)-0.5)*2  归一化到±1之间
    # rescale=1./255,
    # rotation_range=10,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    zoom_range=0.2,
    #     fill_mode='constant',
    #     cval=0
)
ishape = 28
c2 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
train_generator = train_datagen.flow_from_directory(directory='./train',
                                                    target_size=(ishape, ishape),
                                                    batch_size=32,
                                                    classes=c2,
                                                    color_mode='grayscale'
                                                    )

val_generator = val_datagen.flow_from_directory(directory='./val',
                                                target_size=(ishape, ishape),
                                                batch_size=32,
                                                classes=c2,
                                                color_mode='grayscale',
                                                # save_to_dir='./img_create'
                                                )

print(train_generator.class_indices)

model = get_model()
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# optimizer = RMSprop(lr = 0.001, decay=0.0)

# model.compile(optimizer=optimizer, loss = 'categorical_crossentropy',metrics=['accuracy'])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# early_stop = callbacks.EarlyStopping(monitor='val_loss',
#                                            min_delta=0,
#                                            patience=3,
#                                            verbose=1, mode='auto')

history_tl = model.fit_generator(generator=train_generator,
                    #steps_per_epoch=100,#800
                    epochs=2,#2
                    validation_data=val_generator,
                    #validation_steps=12,#12
                    class_weight='auto'
                    )
model.save("model/model_self.h5")