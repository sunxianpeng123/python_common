# encoding: utf-8

"""
@author: sunxianpeng
@file: blood_train_self.py
@time: 2020/6/4 15:53
"""

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D, \
    BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.optimizers import SGD, Adagrad, RMSprop
from tensorflow.keras import callbacks
import os
import cv2
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "3"


def lap_filter(img, k_max=6):
    if np.sum(img) < 0.1:
        return img_to_array(img)
    kernel_lap = np.array([[0, -1, 0],
                           [-1, k_max, -1],
                           [0, -1, 0]])
    f1 = cv2.filter2D(img, -1, kernel_lap)
    fz = 250
    f1[f1 < fz] = 0
    f1[f1 >= fz] = 1
    return img_to_array(f1)


# train_datagen = ImageDataGenerator(
#     #preprocessing_function=lap_filter,# ((x/255)-0.5)*2  归一化到±1之间
#     #rescale=1./255,
#     rotation_range=10,
#     width_shift_range=0.3,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=[1,4],
#     fill_mode='constant',
#     cval=0
# )
# val_datagen = ImageDataGenerator(
#     #preprocessing_function=lap_filter,# ((x/255)-0.5)*2  归一化到±1之间
#     #rescale=1./255,
#     rotation_range=10,
#     width_shift_range=0.3,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=[1,4],
#     fill_mode='constant',
#     cval=0
# )
train_datagen = ImageDataGenerator(
    # preprocessing_function=lap_filter,# ((x/255)-0.5)*2  归一化到±1之间
    # rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2
    #    shear_range=0.2,
    # zoom_range=[1,4],
    # fill_mode='constant',
    # cval=255
)
val_datagen = ImageDataGenerator(
    # preprocessing_function=lap_filter,# ((x/255)-0.5)*2  归一化到±1之间
    # rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2
    # shear_range=0.2,
    # zoom_range=[1,4],
    # fill_mode='constant',
    # cval=255
)
# img = load_img('./val/3/10672487_150.jpg')  # this is a PIL image
# x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
# x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
# print(x.shape)
# i = 0
# for batch in train_datagen.flow(x, batch_size=1,
#                           save_to_dir='create_data', save_prefix='four', save_format='jpeg'):
#     i += 1
#     if i > 50:
#         break  # otherwise the generator would loop indefinitely

ishape = 32
train_generator = train_datagen.flow_from_directory(directory='./train',
                                                    target_size=(ishape, ishape),
                                                    batch_size=32,
                                                    color_mode='grayscale'
                                                    )

val_generator = val_datagen.flow_from_directory(directory='./val',
                                                target_size=(ishape, ishape),
                                                batch_size=32,
                                                color_mode='grayscale',
                                                # save_to_dir='./img_create'
                                                )
print(train_generator.class_indices)


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
    model.add(Dense(2, activation='softmax'))
    print(model.summary())
    return model


model = get_model()
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# optimizer = RMSprop(lr = 0.001, decay=0.0)

# model.compile(optimizer=optimizer, loss = 'categorical_crossentropy',metrics=['accuracy'])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# early_stop = callbacks.EarlyStopping(monitor='val_loss',
#                                            min_delta=0,
#                                            patience=3,
#                                            verbose=1, mode='auto')

# history_tl = model.fit_generator(generator=train_generator,
#                     #steps_per_epoch=100,#800
#                     epochs=2,#2
#                     validation_data=val_generator,
#                     #validation_steps=12,#12
#                     class_weight='auto'
#                     )
# model.save("model/model_self.h5")


history_tl = model.fit_generator(generator=train_generator,
                                 # steps_per_epoch=300,#800
                                 epochs=1,  # 2
                                 validation_data=val_generator,
                                 # validation_steps=12,#12
                                 class_weight='auto'
                                 )
model.save("model/model_tiny_final.h5")
