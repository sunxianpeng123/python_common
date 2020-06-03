# encoding: utf-8

"""
@author: sunxianpeng
@file: digital_train_res50.py
@time: 2020/6/3 13:19
"""
import tensorflow as tf
from keras import Model
from keras.applications import ResNet50
from keras.layers import Flatten, Dense
from keras_preprocessing.image import ImageDataGenerator

from digital_train import check_device, get_this_file_name, get_logger
import time

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
##############################################################
# train
##############################################################

train_datagen = ImageDataGenerator(
    # rescale=1./255,#归一化
    # horizontal_flip=True,#水平反转
    # rotation_range=30,#旋转范围
    width_shift_range=0.2,  # 水平平移范围
    height_shift_range=0.2,  # 垂直平移范围
    zoom_range=0.2  # 缩放范围
    # shear_range=0.2,#透视变换的范围
)
val_datagen = ImageDataGenerator(
    # preprocessing_function=preprocess_input,
    # rescale=1./255
    # horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2
)
ishape = 32
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

model_res = ResNet50(include_top=False, weights=None, input_shape=(ishape, ishape, 1))
model = Flatten()(model_res.output)
model = Dense(10, activation='softmax', name='prediction')(model)
model = Model(model_res.input, model, name='res50_pretrain')
print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_tl = model.fit_generator(generator=train_generator,
                                 # steps_per_epoch=800,#800
                                 epochs=1,  # 2
                                 validation_data=val_generator,
                                 # validation_steps=12,#12
                                 class_weight='auto'
                                 )
model.save("model/model_res.h5")
# krs.models.save_model(model_vgg_mnist_pretrain, "model.h5")
print("save done")
