# Load libraries

import tensorflow as tf
import numpy as np

# shortcut to layers and models provided in tf
layers = tf.contrib.keras.layers
models = tf.contrib.keras.models

def fire_mod(x, fire_id, squeeze=16, expand=64):
    
    # initalize naming convention of components of the fire module
    squeeze1x1 = 'squeeze1x1'
    expand1x1 = 'expand1x1'
    expand3x3 = 'expand3x3'
    relu = 'relu.'
    fid = 'fire' + str(fire_id) + '/'
    
    # define the squeeze layer ~ (1,1) filter
    x = layers.Convolution2D(squeeze, (1,1), padding = 'same', name= fid + squeeze1x1)(x)
    x = layers.Activation('relu', name= fid + relu + squeeze1x1)(x)
    
    # define the expand layer's (1,1) filters
    expand_1x1 = layers.Convolution2D(expand, (1,1), padding='same', name= fid + expand1x1)(x)
    expand_1x1 = layers.Activation('relu', name= fid + relu + expand1x1)(expand_1x1)
    
    # define the expand layer's (3,3) filters
    expand_3x3 = layers.Convolution2D(expand, (3,3), padding='same', name= fid + expand3x3)(x)
    expand_3x3 = layers.Activation('relu', name= fid + relu + expand3x3)(expand_3x3)
    
    # Concatenate
    x = layers.concatenate([expand_1x1, expand_3x3], axis = 3, name = fid + 'concat')
    
    return x

def SqueezeNetMini(input_shape = (32,32,3), classes = 10):
        
    img_input = layers.Input(shape=input_shape)
    
    x = layers.Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv')(img_input)
    x = layers.Activation('relu', name='relu_conv1')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_mod(x, fire_id=2, squeeze=16, expand=64)
    x = fire_mod(x, fire_id=3, squeeze=16, expand=64)

    x = fire_mod(x, fire_id=4, squeeze=32, expand=128)
    x = fire_mod(x, fire_id=5, squeeze=32, expand=128)
    x = layers.Dropout(0.5, name='drop9')(x)

    x = layers.Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)
    x = layers.Activation('relu', name='relu_conv10')(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Activation('softmax', name='loss')(x)

    model = models.Model(img_input, out, name='squeezenetmini')

    return model

def SqueezeNet(input_shape = (32,32,3), classes = 10):
        
    img_input = layers.Input(shape=input_shape)
    
    x = layers.Convolution2D(64, (3, 3), strides=(2, 2), name='conv')(img_input)
    x = layers.Activation('relu', name='relu_conv1')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1', padding='same')(x)

    x = fire_mod(x, fire_id=2, squeeze=16, expand=64)
    x = fire_mod(x, fire_id=3, squeeze=16, expand=64)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3', padding='same')(x)

    x = fire_mod(x, fire_id=4, squeeze=32, expand=128)
    x = fire_mod(x, fire_id=5, squeeze=32, expand=128)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5', padding='same')(x)
    
    x = fire_mod(x, fire_id=6, squeeze=48, expand=192)
    x = fire_mod(x, fire_id=7, squeeze=48, expand=192)
    x = fire_mod(x, fire_id=8, squeeze=64, expand=256)
    x = fire_mod(x, fire_id=9, squeeze=64, expand=256)
    
    x = layers.Dropout(0.5, name='drop9')(x)

    x = layers.Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)
    x = layers.Activation('relu', name='relu_conv10')(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Activation('softmax', name='loss')(x)

    model = models.Model(img_input, out, name='squeezenet')

model = SqueezeNetMini()
print(model.summary())