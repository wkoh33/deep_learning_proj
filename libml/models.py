# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Classifier architectures."""
import functools
import itertools

import tensorflow as tf
from absl import flags

from libml import layers
from libml.train import ClassifySemi
from libml.utils import EasyDict


class CNN13(ClassifySemi):
    """Simplified reproduction of the Mean Teacher paper network. filters=128 in original implementation.
    Removed dropout, Gaussians, forked dense layers, basically all non-standard things."""

    def classifier(self, x, scales, filters, training, getter=None, **kwargs):
        del kwargs
        assert scales == 3  # Only specified for 32x32 inputs.
        conv_args = dict(kernel_size=3, activation=tf.nn.leaky_relu, padding='same')
        bn_args = dict(training=training, momentum=0.999)

        with tf.variable_scope('classify', reuse=tf.AUTO_REUSE, custom_getter=getter):
            y = tf.layers.conv2d((x - self.dataset.mean) / self.dataset.std, filters, **conv_args)
            y = tf.layers.batch_normalization(y, **bn_args)
            y = tf.layers.conv2d(y, filters, **conv_args)
            y = tf.layers.batch_normalization(y, **bn_args)
            y = tf.layers.conv2d(y, filters, **conv_args)
            y = tf.layers.batch_normalization(y, **bn_args)
            y = tf.layers.max_pooling2d(y, 2, 2)
            y = tf.layers.conv2d(y, 2 * filters, **conv_args)
            y = tf.layers.batch_normalization(y, **bn_args)
            y = tf.layers.conv2d(y, 2 * filters, **conv_args)
            y = tf.layers.batch_normalization(y, **bn_args)
            y = tf.layers.conv2d(y, 2 * filters, **conv_args)
            y = tf.layers.batch_normalization(y, **bn_args)
            y = tf.layers.max_pooling2d(y, 2, 2)
            y = tf.layers.conv2d(y, 4 * filters, kernel_size=3, activation=tf.nn.leaky_relu, padding='valid')
            y = tf.layers.batch_normalization(y, **bn_args)
            y = tf.layers.conv2d(y, 2 * filters, kernel_size=1, activation=tf.nn.leaky_relu, padding='same')
            y = tf.layers.batch_normalization(y, **bn_args)
            y = tf.layers.conv2d(y, 1 * filters, kernel_size=1, activation=tf.nn.leaky_relu, padding='same')
            y = tf.layers.batch_normalization(y, **bn_args)
            y = tf.reduce_mean(y, [1, 2])  # (b, 6, 6, 128) -> (b, 128)
            logits = tf.layers.dense(y, self.nclass)

            print(logits.shape, y.shape)
        return EasyDict(logits=logits, embeds=y)


class ResNet(ClassifySemi):
    def classifier(self, x, scales, filters, repeat, training, getter=None, dropout=0, **kwargs):
        del kwargs
        leaky_relu = functools.partial(tf.nn.leaky_relu, alpha=0.1)
        bn_args = dict(training=training, momentum=0.999)

        def conv_args(k, f):
            return dict(padding='same',
                        kernel_initializer=tf.random_normal_initializer(stddev=tf.rsqrt(0.5 * k * k * f)))

        def residual(x0, filters, stride=1, activate_before_residual=False):
            x = leaky_relu(tf.layers.batch_normalization(x0, **bn_args))
            if activate_before_residual:
                x0 = x

            x = tf.layers.conv2d(x, filters, 3, strides=stride, **conv_args(3, filters))
            x = leaky_relu(tf.layers.batch_normalization(x, **bn_args))
            x = tf.layers.conv2d(x, filters, 3, **conv_args(3, filters))

            if x0.get_shape()[3] != filters:
                x0 = tf.layers.conv2d(x0, filters, 1, strides=stride, **conv_args(1, filters))

            return x0 + x

        with tf.variable_scope('classify', reuse=tf.AUTO_REUSE, custom_getter=getter):
            y = tf.layers.conv2d((x - self.dataset.mean) / self.dataset.std, 16, 3, **conv_args(3, 16))
            for scale in range(scales):
                y = residual(y, filters << scale, stride=2 if scale else 1, activate_before_residual=scale == 0)
                for i in range(repeat - 1):
                    y = residual(y, filters << scale)

            y = leaky_relu(tf.layers.batch_normalization(y, **bn_args))
            y = embeds = tf.reduce_mean(y, [1, 2])
            if dropout and training:
                y = tf.nn.dropout(y, 1 - dropout)
            logits = tf.layers.dense(y, self.nclass, kernel_initializer=tf.glorot_normal_initializer())
        
        print(logits.shape, embeds.shape)
        return EasyDict(logits=logits, embeds=embeds)


class ShakeNet(ClassifySemi):
    def classifier(self, x, scales, filters, repeat, training, getter=None, dropout=0, **kwargs):
        del kwargs
        bn_args = dict(training=training, momentum=0.999)

        def conv_args(k, f):
            return dict(padding='same', use_bias=False,
                        kernel_initializer=tf.random_normal_initializer(stddev=tf.rsqrt(0.5 * k * k * f)))

        def residual(x0, filters, stride=1):
            def branch():
                x = tf.nn.relu(x0)
                x = tf.layers.conv2d(x, filters, 3, strides=stride, **conv_args(3, filters))
                x = tf.nn.relu(tf.layers.batch_normalization(x, **bn_args))
                x = tf.layers.conv2d(x, filters, 3, **conv_args(3, filters))
                x = tf.layers.batch_normalization(x, **bn_args)
                return x

            x = layers.shakeshake(branch(), branch(), training)

            if stride == 2:
                x1 = tf.layers.conv2d(tf.nn.relu(x0[:, ::2, ::2]), filters >> 1, 1, **conv_args(1, filters >> 1))
                x2 = tf.layers.conv2d(tf.nn.relu(x0[:, 1::2, 1::2]), filters >> 1, 1, **conv_args(1, filters >> 1))
                x0 = tf.concat([x1, x2], axis=3)
                x0 = tf.layers.batch_normalization(x0, **bn_args)
            elif x0.get_shape()[3] != filters:
                x0 = tf.layers.conv2d(x0, filters, 1, **conv_args(1, filters))
                x0 = tf.layers.batch_normalization(x0, **bn_args)

            return x0 + x

        with tf.variable_scope('classify', reuse=tf.AUTO_REUSE, custom_getter=getter):
            y = tf.layers.conv2d((x - self.dataset.mean) / self.dataset.std, 16, 3, **conv_args(3, 16))
            for scale, i in itertools.product(range(scales), range(repeat)):
                with tf.variable_scope('layer%d.%d' % (scale + 1, i)):
                    if i == 0:
                        y = residual(y, filters << scale, stride=2 if scale else 1)
                    else:
                        y = residual(y, filters << scale)

            y = embeds = tf.reduce_mean(y, [1, 2])
            if dropout and training:
                y = tf.nn.dropout(y, 1 - dropout)
            logits = tf.layers.dense(y, self.nclass, kernel_initializer=tf.glorot_normal_initializer())
        return EasyDict(logits=logits, embeds=embeds)


class SqueezeNet(ClassifySemi):
    def fire_module(self, x, s1, e1, e3):
        x = tf.layers.conv2d(x, filters=s1, kernel_size=1, activation='relu', padding='same')
        
        left = tf.layers.conv2d(x, filters=e1, kernel_size=1, activation='relu', padding='same')
        right = tf.layers.conv2d(x, filters=e3, kernel_size=3, activation='relu', padding='same')
        
        x = tf.concat([left, right], axis=-1)
        return x

    def classifier(self, x, scales, filters, repeat, training, getter=None, dropout=0, **kwargs):
        del kwargs
        # bn_args = dict(training=training, momentum=0.999)

        with tf.variable_scope('classify', reuse=tf.AUTO_REUSE, custom_getter=getter):
            x = (x - self.dataset.mean) / self.dataset.std
            # x = tf.layers.conv2d(x, filters=96, kernel_size=7, strides=2, activation='relu')
            x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=2, activation='relu')
            x = tf.layers.max_pooling2d(x, pool_size=3, strides=2, padding='same')

            x = self.fire_module(x, 16, 64, 64)
            x = self.fire_module(x, 16, 64, 64)
            x = tf.layers.max_pooling2d(x, pool_size=3, strides=2,  padding='same')
            
            x = self.fire_module(x, 32, 128, 128)
            x = self.fire_module(x, 32, 128, 128)
            x = tf.layers.max_pooling2d(x, pool_size=3,strides=2,  padding='same')
            
            x = self.fire_module(x, 48, 192, 192)
            x = self.fire_module(x, 48, 192, 192)
            x = self.fire_module(x, 64, 256, 256)
            x = self.fire_module(x, 64, 256, 256)
            
            if training:
                x = tf.layers.dropout(x, 0.5)

            x = tf.layers.conv2d(x, self.nclass, kernel_size = 1)
            x = tf.nn.relu(x)

            logits = tf.keras.layers.GlobalAveragePooling2D()(x)

            return EasyDict(logits=logits, embeds=x)


class SqueezeNetCifar(ClassifySemi):
    def fire_module(self, x, s1, e1, e3):
        x = tf.layers.conv2d(x, filters=s1, kernel_size=1, activation='relu', padding='same')
        
        left = tf.layers.conv2d(x, filters=e1, kernel_size=1, activation='relu', padding='same')
        right = tf.layers.conv2d(x, filters=e3, kernel_size=3, activation='relu', padding='same')
        
        x = tf.concat([left, right], axis=-1)
        return x

    def classifier(self, x, scales, filters, repeat, training, getter=None, dropout=0, **kwargs):
        del kwargs
        # bn_args = dict(training=training, momentum=0.999)

        with tf.variable_scope('classify', reuse=tf.AUTO_REUSE, custom_getter=getter):
            x = (x - self.dataset.mean) / self.dataset.std
            # x = tf.layers.conv2d(x, filters=96, kernel_size=7, strides=2, activation='relu')
            x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=2, activation='relu')
            x = tf.layers.max_pooling2d(x, pool_size=3, strides=2, padding='same')

            x = self.fire_module(x, 16, 64, 64)
            x = self.fire_module(x, 16, 64, 64)
            x = tf.layers.max_pooling2d(x, pool_size=2, strides=1,  padding='same')
            
            x = self.fire_module(x, 32, 128, 128)
            x = self.fire_module(x, 32, 128, 128)
            x = tf.layers.max_pooling2d(x, pool_size=2,strides=1,  padding='same')
            
            x = self.fire_module(x, 48, 192, 192)
            x = self.fire_module(x, 48, 192, 192)
            x = self.fire_module(x, 64, 256, 256)
            x = self.fire_module(x, 64, 256, 256)
            
            if training:
                x = tf.layers.dropout(x, 0.5)

            x = tf.layers.conv2d(x, self.nclass, kernel_size = 1)
            x = tf.nn.relu(x)

            logits = tf.keras.layers.GlobalAveragePooling2D()(x)

            return EasyDict(logits=logits, embeds=x)


class SqueezeNetMini(ClassifySemi):
    # https://github.com/zshancock/SqueezeNet_vs_CIFAR10/blob/master/squeezenet_architecture.py
    def fire_module(self, x, s1, e1, e3):
        x = tf.layers.conv2d(x, s1, (1,1), activation='relu', padding = 'valid')
        
        # define the expand layer's (1,1) filters
        expand_1x1 = tf.layers.conv2d(x, e1, (1,1), activation='relu', padding='valid')
        
        # define the expand layer's (3,3) filters
        expand_3x3 = tf.layers.conv2d(x, e3, (3,3), activation='relu', padding='same')
        
        x = tf.concat([expand_1x1, expand_3x3], axis=-1)
        return x

    def classifier(self, x, scales, filters, repeat, training, getter=None, dropout=0, **kwargs):
        del kwargs
        # bn_args = dict(training=training, momentum=0.999)

        with tf.variable_scope('classify', reuse=tf.AUTO_REUSE, custom_getter=getter):
            x = (x - self.dataset.mean) / self.dataset.std
            # x = tf.layers.conv2d(x, filters=96, kernel_size=7, strides=2, activation='relu')
            x = tf.layers.conv2d(x, 64, (3, 3), strides=(2, 2), activation='relu', padding='valid')
            x = tf.layers.max_pooling2d(x, pool_size=(3, 3), strides=(2, 2))

            x = self.fire_module(x, s1=16, e1=64, e3=64)
            x = self.fire_module(x, s1=16, e1=64, e3=64)

            x = self.fire_module(x, s1=32, e1=128, e3=128)
            x = self.fire_module(x, s1=32, e1=128, e3=128)
            if training:
                x = tf.layers.dropout(x, 0.5)

            x = tf.layers.conv2d(x, self.nclass, (1, 1), activation='relu', padding='valid')
            logits = tf.keras.layers.GlobalAveragePooling2D()(x)

            return EasyDict(logits=logits, embeds=x)


class SqueezeNetMini1(ClassifySemi):
    # https://github.com/zshancock/SqueezeNet_vs_CIFAR10/blob/master/squeezenet_architecture.py
    def fire_module(self, x, s1, e1, e3):
        x = tf.layers.conv2d(x, s1, (1,1), activation='relu', padding = 'valid')
        
        # define the expand layer's (1,1) filters
        expand_1x1 = tf.layers.conv2d(x, e1, (1,1), activation='relu', padding='valid')
        
        # define the expand layer's (3,3) filters
        expand_3x3 = tf.layers.conv2d(x, e3, (3,3), activation='relu', padding='same')
        
        x = tf.concat([expand_1x1, expand_3x3], axis=-1)
        return x

    def classifier(self, x, scales, filters, repeat, training, getter=None, dropout=0, **kwargs):
        del kwargs
        # bn_args = dict(training=training, momentum=0.999)

        with tf.variable_scope('classify', reuse=tf.AUTO_REUSE, custom_getter=getter):
            x = (x - self.dataset.mean) / self.dataset.std
            # x = tf.layers.conv2d(x, filters=96, kernel_size=7, strides=2, activation='relu')
            x = tf.layers.conv2d(x, 64, (3, 3), strides=(1, 1), activation='relu', padding='valid')
            x = tf.layers.max_pooling2d(x, pool_size=(3, 3), strides=(2, 2))

            x = self.fire_module(x, s1=16, e1=64, e3=64)
            x = self.fire_module(x, s1=16, e1=64, e3=64)

            x = self.fire_module(x, s1=32, e1=128, e3=128)
            x = self.fire_module(x, s1=32, e1=128, e3=128)
            if training:
                x = tf.layers.dropout(x, 0.5)

            x = tf.layers.conv2d(x, self.nclass, (1, 1), activation='relu', padding='valid')
            logits = tf.keras.layers.GlobalAveragePooling2D()(x)

            return EasyDict(logits=logits, embeds=x)



class MultiModel(CNN13, ResNet, ShakeNet, SqueezeNet, SqueezeNetCifar, SqueezeNetMini, SqueezeNetMini1):
    MODELS = ('cnn13', 'resnet', 'shake', 'squeezenet' , 'squeezenetcifar', 'squeezenetmini', 'squeezenetmini1')
    MODEL_CNN13, MODEL_RESNET, MODEL_SHAKE, MODEL_SQUEEZENET, MODEL_SQUEEZENETCIFAR, MODEL_SQUEEZENETMINI, MODEL_SQUEEZENETMINI1  = MODELS

    def augment(self, x, l, smoothing, **kwargs):
        del kwargs
        return x, l - smoothing * (l - 1. / self.nclass)

    def classifier(self, x, arch, **kwargs):
        if arch == self.MODEL_CNN13:
            return CNN13.classifier(self, x, **kwargs)
        elif arch == self.MODEL_RESNET:
            return ResNet.classifier(self, x, **kwargs)
        elif arch == self.MODEL_SHAKE:
            return ShakeNet.classifier(self, x, **kwargs)
        elif arch == self.MODEL_SQUEEZENET:
            return SqueezeNet.classifier(self, x, **kwargs)
        elif arch == self.MODEL_SQUEEZENETCIFAR:
            return SqueezeNetCifar.classifier(self, x, **kwargs)
        elif arch == self.MODEL_SQUEEZENETMINI:
            return SqueezeNetMini.classifier(self, x, **kwargs)
        elif arch == self.MODEL_SQUEEZENETMINI1:
            return SqueezeNetMini1.classifier(self, x, **kwargs)
        raise ValueError('Model %s does not exists, available ones are %s' % (arch, self.MODELS))


flags.DEFINE_enum('arch', MultiModel.MODEL_RESNET, MultiModel.MODELS, 'Architecture.')