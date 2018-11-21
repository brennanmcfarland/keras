from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.layers import Lambda
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import matplotlib.pyplot as plt
import argparse
import os
import math

import sys
import csv
import re
import random
from skimage import io
from skimage.transform import downscale_local_mean
import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras.utils import Sequence
from keras.models import Model
from keras.layers import Multiply, Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Activation, Reshape
from keras.optimizers import RMSprop, Adam
from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras import initializers


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


data_root = 'E:/ML/keras/data/cartoons/'
random.seed()

# TODO: these two helper functions were copied directly from the example,
# look at them closer when you get the chance
def load_metadata():
    metadata = []
    with open(data_root + 'metadata.csv', 'r', newline='') as metadata_file:
        reader = csv.reader(metadata_file, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            metadatum = row[:2] + [int(n) for n in re.findall(r'\d+', row[2])]
            metadata.append(metadatum)
    return metadata


metadata = load_metadata()
random.shuffle(metadata)
metadata_train = metadata[len(metadata)//10:]
metadata_test = metadata[:len(metadata_train)]
num_data = len(metadata)

classes = set([datum[0] for datum in metadata])
# build a dictionary mapping between name strings and ids
class_to_id = dict((n, i) for i, n in enumerate(classes))
id_to_class = dict((i, n) for i, n in enumerate(classes))
num_classes = len(classes)

# if outside the tolerance range, return None (it's not a valid datum)
# if to large, crop from both sides to fit
# if to small, pad with maximum value (white) to fit
def scale_to_target(image, initial_y, target_y, shrink_tolerance, grow_tolerance):
    if(target_y-initial_y > grow_tolerance or initial_y-target_y > shrink_tolerance):
        #print('image oustide acceptable dimensions, ', image.shape)
        return None
    elif(initial_y > target_y):
        #print('shrinking to fit target dimensions')
        return image[:target_y]
    else: # initial_y <= target_y
        #print('growing to fit target dimensions')
        padding = (target_y-initial_y)//2
        return np.pad(image, ((padding, target_y - initial_y - padding),(0,0)), 'maximum')

def get_image(metadata, datum_index):
    img_x, img_y = 150, 450
    metadatum = metadata[datum_index]
    # TODO: we'll need to make it color for some strips later
    img_raw = io.imread(data_root + 'images/' + metadatum[0] + metadatum[1] + '.png', as_gray=True)
    # print('raw image: ', img_raw.shape)
    img_scaled = scale_to_target(img_raw, metadatum[2], img_x*2, 5, 120)
    return img_scaled


# 1 is white, 0 is black, remember
def convert_to_sample(img):
    p_black = .1
    sample = np.random.choice((0, 1), size=img.shape, p=(p_black, 1.0-p_black)) # p is probability of each option
    # element-wise max
    return np.maximum(sample, img)

# TODO: optimize the batches to use np arrays from the getgo?
def get_batch(batch_size, metadata):
    img_x, img_y = 150, 450
    batch_x = np.zeros((batch_size, img_x, img_y), dtype=float)
    batch_y = np.zeros((batch_size, img_x, img_y), dtype=float)
    for i in range(batch_size):
        datum_indices = np.random.randint(0, len(metadata), size=2)
        img_scaled = None
        j = 0
        while img_scaled is None:
            img_scaled = get_image(metadata, (datum_indices[0] + j) % len(metadata))
            metadatum = metadata[datum_indices[0]]
            j += 1
            #datum_indices[0] = (datum_indices[0] + 1) % len(metadata)
        
        # put it in a tensor after downscaling it and padding it
        img_downscaled = downscale_local_mean(img_scaled, (2, 2))
        # normalize channel values
        # TODO: was maximum, but maybe manuallly setting to 1 will make ti work better for now
        batch_x[i] = np.pad(img_downscaled, ((0,0), (0, img_y-img_downscaled.shape[1])), 'constant', constant_values=1)

        img_scaled = None
        j = 0
        while img_scaled is None:
            img_scaled = get_image(metadata, (datum_indices[1] + j) % len(metadata))
            metadatum = metadata[datum_indices[1]]
            j += 1
            #datum_indices[1] = (datum_indices[1] + 1) % len(metadata)
        
        # put it in a tensor after downscaling it and padding it
        img_downscaled = downscale_local_mean(img_scaled, (2, 2))
        # normalize channel values
        batch_y[i] = np.pad(img_downscaled, ((0,0), (0, img_y-img_downscaled.shape[1])), 'maximum')
        batch_x[i] /= 255
        batch_y[i] /= 255
        batch_y = convert_to_sample(batch_x)
        #batch_y[i][class_to_id[metadatum[0]]] = 1
    #return np.expand_dims(batch_x, axis=3), batch_y
    return np.expand_dims(batch_x, axis=3), np.expand_dims(batch_y, axis=3)
    # TODO: need to have it differentiate different strip sizes, ie sunday vs weekday strips


# TODO: may want to try using the ImageDataGenerator class, but the problem is that it would
# have to be able to work with one batch at a time loaded with its respective metadata from
# files
# could write a quick script to rename all the images with ids and make that the first value in
# each csv row, then use flow_from_directory and use that id to get the metadata, but I'm too
# lazy to do that right now
class DataProvider(Sequence):

    metadata = None
    batch_size = 1 # TODO: obviously we'll need to find the optimal batch size
    
    def __init__(self, metadata):
        self.metadata = metadata
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        x, y = get_batch(self.batch_size, self.metadata)
        if x is None or y is None:
            raise ValueError("input x or y is none")
        return y, x


# TODO: condense these repeated layer patterns into "super layers"?
# TODO: obviously, this will need tweaking, try adding dropout, more layers, etc
# if it gets an OOM error, the network or data input size or batch size has to be scaled down

# TODO: does this need to correspond to the # output layers for hte encoder?
latent_dim = 2 # TODO: this will probably need to be upped a lot (or does it, since it's not one-hot but just regular vectors?

class PixelCNN(Conv2D):
    ''' Start w/ simple PixelCNN and then make it better once it works '''

    def __init__(self, filters, *args, mask_current=True, n_channels=1, mono=False, **kwargs):
        self.mask_current = mask_current
        self.mask = None
        self.num_filters = filters
        self.num_channels = n_channels
        super(PixelCNN, self).__init__(filters, *args, **kwargs)


    # this is where you will define your weights. This method must set
    # self.built = True at the end, which can be done by calling
    # super([Layer], self).build()
    def build(self, input_shape):
        # kernel.shape was W_shape
        # kernel_shape = (150, 450, 1, 1)
        print(self.kernel_size)
        #kernel_shape = (3, 3, 1, 1)
        kernel_shape = self.kernel_size + (self.num_channels, self.num_channels)
        # TODO: fiddle with dimensions or better make them automatic/parrams
        self.kernel = self.add_weight(name='kernel',
                                      shape=kernel_shape,
                                      initializer=initializers.random_uniform(minval=-1.0, maxval=1.0),
                                      trainable=True)
        self.mask = np.zeros(self.kernel.shape) # W_shape must be inherit from Conv2D or layer
        assert self.mask.shape[0] == self.mask.shape[1] # assert that mask height = width
        filter_size = self.mask.shape[0]
        filter_center = filter_size/2

        # unmask everything before the center
        self.mask[:math.floor(filter_center)] = 1 # unmask rows above
        self.mask[:math.ceil(filter_center), :math.floor(filter_center)] = 1 # unmask cols to the left in same row

        if not self.mask_current:
            self.mask[math.ceil(filter_center), math.ceil(filter_center)] = 1

        self.mask = K.variable(self.mask)

        self.bias = None
        self.built = True
        #super(PixelCNN, self).build(input_shape)

    # this is where the layer's logic lives. Unless you want your layer to
    # support masking, you only have to care about the first argument
    # passed to call: the input tensor
    def call(self, x, mask=None):
        ''' calculate gated activation maps given input maps '''
        #print("KERNEL:")
        #print(K.eval(self.kernel))
        #print("MASKED KERNEL:")
        #print(K.eval(self.kernel * self.mask))
        # kernel used to be W
        # TODO: I did this just to make everything masked white, but can we do this when
        # precalculating the mask instead?
        output = K.conv2d(x, self.kernel * self.mask + (-self.mask + 1),
                          strides=self.strides,
                          padding=self.padding,
                          data_format=self.data_format)
        #return output
        # the above used to also include filter_shape=self.kernel.shape,
        # but I think that was just to specify the output shape and is
        # now deprecated
        #if self.bias:
        #    if self.data_format == 'channels_last':
        #        # get_weights() returns [W, b]
        #        output += K.reshape(self.get_weights()[1], (1, 1, 1, self.filters))
        #    else:
        #        print(self.data_format)
        #        raise ValueError('PixelCNN layer works with tensorflow backend only')
        output = self.activation(output)
        return output

    def get_config(self):
        return dict(list(super().get_config().items()) + list({'mask': self.mask_current}.items()))
    # in case your layer modifies the shape of its input, you should
    # specify here the shape transformation logic. This allows Keras to
    # do automatic shape inference
    # implemented in base ok i think
    def compute_output_shape(self, input_shape):
        return super(PixelCNN, self).compute_output_shape(input_shape)



# decoder
# upscales the image between convolutions until it gets to the original dim,
# so using different sized images will require this to be adjusted
# input = Input(shape=(150, 450, 3, num_classes), name='z_sampling')

#input = Input(shape=(num_classes,), name='z_sampling')
#layer = Dense(67500, activation='sigmoid')(input)
#layer = Reshape((150, 450, 1))(layer)
#output = PixelCNN(1, 7, strides=1, mask_current=True, padding='same')(layer)

# TODO: here we're just having it generate from one class, should try to parameterize it
# later to work for multiple
input = Input(shape=(150, 450, 1), name='z_sampling')
layer = PixelCNN(1, 7, strides=1, mask_current=True, padding='same')(input)
output = PixelCNN(1, 7, strides=1, mask_current=True, padding='same')(layer)

# layer = Dense(2700, activation='relu')(input)
# layer = Dense(2700, activation='relu')(layer)
# layer = Dense(2700, activation='relu')(layer)
# layer = Reshape((30, 90, 1))(layer)
# layer = Conv2DTranspose(32, (3, 3), padding='same', activation='sigmoid')(layer)
# layer = Conv2DTranspose(32, (3, 3), padding='same', activation='sigmoid')(layer)
# layer = UpSampling2D(size=(5, 5))(layer)
# layer = Conv2DTranspose(32, (3, 3), padding='same', activation='sigmoid')(layer)
#layer = UpSampling2D(size=(2, 2))(layer)
# output = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(layer)


# TODO: the shape is the same as returned by Conv2D, but that also throws an error
# substituting with Conv2D returns ValueError: cannot select an axis to squeeze out which has size not equal to one
# args are filters, rank, kernel size
# layer = PixelCNN(32, 2, 16)(layer)
# output = UpSampling2D(size=(6, 6))(layer)


decoder = Model(input, output, name='decoder')

original_dim = 150 * 450 # TODO: set this to just the dimensions of the input later instead of hard coding it
optimizer = Adam(lr=.00001)
decoder.compile(loss='binary_crossentropy', optimizer=optimizer) # TODO: find out which optimizer works best
decoder.summary()

def report_epoch_progress(epoch, logs):
    print('epoch progress report:')
    for i in range(num_classes):
        #latent = np.zeros(num_classes)
        #latent[i] = 1
        #latent = np.expand_dims(latent, axis=0)
        #latent = np.array([[0,0,0]])
        #print(latent.shape)
        # the prediction is coming back as None, why?
        example = data_test.__getitem__(3)
        latent = example[1]
        #latent = np.expand_dims(latent, axis=0)
        print('LATENT: ', latent.shape)
        img = decoder.predict(latent)
        print(img.shape)
        img = np.squeeze(img, axis=(0,3))
        img *= 255
        img = img.astype(int)
        print('image shape:', img.shape)
        filename = 'epoch-output/latest-' + str(i) + '-predicted.png'
        io.imsave(filename, img)
        actual = example[0]
        actual = np.squeeze(actual, axis=(0,3))
        actual *= 255
        actual = actual.astype(int)
        filename2 = 'epoch-output/latest-' + str(i) + '-actual.png'
        io.imsave(filename2, actual)

        latent = np.squeeze(latent, axis=(0,3))
        latent *= 255
        latent = latent.astype(int)
        filename3 = 'epoch-output/latest-' + str(i) + '-input.png'
        io.imsave(filename3, latent)


progress_callback = LambdaCallback(on_epoch_end=report_epoch_progress)
checkpoint_callback = ModelCheckpoint('./model-checkpoint.ckpt')
tensorboard_callback = TensorBoard(log_dir='../logs/tensorboard-logs', write_images=True)
callbacks = [progress_callback, checkpoint_callback, tensorboard_callback]

# TODO: add validation data (split the training data)
data_train = DataProvider(metadata_train)
data_test = DataProvider(metadata_test)
epochs = 60
report_epoch_progress(None, None)
decoder.fit_generator(
    data_train,
    validation_data=data_test,
    steps_per_epoch=num_data//data_train.batch_size,
    epochs=epochs,
    callbacks=callbacks
)
