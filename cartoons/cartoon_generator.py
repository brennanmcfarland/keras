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

import csv
import re
import random
from skimage import io
from skimage.transform import downscale_local_mean
import numpy as np
from keras.utils import Sequence
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dense, Flatten, Activation, Reshape
from keras.optimizers import RMSprop, Adam
from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard


# https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py

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


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    pass # need to install graphviz for this to work
#    encoder, decoder = models
#    x_test, y_test = data
#    os.makedirs(model_name, exist_ok=True)
#
#    filename = os.path.join(model_name, "vae_mean.png")
#    # display a 2D plot of the digit classes in the latent space
#    z_mean, _, _ = encoder.predict(x_test,
#                                   batch_size=batch_size)
#    plt.figure(figsize=(12, 10))
#    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
#    plt.colorbar()
#    plt.xlabel("z[0]")
#    plt.ylabel("z[1]")
#    plt.savefig(filename)
#    plt.show()
#
#    filename = os.path.join(model_name, "digits_over_latent.png")
#    # display a 30x30 2D manifold of digits
#    n = 30
#    digit_size = 28
#    figure = np.zeros((digit_size * n, digit_size * n))
#    # linearly spaced coordinates corresponding to the 2D plot
#    # of digit classes in the latent space
#    grid_x = np.linspace(-4, 4, n)
#    grid_y = np.linspace(-4, 4, n)[::-1]
#
#    for i, yi in enumerate(grid_y):
#        for j, xi in enumerate(grid_x):
#            z_sample = np.array([[xi, yi]])
#            x_decoded = decoder.predict(z_sample)
#            digit = x_decoded[0].reshape(digit_size, digit_size)
#            figure[i * digit_size: (i + 1) * digit_size,
#                   j * digit_size: (j + 1) * digit_size] = digit
#
#    plt.figure(figsize=(10, 10))
#    start_range = digit_size // 2
#    end_range = n * digit_size + start_range + 1
#    pixel_range = np.arange(start_range, end_range, digit_size)
#    sample_range_x = np.round(grid_x, 1)
#    sample_range_y = np.round(grid_y, 1)
#    plt.xticks(pixel_range, sample_range_x)
#    plt.yticks(pixel_range, sample_range_y)
#    plt.xlabel("z[0]")
#    plt.ylabel("z[1]")
#    plt.imshow(figure, cmap='Greys_r')
#    plt.savefig(filename)
#    plt.show()



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


# TODO: optimize the batches to use np arrays from the getgo?
def get_batch(batch_size, metadata):
    img_x, img_y = 150, 450
    batch_x = np.zeros((batch_size, img_x, img_y), dtype=float)
    batch_y = np.zeros((batch_size, num_classes), dtype=float)
    for i in range(batch_size):
        img_scaled = None
        while img_scaled is None:
            metadatum = metadata[random.randrange(len(metadata))]
            # TODO: we'll need to make it color for some strips later
            img_raw = io.imread(data_root + 'images/' + metadatum[0] + metadatum[1] + '.png', as_gray=True)
            # print('raw image: ', img_raw.shape)
            img_scaled = scale_to_target(img_raw, metadatum[2], img_x*2, 5, 120)
        # put it in a tensor after downscaling it and padding it
        img_downscaled = downscale_local_mean(img_scaled, (2, 2))
        # normalize channel values
        img_downscaled /= 255# TODO: test to see if this is actually helpful, maybe research it
        batch_x[i] = np.pad(img_downscaled, ((0,0), (0, img_y-img_downscaled.shape[1])), 'maximum')
        batch_y[i][class_to_id[metadatum[0]]] = 1
    return np.expand_dims(batch_x, axis=3), batch_y
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
        return y, x


# TODO: condense these repeated layer patterns into "super layers"?
# TODO: obviously, this will need tweaking, try adding dropout, more layers, etc
# if it gets an OOM error, the network or data input size or batch size has to be scaled down

# TODO: does this need to correspond to the # output layers for hte encoder?
latent_dim = 2 # TODO: this will probably need to be upped a lot (or does it, since it's not one-hot but just regular vectors?

# decoder
# upscales the image between convolutions until it gets to the original dim,
# so using different sized images will require this to be adjusted
input = Input(shape=(num_classes,), name='z_sampling')
layer = Dense(2700, activation='relu')(input)
layer = Dense(2700, activation='relu')(layer)
layer = Dense(2700, activation='relu')(layer)
layer = Reshape((30, 90, 1))(layer)
layer = Conv2DTranspose(32, (3, 3), padding='same', activation='sigmoid')(layer)
layer = Conv2DTranspose(32, (3, 3), padding='same', activation='sigmoid')(layer)
layer = UpSampling2D(size=(5, 5))(layer)
layer = Conv2DTranspose(32, (3, 3), padding='same', activation='sigmoid')(layer)
#layer = UpSampling2D(size=(2, 2))(layer)
output = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(layer)


decoder = Model(input, output, name='decoder')
decoder.summary()
# plot_model(decoder, to_file='cartoon_vae_decoder.png', show_shapes=True)

original_dim = 150 * 450 # TODO: set this to just the dimensions of the input later instead of hard coding it
optimizer = Adam(lr=.0001)
decoder.compile(loss='binary_crossentropy', optimizer=optimizer) # TODO: find out which optimizer works best
decoder.summary()
# plot_model(vae, to_file='cartoon_vae.png', show_shapes=True)

def report_epoch_progress(epoch, logs):
    print('epoch progress report:')
    for i in range(num_classes):
        latent = np.zeros(num_classes)
        latent[i] = 1
        latent = np.expand_dims(latent, axis=0)
        #latent = np.array([[0,0,0]])
        print(latent.shape)
        img = decoder.predict(latent)
        img = np.squeeze(img, axis=(0,3))
        img *= 255
        img = img.astype(int)
        print('image shape:', img.shape)
        filename = 'epoch-output/latest-' + str(i) + '.png'
        io.imsave(filename, img)


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
    # validation_data=data_test,
    steps_per_epoch=num_data//data_train.batch_size,
    epochs=epochs,
    callbacks=callbacks
)
