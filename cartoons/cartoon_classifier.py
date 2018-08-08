import csv
import re
import random
from skimage import io
from skimage.transform import downscale_local_mean
import numpy as np
from keras.utils import Sequence
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Activation
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard


data_root = 'E:/ML/keras/data/cartoons/'
random.seed()

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


# TODO: optimize the batches to use np arrays from the getgo?
def get_batch(batch_size, metadata):
    img_x, img_y = 145, 450
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
    batch_size = 2 # TODO: obviously we'll need to find the optimal batch size
    
    def __init__(self, metadata):
        self.metadata = metadata
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        return get_batch(self.batch_size, self.metadata)


# TODO: condense these repeated layer patterns into "super layers"?
# TODO: obviously, this will need tweaking, try adding dropout, more layers, etc
# if it gets an OOM error, the network or data input size or batch size has to be scaled down
input = Input(shape=(145, 450, 1))
layer = Conv2D(32, (3, 3), padding='same', activation='relu')(input)
layer = MaxPooling2D(pool_size=(2, 2))(layer)
layer = Conv2D(32, (3, 3), padding='same')(layer)
layer = MaxPooling2D(pool_size=(2, 2))(layer)

layer = Conv2D(32, (3, 3), padding='same', activation='relu')(layer)
layer = MaxPooling2D(pool_size=(2, 2))(layer)
layer = Conv2D(32, (3, 3), padding='same')(layer)
layer = MaxPooling2D(pool_size=(2, 2))(layer)

layer = Flatten()(layer)
layer = Dense(512)(layer)
layer = Activation('relu')(layer)
layer = Dense(num_classes)(layer)
layer = Activation('softmax')(layer)

model = Model(input, layer)
optimizer = RMSprop(lr = .0005)
model.compile(loss = 'categorical_crossentropy', optimizer=optimizer)

def report_epoch_progress(epoch, logs):
    print('epoch progress report:')


progress_callback = LambdaCallback(on_epoch_end=report_epoch_progress)
checkpoint_callback = ModelCheckpoint('./model-checkpoint.ckpt')
tensorboard_callback = TensorBoard(log_dir='../logs/tensorboard-logs', write_images=True)
callbacks = [progress_callback, checkpoint_callback, tensorboard_callback]

# TODO: add validation data (split the training data)
data_train = DataProvider(metadata_train)
data_test = DataProvider(metadata_test)
epochs = 60
model.fit_generator(
    data_train,
    validation_data=data_test,
    steps_per_epoch=num_data//data_train.batch_size,
    epochs=epochs,
    callbacks=callbacks
)
