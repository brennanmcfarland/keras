from bs4 import BeautifulSoup
import urllib.request as urllib
import random
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM, BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import numpy as np
import sys


root = "file:./data/"

def source_from_html(url):
    html = urllib.urlopen(url)
    soup = BeautifulSoup(html, 'html.parser')
    # if url == "file:/sym/ML/LOTR_fellowship_of_ring.html":
    if url == root + "LOTR_fellowship_of_ring.htm":
        source_tagged = soup.h3.parent.find_next_siblings('p')
    else:
        source_tagged = soup.h3.parent.parent.find_next_siblings('p')
    source = ' '.join([t.get_text().replace('\r\n', ' ').replace('\n', ' ') for t in source_tagged[:-2]])
    return source


# TODO: is removing punctuation necessary?  first try it without
# doing so and see what happens
source_urls = [
    root + "LOTR_fellowship_of_ring.htm",
    root + "LOTR_two_towers.htm",
    root +"LOTR_return_of_king.htm"]
sources = [source_from_html(source_url) for source_url in source_urls]
source = ' '.join(sources)

source_len = len(source)

# assign each char to a number as a dict key
chars = sorted(list(set(source)))
char_count = len(chars)
print('char count:', char_count)
char_to_index = dict((c, i) for i, c in enumerate(chars))
index_to_char = dict((i, c) for i, c in enumerate(chars))

# split the source into overlapping sequences of characters
sequence_length = 40 # all sequences are this length if the source is
sequence_step = 3 # amount offset from each sequence to the next
sequences = []
ground_truth = []
for i in range(0, len(source)-sequence_length, sequence_step):
    sequences.append(source[i: i+sequence_length])
    ground_truth.append(source[i+sequence_length])
sequence_count = len(sequences)
print('sequence count:', sequence_count)

# the given sequences
x_train = np.zeros((sequence_count, sequence_length, char_count), dtype=np.bool)
# what the next char after each sequence is
y_train = np.zeros((sequence_count, char_count), dtype=np.bool)
for i, sequence in enumerate(sequences):
    for t, char in enumerate(sequence): # t for time!
        x_train[i, t, char_to_index[char]] = 1
    y_train[i, char_to_index[ground_truth[i]]] = 1


# TODO: I saw another example that used 3 LSTM layers, which
# would likely allow it to capture higher level abstractions,
# at least on a grammatical level.  run it with 1 first since
# it will train quicker and then once I have an MVP, improve it
out_dim = 128

model = Sequential()
model.add(LSTM(out_dim, input_shape=(sequence_length, char_count), return_sequences=True))
model.add(LSTM(out_dim, input_shape=(sequence_length, char_count), return_sequences=True))
model.add(LSTM(out_dim, input_shape=(sequence_length, char_count), return_sequences=True))
model.add(LSTM(out_dim, input_shape=(sequence_length, char_count)))
model.add(Dense(char_count))
model.add(Dense(char_count))
model.add(Dense(char_count))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=.0005)
# we use crossentropy because it's a form of classification problem,
# where the classes are which character comes next
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# get the highest probability index based on our predictions, the best guess from our network
def sample_index(predictions, temperature=1.0):
    # not sure what all the math is for, hopefully this will
    # become more clear as I continue to work with it
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temperature
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)
    probabilities = np.random.multinomial(1, predictions, 1)
    return np.argmax(probabilities)

    
def generate_text(output_len):
    start_index = random.randint(0, source_len - sequence_length - 1)
    for diversity in [.2, .5, 1.0, 1.2]:
        
        # grab an input sequence
        sequence = source[start_index: start_index+sequence_length]
        print('generating with input: "', str(sequence), '"')
        
        # vectorize it
        # TODO: make this a function
        for i in range(output_len):
            x_pred = np.zeros((1, sequence_length, char_count))
            for t, char in enumerate(sequence):
                x_pred[0, t, char_to_index[char]] = 1
        
            # predict the next character
            # not sure what the [0] is for, an added dimension for the number of vectors?  in this case, we only have one
            predictions = model.predict(x_pred, verbose=0)[0]
            
            # and add it to the sequence, cutting off the earliest char so it stays the same length
            # next_index is our best guess as to what character's index comes next
            # predictions finds that for us from the predictions array
            next_index = sample_index(predictions, diversity)
            next_char = index_to_char[next_index]
            sequence = sequence[1:] + next_char
        
            # of course we have to actually print the generated char
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

def report_epoch_progress(epoch, logs):
    print('\n===EPOCH COMPLETE===')
    print('end of epoch', epoch)
    print('generating sample text...')
    generate_text(400)


progress_callback = LambdaCallback(on_epoch_end=report_epoch_progress)
model_checkpoint = ModelCheckpoint("./model-checkpoint.ckpt")
tensorboard_callback = TensorBoard(log_dir='./logs/tensorboard-logs', write_images=True)

try:
    model = load_model("./model-checkpoint.ckpt")
    print("checkpoint file found; training from checkpoint weights")
except OSError:
    print("checkpoint file not found; training from initial weights...")

batch_size = 128
epochs=60
callbacks=[progress_callback, model_checkpoint, tensorboard_callback]
generate_text(400)
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
