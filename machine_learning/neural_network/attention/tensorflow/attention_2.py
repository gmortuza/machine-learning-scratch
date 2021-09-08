import io
import unicodedata
import re

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Input
from tensorflow.python.keras.activations import softmax
from tensorflow.python.keras.layers import RepeatVector, Concatenate, Dense, Activation, Dot, LSTM, Bidirectional
from tensorflow.python.keras.models import Model

repeator = RepeatVector(30)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation='tanh')
densor2 = Dense(1, activation='relu')
activator = Activation(softmax, name="attention model")
dotor = Dot(axes=1)
n_a = 32  # number of units for the pre-attention, bi-directional LSTM's hidden state 'a'
n_s = 64  # number of units for the post-attention LSTM's hidden state 's'

"""
Dataset
"""
# set tensorflow to use the second GPU. There is something wrong with first GPU on iapetus
try:
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
except:
    print("Using the CPU")
# path_to_zip = tf.keras.utils.get_file('ben-eng.zip', origin='http://www.manythings.org/anki/ben-eng.zip',extract=True)
path_to_file = "dataset/spa.txt"


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


# en_sentence = u"May I borrow this book?"
# bn_sentence = u"আমাকে বইটা ধার দিবা?"
# print(preprocess_sentence(en_sentence))
# print(preprocess_sentence(bn_sentence))


# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, BENGALI]
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    return zip(*word_pairs)


en, bn = create_dataset(path_to_file, None)


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    # creating cleaned input, output pairs
    targ_lang, inp_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


# Try experimenting with the size of that dataset
num_examples = 30000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)

# Calculate max_length of the target tensors
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                target_tensor,
                                                                                                test_size=0.2)






post_activation_LSTM_cell = LSTM(n_s, return_state=True)
output_layer = Dense(max_length_targ, activation=softmax)


def one_step_attention(a, s_prev):
    s_prev = repeator(s_prev)
    concat = concatenator([a, s_prev])
    e = densor1(concat)
    energies = densor2(e)
    alphas = activator(energies)
    context = dotor([alphas, a])
    return context


def get_model(n_a, n_s, input_vocab_size, output_vocab_size):
    X = Input(shape=(input_vocab_size, ))
    s0 = Input(shape=(n_s, ), name="s0")
    c0 = Input(shape=(n_s, ), name="c0")
    s = s0
    c = c0

    outputs = []

    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)
    for t in range(output_vocab_size):
        context = one_step_attention(a, s)
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])
        out = output_layer(s)
        outputs.append(out)

    return Model(inputs=[X, s0, c0], outputs=outputs)


model = get_model(n_a, n_s, max_length_inp, max_length_targ)
model.summary()
print(model.summary())
