import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as text

dataset_name = "ted_hrlr_translate/pt_to_en"


# Convert the text into token
tokenizer_model = "ted_hrlr_translate_pt_en_converter"
tf.keras.utils.get_file(
    f"{tokenizer_model}.zip",
    f"https://storage.googleapis.com/download.tensorflow.org/models/{tokenizer_model}.zip",
    cache_dir='..', cache_subdir='', extract=True
)
tokenizer = tf.saved_model.load(tokenizer_model)


def tokenize_pair(pt, en):
    return tokenizer.pt.tokenize(pt).to_tensor(), tokenizer.en.tokenize(en).to_tensor()


def make_batches(ds, batch_size, buffer_size):
    return (
        ds.cache()
        .shuffle(buffer_size)
        .batch(batch_size)
        .map(tokenize_pair, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )


def get_data(batch_size=64, buffer_size=20000):
    dataset = tfds.load(dataset_name, as_supervised=True, shuffle_files=True)
    train_data, validation_data, test_data = dataset["train"], dataset["validation"], dataset["test"]
    return make_batches(train_data, batch_size, buffer_size), make_batches(validation_data, batch_size, buffer_size),\
           make_batches(test_data, batch_size, buffer_size), tokenizer

