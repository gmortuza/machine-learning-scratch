import tensorflow as tf
from position import get_position_encoding
from EncodingLayer import EncodingLayer


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=.1):
        super(Encoder, self).__init__()
        self.d_model = d_model

        self.position_encoding = get_position_encoding(maximum_position_encoding, d_model)
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.layers = [EncodingLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, is_training, mask):
        # Add input embedding with positional_encoding
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        x += self.position_encoding[:, seq_len, :]
        # Dropout
        x = self.dropout(x, training=is_training)

        for layer in self.layers:
            x = layer(x, is_training, mask)

        return x  # (batch_size, input_seq_len, d_model)


if __name__ == '__main__':
    tf.config.experimental.set_visible_devices(tf.config.list_physical_devices('GPU')[1], 'GPU')
    sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8, dff=2048, input_vocab_size=8500,
                             maximum_position_encoding=10000)
    temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

    sample_encoder_output = sample_encoder(temp_input, is_training=False, mask=None)
    print(sample_encoder_output.shape)
