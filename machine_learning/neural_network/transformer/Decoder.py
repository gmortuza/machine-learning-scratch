import tensorflow as tf
from DecodingLayer import DecodingLayer
from position import get_position_encoding


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.position_encoding = get_position_encoding(maximum_position_encoding, d_model)
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.layers = [DecodingLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate=rate)

    def call(self, x, encoder_output, is_training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        x += self.position_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for i, layer in enumerate(self.layers):
            x, block_1, block_2 = layer(x, encoder_output, is_training, look_ahead_mask, padding_mask)
            attention_weights[f"decoder_layer_{i + 1}_block_1"] = block_1
            attention_weights[f"decoder_layer_{i + 1}_block_2"] = block_2

        return x, attention_weights
