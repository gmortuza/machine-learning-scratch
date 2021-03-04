import tensorflow as tf
from MultiHeadAttention import MultiHeadAttention


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation="relu"),
        tf.keras.layers.Dense(d_model)
    ])


class EncodingLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=.1):
        super(EncodingLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()

        # Dropout
        self.dropout_1 = tf.keras.layers.Dropout(rate)
        self.dropout_2 = tf.keras.layers.Dropout(rate)

    def call(self, x, is_training, mask):
        attention_output, _ = self.mha(x, x, x, mask)
        attention_output = self.dropout_1(attention_output, training=is_training)
        # Add residual connection and perform normalization
        out_1 = self.layer_norm_1(x + attention_output)

        ffn_output = self.ffn(out_1)
        ffn_output = self.dropout_2(ffn_output, training=is_training)
        out_2 = self.layer_norm_2(out_1 + ffn_output)

        return out_2  # (batch_size, input_seq_len, d_model)



