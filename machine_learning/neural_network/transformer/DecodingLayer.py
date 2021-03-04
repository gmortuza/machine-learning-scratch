import tensorflow as tf
from MultiHeadAttention import MultiHeadAttention
from EncodingLayer import point_wise_feed_forward_network


class DecodingLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=.1):
        super(DecodingLayer, self).__init__()

        self.mha_1 = MultiHeadAttention(d_model, num_heads)
        self.mha_2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        # Normalization
        self.layer_norm_1 = tf.keras.layers.Layer()
        self.layer_norm_2 = tf.keras.layers.Layer()
        self.layer_norm_3 = tf.keras.layers.Layer()

        # Dropout
        self.dropout_1 = tf.keras.layers.Dropout(rate)
        self.dropout_2 = tf.keras.layers.Dropout(rate)
        self.dropout_3 = tf.keras.layers.Dropout(rate)

    def call(self, x, encoding_out, is_training, look_ahead_mask, padding_mask):
        attn_1, attn_weights_block_1 = self.mha_1(x, x, x, look_ahead_mask)
        attn_1 = self.dropout_1(attn_1)
        out_1 = self.layer_norm_1(x + attn_1)

        attn_2, attn_weights_block_2 = self.mha_2(encoding_out, encoding_out, out_1, padding_mask)
        attn_2 = self.dropout_2(attn_2)
        out_2 = self.layer_norm_2(out_1 + attn_2)

        ffn_output = self.ffn(out_2)
        ffn_output = self.dropout_3(ffn_output)
        out_3 = self.layer_norm_3(out_2 + ffn_output)

        return out_3, attn_weights_block_1, attn_weights_block_2


if __name__ == '__main__':
    pass
