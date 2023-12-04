"""
Implementation of figure 2 of the paper
"""
import tensorflow as tf


def get_scaled_dot_product_attention(q, k, v, mask):
    """
    Implementation of fig 2(left) of original paper: https://arxiv.org/abs/1706.03762

    :param q:
    :param k:
    :param v:
    :param mask:
    :return:
    """
    dot_prod = tf.matmul(q, k, transpose_b=True)
    # scaled
    dot_prod_scaled = dot_prod / tf.math.sqrt(tf.cast(k.shape[-1], dtype=tf.float32))  # (batch_size, num_heads, seq_len, seq_len)
    # masked (optional)
    if mask is not None:
        dot_prod_scaled += (mask * -1e9)
    attention_weights = tf.nn.softmax(dot_prod_scaled, axis=-1)

    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.wq = tf.keras.layers.Dense(self.d_model)
        self.wk = tf.keras.layers.Dense(self.d_model)
        self.wv = tf.keras.layers.Dense(self.d_model)

        self.dense = tf.keras.layers.Dense(self.d_model)

    def split_head(self, x, batch_size):
        # (batch_size, seq_len, d_model) --> (batch_size, num_heads, seq_len, depth)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_model // self.num_heads))
        return tf.transpose(x, perm=(0, 2, 1, 3))

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # batch_size, seq_len, d_model
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_head(q, batch_size)
        k = self.split_head(k, batch_size)
        v = self.split_head(v, batch_size)

        # Scaled Dot-Product
        scaled_attention, attention_weights = get_scaled_dot_product_attention(q, k, v, mask)

        # Concat scaled attention
        # (batch_size, num_heads, seq_len, depth) --> (batch_size, seq_len, d_model)
        concat_attention = tf.transpose(scaled_attention, perm=(0, 2, 1, 3))  # (bs, seq_len, num_heads, depth)
        concat_attention = tf.reshape(concat_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)

        return output, attention_weights







