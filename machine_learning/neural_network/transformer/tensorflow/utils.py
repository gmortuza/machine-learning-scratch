import tensorflow as tf


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, dtype=tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        args1 = tf.math.rsqrt(step)
        args2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.minimum(args1, args2)


# Loss
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")


def loss_function(real, pred):
    loss_ = loss_object(real, pred)
    mask = tf.cast(tf.math.logical_not(tf.math.equal(real, 0)), dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracy = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracy = tf.math.logical_and(accuracy, mask)

    mask = tf.cast(mask, dtype=tf.float32)
    accuracy = tf.cast(accuracy, dtype=tf.float32)

    return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), dtype=tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def get_mask(inp, tar):
    encoding_padding_mask = create_padding_mask(inp)

    decoding_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return encoding_padding_mask, combined_mask, decoding_padding_mask
