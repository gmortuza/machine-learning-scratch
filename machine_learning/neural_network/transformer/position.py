import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def get_angle(pos, i, d_model):
    return pos / np.power(10000, 2*i/np.float32(d_model))


def get_position_encoding(position, d_model):
    angle_rad = get_angle(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    # apply sin to even indices
    angle_rad[:, 0::2] = np.sin(angle_rad[:, 0::2])
    # apply cos to odd indices
    angle_rad[:, 1::2] = np.sin(angle_rad[:, 1::2])

    pos_encoding = angle_rad[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


if __name__ == '__main__':
    n, d = 2048, 512
    position_encoding = get_position_encoding(n, d)
    print(position_encoding.shape)
    pos_encoding = position_encoding[0]

    # Juggle the dimensions for the plot
    pos_encoding = tf.reshape(pos_encoding, (n, d // 2, 2))
    pos_encoding = tf.transpose(pos_encoding, (2, 1, 0))
    pos_encoding = tf.reshape(pos_encoding, (d, n))

    plt.pcolormesh(pos_encoding, cmap='RdBu')
    plt.ylabel('Depth')
    plt.xlabel('Position')
    plt.colorbar()
    plt.show()
