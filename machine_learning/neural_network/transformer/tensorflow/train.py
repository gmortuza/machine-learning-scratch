import time
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 1:
    # Using the second GPU
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
import matplotlib.pyplot as plt
from data import get_data
from Transformer import Transformer
from utils import CustomSchedule, loss_function, accuracy_function, get_mask

train_ds, valid_ds, test_ds, tokenizer = get_data()

#### =========== SET HYPER_PARAMETER ============ ####
num_layer = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = .1
epochs = 0
check_point_path = "./checkpoint/train"

# Optimizer
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=.9, beta_2=.98, epsilon=1e-9)

train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")

transformer = Transformer(num_layer, d_model, num_heads, dff, tokenizer.pt.get_vocab_size(),
                          tokenizer.en.get_vocab_size(), input_=1000, target_=1000, rate=dropout_rate)

ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, check_point_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Latest checkpoint restored")

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64)
]


# @tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, look_ahead_mask, dec_padding_mask = get_mask(inp, tar_inp)

    with tf.GradientTape() as tape:
        prediction, _ = transformer(inp, tar_inp, True, enc_padding_mask, look_ahead_mask, dec_padding_mask)

        loss = loss_function(tar_real, prediction)

    gradient = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradient, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, prediction))


# Do the training
for epoch in range(epochs):
    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()
    for (batch, (inp, tar)) in enumerate(train_ds):
        train_step(inp, tar)

        if batch % 50 == 0:
            print(f"Epoch {epoch + 1} batch {batch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}")
    if (epoch + 1) % 5 == 0:
        chkpt_save_path = ckpt_manager.save()
        print(f"Saving checkpoint for epoch {epoch + 1} at {chkpt_save_path}")

    print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')


