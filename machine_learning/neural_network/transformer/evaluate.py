import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 1:
    # Using the second GPU
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
import matplotlib.pyplot as plt
from data import tokenizer
from Transformer import Transformer
from utils import get_mask
from utils import CustomSchedule


#### =========== SET HYPER_PARAMETER ============ ####
num_layer = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = .1
check_point_path = "./checkpoint/train"

# Optimizer
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=.9, beta_2=.98, epsilon=1e-9)

# Load transformer
transformer = Transformer(num_layer, d_model, num_heads, dff, tokenizer.pt.get_vocab_size(),
                          tokenizer.en.get_vocab_size(), input_=1000, target_=1000, rate=dropout_rate)

ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, check_point_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Latest checkpoint restored")


def evaluate(sentence, max_length=10):
    sentence = tf.convert_to_tensor([sentence])
    sentence = tokenizer.pt.tokenize(sentence).to_tensor()

    encoder_input = sentence

    start, end = tokenizer.en.tokenize([''])[0]
    output = tf.convert_to_tensor([start])
    output = tf.expand_dims(output, 0)

    for i in range(max_length):
        enc_padding_mask, combined_mask, dec_padding_mask = get_mask(encoder_input, output)
        predictions, attention_weights = transformer(encoder_input, output, False, enc_padding_mask, combined_mask, dec_padding_mask)
        predictions = predictions[:, -1:, :]  # batch_size, 1, vocab_size

        predicted_id = tf.argmax(predictions, axis=-1)
        output = tf.concat([output, predicted_id], axis=-1)

        if predicted_id == end:
            break

    text = tokenizer.en.detokenize(output)[0]

    tokens = tokenizer.en.lookup(output)[0]

    return text, tokens, attention_weights


def plot_attention_head(in_token, translated_token, attention):
    # translated_tokens = translated_token[1:]

    ax = plt.gca()
    ax.matshow(attention)
    ax.set_xticks(range(len(in_token)))
    ax.set_yticks(range(len(translated_token[1:])))

    labels = [label.decode('utf-8') for label in in_token.numpy()]

    ax.set_xticklabels(labels, rotation=90)

    labels = [label.decode('utf-8') for label in translated_token[1:].numpy()]
    ax.set_yticklabels(labels)


def plot_attention_weight(sentence, translated_tokens, attention_heads):
    in_tokens = tf.convert_to_tensor([sentence])
    in_tokens = tokenizer.pt.tokenize(in_tokens).to_tensor()
    in_tokens = tokenizer.pt.lookup(in_tokens)[0]

    fig = plt.figure(figsize=(16, 8))

    for h, head in enumerate(attention_heads):
        ax = fig.add_subplot(2, 4, h+1)
        plot_attention_head(in_tokens, translated_tokens, head)
        ax.set_xlabel(f"Head {h}")

    plt.tight_layout()
    plt.show()


sentence = "Eu li sobre triceratops na enciclopédia."
ground_truth = "I read about triceratops in the encyclopedia."

translated_text, translated_tokens, attention_weights = evaluate(sentence)


plot_attention_weight(sentence, translated_tokens, attention_weights['decoder_layer_4_block_2'][0])

print(translated_text)
