
import matplotlib.pyplot as plt
import tensorflow as tf

from mask import create_masks
from transformer import Transformer, CustomSchedule

# prepare tokenizers
model_name = "ted_hrlr_translate_pt_en_converter"
tf.keras.utils.get_file(
    f"{model_name}.zip",
    f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
    cache_dir='.', cache_subdir='', extract=True
)
tokenizers = tf.saved_model.load(model_name)

transformer = Transformer(
    num_layers=4,
    d_model=128,
    num_heads=8,
    dff=512,
    input_vocab_size=tokenizers.pt.get_vocab_size(),
    target_vocab_size=tokenizers.en.get_vocab_size(),
    pe_input=1000,
    pe_target=1000,
    dropout_rate=0.1,
)
learning_rate = CustomSchedule(d_model=128)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

checkpoint_path = './checkpoints/train'
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Restored')


def evaluate(sentence: str, max_length=40):
    sentence = tf.convert_to_tensor([sentence])
    sentence = tokenizers.pt.tokenize(sentence).to_tensor()
    encoder_input = sentence

    start, end = tokenizers.en.tokenize([''])[0]
    output = tf.convert_to_tensor([start])
    output = tf.expand_dims(output, 0)

    for i in range(max_length):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        predictions, attention_weights = transformer(encoder_input, output, False, enc_padding_mask, combined_mask, dec_padding_mask)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.argmax(predictions, axis=-1)
        output = tf.concat([output, predicted_id], axis=-1)

        if predicted_id == end:
            break

    text = tokenizers.en.detokenize(output)[0]
    tokens = tokenizers.en.lookup(output)[0]

    return text, tokens, attention_weights


def plot_attention_head(input_tokens, translated_tokens, attention):
    translated_tokens = translated_tokens[1:]

    ax = plt.gca()
    ax.matshow(attention)
    ax.set_xticks(range(len(input_tokens)))
    ax.set_yticks(range(len(translated_tokens)))

    labels = [label.decode('utf-8') for label in input_tokens.numpy()]
    ax.set_xticklabels(labels, rotation=90)

    labels = [label.decode('utf-8') for label in translated_tokens.numpy()]
    ax.set_yticklabels(labels)


def plot_attention_weights(sentence, translated_tokens, attention_heads):
    input_tokens = tf.convert_to_tensor([sentence])
    input_tokens = tokenizers.pt.tokenize(input_tokens).to_tensor()
    input_tokens = tokenizers.pt.lookup(input_tokens)[0]

    fig = plt.figure(figsize=(16, 8))
    for h, head in enumerate(attention_heads):
        ax = fig.add_subplot(2, 4, h+1)
        plot_attention_head(input_tokens, translated_tokens, head)
        ax.set_xlabel(f'Head {h+1}')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    sentence = 'este é o primeiro livro que eu fiz.'
    ground_truth = "this is the first book i've ever done."
    translated_text, translated_tokens, attention_weights = evaluate(sentence)
    print(sentence)
    print(translated_text.numpy().decode('utf-8'))
    print(ground_truth)

    plot_attention_weights(sentence, translated_tokens, attention_weights['dec4_block2'][0])
