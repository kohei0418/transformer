import tensorflow as tf

from downloader import get_tokenizers
from mask import create_masks
from optimizer import CustomSchedule
from transformer import Transformer

tokenizers = get_tokenizers()

transformer = Transformer(
    num_layers=4,
    d_model=128,
    num_heads=8,
    dff=512,
    input_vocab_size=tokenizers.pt.get_vocab_size(),
    target_vocab_size=tokenizers.en.get_vocab_size(),
    input_max_len=1000,
    target_max_len=1000,
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

        predictions = transformer(inputs=(encoder_input, output, enc_padding_mask, combined_mask, dec_padding_mask))
        predictions = predictions[:, -1:, :]
        predicted_id = tf.argmax(predictions, axis=-1)
        output = tf.concat([output, predicted_id], axis=-1)

        if predicted_id == end:
            break

    text = tokenizers.en.detokenize(output)[0]
    tokens = tokenizers.en.lookup(output)[0]

    return text, tokens


if __name__ == '__main__':
    sentence = 'este é o primeiro livro que eu fiz.'
    ground_truth = "this is the first book i've ever done."
    translated_text, translated_tokens, attention_weights = evaluate(sentence)
    print(sentence)
    print(translated_text.numpy().decode('utf-8'))
    print(ground_truth)
