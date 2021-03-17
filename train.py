
import logging

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model

from downloader import get_tokenizers
from mask import create_masks
from optimizer import CustomSchedule, masked_accuracy, masked_sparse_categorical_crossentropy
from transformer import Transformer

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings


def main():
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                                   as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']

    tokenizers = get_tokenizers()

    def tokenize_and_create_masks(pt: tf.Tensor, en: tf.Tensor):
        pt: tf.Tensor = tokenizers.pt.tokenize(pt).to_tensor()
        en: tf.Tensor = tokenizers.en.tokenize(en).to_tensor()
        target_input = en[:, :-1]
        target_real = en[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(pt, target_input)
        return (pt, target_input, enc_padding_mask, combined_mask, dec_padding_mask), target_real

    def make_batches(ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.cache().shuffle(20_000).batch(64).map(tokenize_and_create_masks, tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    train_batches = make_batches(train_examples)
    val_batches = make_batches(val_examples)

    d_model = 128
    num_layers = 4
    num_heads = 8
    dff = 512
    dropout_rate = 0.1

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    source = Input(shape=[None], dtype=tf.int64)
    target = Input(shape=[None], dtype=tf.int64)
    source_pad_mask = Input(shape=[1, 1, None], dtype=tf.float32)
    look_ahead_mask = Input(shape=[1, None, None], dtype=tf.float32)
    target_pad_mask = Input(shape=[1, 1, None], dtype=tf.float32)
    inputs = [source, target, source_pad_mask, look_ahead_mask, target_pad_mask]

    transformer_out = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
        target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
        input_max_len=1000,
        target_max_len=1000,
        dropout_rate=dropout_rate,
    )(inputs)
    transformer = Model(inputs=inputs, outputs=transformer_out)
    transformer.compile(
        optimizer=optimizer,
        loss=masked_sparse_categorical_crossentropy,
        metrics=[masked_accuracy],
    )
    print(transformer.summary())

    checkpoint_path = './checkpoints/train'
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    class Checkpointer(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 5 == 0:
                ckpt_save_path = ckpt_manager.save()
                print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

    transformer.fit(
        x=train_batches,
        epochs=20,
        validation_data=val_batches,
        callbacks=[Checkpointer()]
    )

    transformer.save('pt2en.tf')


if __name__ == '__main__':
    main()
