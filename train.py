
import logging

import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model

from dataset import preprocess_dataset
from optimizer import CustomSchedule, masked_accuracy, masked_sparse_categorical_crossentropy
from tokenizers import get_vocab_sizes
from transformer import Transformer

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings


def main():
    train_batches, val_batches = preprocess_dataset()
    input_vocab_size, target_vocab_size = get_vocab_sizes()

    d_model = 128
    num_layers = 4
    num_heads = 8
    dff = 512
    dropout_rate = 0.1

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    source = Input(shape=[None], dtype=tf.int64, name='source')
    target = Input(shape=[None], dtype=tf.int64, name='target')
    inputs = [source, target]

    transformer_out = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
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
