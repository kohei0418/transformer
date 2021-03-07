
import logging
import time

import tensorflow as tf
import tensorflow_datasets as tfds

from downloader import get_tokenizers
from mask import create_masks
from transformer import Transformer

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model: int, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def main():
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                                   as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']

    tokenizers = get_tokenizers()

    def tokenize_pairs(pt: tf.Tensor, en: tf.Tensor):
        pt = tokenizers.pt.tokenize(pt)
        en = tokenizers.en.tokenize(en)
        return pt.to_tensor(), en.to_tensor()

    def make_batches(ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.cache().shuffle(20_000).batch(64).map(tokenize_pairs, tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    train_batches = make_batches(train_examples)
    val_batches = make_batches(val_examples)

    d_model = 128
    num_layers = 4
    num_heads = 8
    dff = 512
    dropout_rate = 0.1

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=tokenizers.pt.get_vocab_size(),
        target_vocab_size=tokenizers.en.get_vocab_size(),
        input_max_len=1000,
        target_max_len=1000,
        dropout_rate=dropout_rate,
    )

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    def loss_func(real, pred):
        loss = loss_object(real, pred)
        mask = tf.cast(tf.math.logical_not(tf.math.equal(real, 0)), dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def accuracy_func(real, pred):
        acc = tf.equal(real, tf.argmax(pred, axis=2))
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        acc = tf.math.logical_and(mask, acc)

        return tf.reduce_sum(tf.cast(acc, dtype=tf.float32)) / tf.reduce_sum(tf.cast(mask, dtype=tf.float32))

    checkpoint_path = './checkpoints/train'
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(input: tf.Tensor, target: tf.Tensor):
        target_input = target[:, :-1]
        target_real = target[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(input, target_input)

        with tf.GradientTape() as tape:
            pred, _ = transformer(inputs=(input, target_input, enc_padding_mask, combined_mask, dec_padding_mask), training=True)
            loss = loss_func(target_real, pred)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy_func(target_real, pred))

    for epoch in range(20):
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (input, target)) in enumerate(train_batches):
            train_step(input, target)
            if batch % 50 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')


if __name__ == '__main__':
    main()
