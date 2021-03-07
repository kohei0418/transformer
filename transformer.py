import collections
import logging
import os
import pathlib
import re
import string
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf

from mask import create_masks

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings


def positional_encoding(position: int, d_model: int) -> tf.Tensor:
    def get_angles(pos: np.ndarray, i: np.ndarray, dim: int) -> np.ndarray:
        angle_rates = 1 / np.power(10_000, (2 * (i//2)) / np.float32(dim))
        return pos * angle_rates

    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def _split_heads(self, x: tf.Tensor, batch_size: int):
        x = tf.reshape(x, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=(0, 2, 1, 3))

    def _scaled_dot_product_attention(self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, mask: tf.Tensor):
        qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

    def call(self, v: tf.Tensor, k: tf.Tensor, q: tf.Tensor, mask: tf.Tensor):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self._split_heads(q, batch_size)
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)

        scaled_attention, attention_weights = self._scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)

        return output, attention_weights


def point_wise_ff_network(d_model: int, dff: int) -> tf.keras.Model:
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_ff_network(d_model, dff)

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, x: tf.Tensor, training: bool, mask: tf.Tensor):
        attention, _ = self.mha(x, x, x, mask)
        attention = self.dropout(attention, training=training)
        out1 = self.norm1(x + attention)

        ffn_out = self.ffn(out1)
        ffn_out = self.dropout(ffn_out, training=training)
        out2 = self.norm2(out1 + ffn_out)

        return out2


class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_ff_network(d_model, dff)

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, x: tf.Tensor, enc_output: tf.Tensor, training: bool, look_ahead_mask: tf.Tensor, padding_mask: tf.Tensor):
        attention1, attention_weights1 = self.mha1(x, x, x, look_ahead_mask)
        attention1 = self.dropout(attention1, training=training)
        out1 = self.norm1(attention1 + x)

        attention2, attention_weights2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attention2 = self.dropout(attention2, training=training)
        out2 = self.norm2(attention2 + out1)

        ffn_out = self.ffn(out2)
        ffn_out = self.dropout(ffn_out, training=training)
        out3 = self.norm3(ffn_out + out2)

        return out3, attention_weights1, attention_weights2


class Encoder(tf.keras.layers.Layer):

    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int, input_vocab_size: int,
                 maximum_position_encoding: int, dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_dim=input_vocab_size, output_dim=d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x: tf.Tensor, training: bool, mask: tf.Tensor):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x


class Decoder(tf.keras.layers.Layer):

    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int, target_vocab_size: int,
                 maximum_position_encoding: int, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_dim=target_vocab_size, output_dim=d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x: tf.Tensor, enc_output: tf.Tensor, training: bool, look_ahead_mask: tf.Tensor, padding_mask: tf.Tensor):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, b1, b2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights[f'dec{i+1}_block1'] = b1
            attention_weights[f'dec{i+1}_block2'] = b2

        return x, attention_weights


class Transformer(tf.keras.Model):

    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int,
                 input_vocab_size: int, target_vocab_size: int, pe_input: int, pe_target: int, dropout_rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, dropout_rate)
        self.dense = tf.keras.layers.Dense(target_vocab_size)

    def call(self, input: tf.Tensor, target: tf.Tensor, training: bool, enc_padding_mask: tf.Tensor, look_ahead_mask: tf.Tensor, dec_padding_mask: tf.Tensor):
        enc_output = self.encoder(input, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(target, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.dense(dec_output)

        return final_output, attention_weights


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
    # prepare datasets
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                                   as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']

    # prepare tokenizers
    model_name = "ted_hrlr_translate_pt_en_converter"
    tf.keras.utils.get_file(
        f"{model_name}.zip",
        f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
        cache_dir='.', cache_subdir='', extract=True
    )
    tokenizers = tf.saved_model.load(model_name)

    for pt_examples, en_examples in train_examples.batch(3).take(1):
        for pt in pt_examples.numpy():
            print(pt.decode('utf-8'))

        print()

        for en in en_examples.numpy():
            print(en.decode('utf-8'))

    def tokenize_pairs(pt, en):
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
        pe_input=1000,
        pe_target=1000,
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
            pred, _ = transformer(input, target_input, True, enc_padding_mask, combined_mask, dec_padding_mask)
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
