import logging

import numpy as np
import tensorflow as tf

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

    @staticmethod
    def _scaled_dot_product_attention(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, mask: tf.Tensor):
        qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

    def call(self, inputs, **kwargs):
        q, k, v, mask = inputs
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

    def get_config(self):
        return {
            'd_model': self.d_model,
            'num_heads': self.num_heads,
        }


def point_wise_ff_network(d_model: int, dff: int) -> tf.keras.Model:
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.dff = dff
        self.dropout_rate = dropout_rate

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_ff_network(d_model, dff)

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, inputs, **kwargs):
        x, mask = inputs
        training: bool = True if kwargs.get('training') else False
        attention, _ = self.mha(inputs=(x, x, x, mask))
        attention = self.dropout(attention, training=training)
        out1 = self.norm1(x + attention)

        ffn_out = self.ffn(out1)
        ffn_out = self.dropout(ffn_out, training=training)
        out2 = self.norm2(out1 + ffn_out)

        return out2

    def get_config(self):
        return {
            'd_model': self.mha.d_model,
            'num_heads': self.mha.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate,
        }


class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.dff = dff
        self.dropout_rate = dropout_rate

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_ff_network(d_model, dff)

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, inputs, **kwargs):
        x, enc_output, look_ahead_mask, padding_mask = inputs
        training: bool = True if kwargs.get('training') else False
        attention1, attention_weights1 = self.mha1(inputs=(x, x, x, look_ahead_mask))
        attention1 = self.dropout(attention1, training=training)
        out1 = self.norm1(attention1 + x)

        attention2, attention_weights2 = self.mha2(inputs=(out1, enc_output, enc_output, padding_mask))
        attention2 = self.dropout(attention2, training=training)
        out2 = self.norm2(attention2 + out1)

        ffn_out = self.ffn(out2)
        ffn_out = self.dropout(ffn_out, training=training)
        out3 = self.norm3(ffn_out + out2)

        return out3, attention_weights1, attention_weights2

    def get_config(self):
        return {
            'd_model': self.mha1.d_model,
            'num_heads': self.mha1.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate,
        }


class Encoder(tf.keras.layers.Layer):

    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int, vocab_size: int,
                 maximum_position_encoding: int, dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.vocab_size = vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.dropout_rate = dropout_rate

        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, **kwargs):
        x, mask = inputs
        training: bool = True if kwargs.get('training') else False
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](inputs=(x, mask), training=training)

        return x

    def get_config(self):
        return {
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'vocab_size': self.vocab_size,
            'maximum_position_encoding': self.maximum_position_encoding,
            'dropout_rate': self.dropout_rate,
        }


class Decoder(tf.keras.layers.Layer):

    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int, vocab_size: int,
                 maximum_position_encoding: int, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.vocab_size = vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.dropout_rate = dropout_rate

        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, **kwargs):
        x, enc_output, look_ahead_mask, padding_mask = inputs
        training: bool = True if kwargs.get('training') else False
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        return x

    def get_config(self):
        return {
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'vocab_size': self.vocab_size,
            'maximum_position_encoding': self.maximum_position_encoding,
            'dropout_rate': self.dropout_rate,
        }


class Transformer(tf.keras.Model):

    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int,
                 input_vocab_size: int, target_vocab_size: int, input_max_len: int, target_max_len: int, dropout_rate=0.1):
        super(Transformer, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.input_max_len = input_max_len
        self.target_max_len = target_max_len
        self.dropout_rate = dropout_rate

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, input_max_len, dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, target_max_len, dropout_rate)
        self.dense = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, **kwargs):
        x, target, enc_padding_mask, look_ahead_mask, dec_padding_mask = inputs
        training: bool = True if kwargs.get('training') else False
        enc_output = self.encoder(inputs=(x, enc_padding_mask), training=training)
        dec_output, _ = self.decoder(inputs=(target, enc_output, look_ahead_mask, dec_padding_mask), training=training)
        final_output = self.dense(dec_output)

        return final_output

    def get_config(self):
        return {
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'input_vocab_size': self.input_vocab_size,
            'target_vocab_size': self.target_vocab_size,
            'input_max_len': self.input_max_len,
            'target_max_len': self.target_max_len,
            'dropout_rate': self.dropout_rate,
        }