
import tensorflow as tf
from tensorflow.python.keras.losses import sparse_categorical_crossentropy


def masked_sparse_categorical_crossentropy(y_true, y_pred):
    loss = sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    mask = tf.cast(tf.math.logical_not(tf.math.equal(y_true, 0)), dtype=loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def masked_accuracy(y_true, y_pred):
    acc = tf.equal(tf.cast(y_true, dtype=tf.int64), tf.argmax(y_pred, axis=2))
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    acc = tf.math.logical_and(mask, acc)

    return tf.reduce_sum(tf.cast(acc, dtype=tf.float32)) / tf.reduce_sum(tf.cast(mask, dtype=tf.float32))


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model: int, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            'd_model': int(self.d_model.numpy()),
            'warmup_steps': self.warmup_steps,
        }
