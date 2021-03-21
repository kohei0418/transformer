from typing import Tuple

import tensorflow as tf
import tensorflow_datasets as tfds

from tokenizers import get


def preprocess_dataset() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                                   as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']

    tokenizers = get()

    def tokenize_and_create_masks(pt: tf.Tensor, en: tf.Tensor):
        pt: tf.Tensor = tokenizers.pt.tokenize(pt).to_tensor()
        en: tf.Tensor = tokenizers.en.tokenize(en).to_tensor()
        target_input = en[:, :-1]
        target_real = en[:, 1:]
        return (pt, target_input), target_real

    def make_batches(ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.cache().shuffle(20_000).batch(64).map(tokenize_and_create_masks, tf.data.AUTOTUNE).prefetch(
            tf.data.AUTOTUNE)

    train_batches = make_batches(train_examples)
    val_batches = make_batches(val_examples)

    tf.data.experimental.save(train_batches, 'dataset/train')
    tf.data.experimental.save(val_batches, 'dataset/val')

    tensor_spec = tf.TensorSpec(shape=(None, None), dtype=tf.int64)
    element_spec = ((tensor_spec, tensor_spec), tensor_spec)
    train_batches = tf.data.experimental.load('dataset/train', element_spec)
    val_batches = tf.data.experimental.load('dataset/val', element_spec)

    return train_batches, val_batches
