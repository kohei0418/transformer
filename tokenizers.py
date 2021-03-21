
import tensorflow as tf


def get():
    import tensorflow_text as text
    model_name = "ted_hrlr_translate_pt_en_converter"
    tf.keras.utils.get_file(
        f"{model_name}.zip",
        f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip",
        cache_dir='.', cache_subdir='', extract=True
    )
    return tf.saved_model.load(model_name)


def get_vocab_sizes():
    tokenizers = get()
    return tokenizers.pt.get_vocab_size().numpy(), tokenizers.en.get_vocab_size().numpy()
