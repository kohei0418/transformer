import tensorflow as tf

from downloader import get_tokenizers

tokenizers = get_tokenizers()
saved_model = tf.saved_model.load('pt2en.tf')
transformer = saved_model.signatures['serving_default']


def evaluate(sentence: str, max_length=40):
    source = tf.convert_to_tensor([sentence])
    source = tokenizers.pt.tokenize(source).to_tensor()

    start, end = tokenizers.en.tokenize([''])[0]
    target = tf.convert_to_tensor([start])
    target = tf.expand_dims(target, 0)

    for i in range(max_length):
        predictions = transformer(source=source, target=target)['transformer']
        predictions = predictions[:, -1:, :]
        predicted_id = tf.argmax(predictions, axis=-1)
        target = tf.concat([target, predicted_id], axis=-1)

        if predicted_id == end:
            break

    text = tokenizers.en.detokenize(target)[0]
    tokens = tokenizers.en.lookup(target)[0]

    return text, tokens


if __name__ == '__main__':
    sentence = 'este é o primeiro livro que eu fiz.'
    ground_truth = "this is the first book i've ever done."
    translated_text, translated_tokens = evaluate(sentence)
    print(sentence)
    print(translated_text.numpy().decode('utf-8'))
    print(ground_truth)
