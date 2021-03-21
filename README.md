# A Transformer implementation using Tensorflow

## References
* Tensorflow tutorial: Transformer model for language understanding
    * https://www.tensorflow.org/tutorials/text/transformer#encoder_layer

## Features
* Transformer components are written with Keras Layer and Model for re-usability
* Savable & loadable via Tensorflow's saved model format
* A pre-trained Portuguese to English translation model available: See evaluate.py for more detail

## Usages
* Run train.py to train a Portuguese to English translation model. The trained model is going to be saved at "./pt2en.tf"
* Run evaluate.py to test pt2en translation model