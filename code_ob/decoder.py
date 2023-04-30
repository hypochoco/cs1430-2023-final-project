import tensorflow as tf
from transformer import TransformerBlock, PositionalEncoding


class TransformerDecoder(tf.keras.Model):
    """ temp
    """
    
    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):
        """ temp
        """

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # init layers
        self.image_embedding = tf.keras.layers.Dense(self.hidden_size)
        self.encoding = PositionalEncoding(self.vocab_size, self.hidden_size, self.window_size)
        self.decoder = TransformerBlock(self.hidden_size)
        self.classifier = tf.keras.layers.Dense(self.vocab_size)


    def call(self, encoded_images, captions):
        """ temp
        """

        encoded_images = tf.expand_dims(encoded_images, axis=1)
        embedded_images = self.image_embedding(encoded_images)
        pos_encoding = self.encoding(captions)
        results = self.decoder(pos_encoding, embedded_images)

        probs = self.classifier(results)
        return probs