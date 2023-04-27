import tensorflow as tf
from keras.layers import \
       Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization

from keras.optimizers import SGD
from keras.losses import SparseCategoricalCrossentropy

import hyperparameters_obj as hp

#THE CODE BELOW IS JUST A PLACEHOLDER, PLEASE EDIT AS YOU LIKE


class YourModel_obj(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(YourModel_obj, self).__init__()

        self.optimizer = SGD(learning_rate=hp.learning_rate, momentum=hp.momentum)



        self.architecture = [
            Conv2D(10, (5, 5), activation='relu', input_shape=(None, None, 1)),
            MaxPool2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(32, activation='relu'),
            Dropout(0.25),
            Dense(15, activation='softmax')
        ]

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        lossFunction = SparseCategoricalCrossentropy()
        return lossFunction(labels,predictions)