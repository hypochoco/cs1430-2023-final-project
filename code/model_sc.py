import tensorflow as tf
from keras.layers import \
       Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization

from keras.optimizers import SGD
from keras.losses import SparseCategoricalCrossentropy

import hyperparameters_sc as hp


class YourModel_sc(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(YourModel_sc, self).__init__()

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


# class VGGModel_sc(tf.keras.Model):
#     def __init__(self):
#         super(VGGModel_sc, self).__init__()

#         # TASK 3
#         # TODO: Select an optimizer for your network (see the documentation
#         #       for tf.keras.optimizers)

#        #Add in momentum?
#         self.optimizer = SGD(learning_rate=hp.learning_rate)

#         # Don't change the below:

#         self.vgg16 = [
#             # Block 1
#             Conv2D(64, 3, 1, padding="same",
#                    activation="relu", name="block1_conv1"),
#             Conv2D(64, 3, 1, padding="same",
#                    activation="relu", name="block1_conv2"),
#             MaxPool2D(2, name="block1_pool"),
#             # Block 2
#             Conv2D(128, 3, 1, padding="same",
#                    activation="relu", name="block2_conv1"),
#             Conv2D(128, 3, 1, padding="same",
#                    activation="relu", name="block2_conv2"),
#             MaxPool2D(2, name="block2_pool"),
#             # Block 3
#             Conv2D(256, 3, 1, padding="same",
#                    activation="relu", name="block3_conv1"),
#             Conv2D(256, 3, 1, padding="same",
#                    activation="relu", name="block3_conv2"),
#             Conv2D(256, 3, 1, padding="same",
#                    activation="relu", name="block3_conv3"),
#             MaxPool2D(2, name="block3_pool"),
#             # Block 4
#             Conv2D(512, 3, 1, padding="same",
#                    activation="relu", name="block4_conv1"),
#             Conv2D(512, 3, 1, padding="same",
#                    activation="relu", name="block4_conv2"),
#             Conv2D(512, 3, 1, padding="same",
#                    activation="relu", name="block4_conv3"),
#             MaxPool2D(2, name="block4_pool"),
#             # Block 5
#             Conv2D(512, 3, 1, padding="same",
#                    activation="relu", name="block5_conv1"),
#             Conv2D(512, 3, 1, padding="same",
#                    activation="relu", name="block5_conv2"),
#             Conv2D(512, 3, 1, padding="same",
#                    activation="relu", name="block5_conv3"),
#             MaxPool2D(2, name="block5_pool")
#         ]

#         # TASK 3
#         # TODO: Make all layers in self.vgg16 non-trainable. This will freeze the
#         #       pretrained VGG16 weights into place so that only the classificaiton
#         #       head is trained.

#         for layer in self.vgg16:
#             layer.trainable = False

#         # TODO: Write a classification head for our 15-scene classification task.

#        #Add names?
#         self.head = [
#             Flatten(),
#             Dense(512, activation="relu"),
#             BatchNormalization(),
#             Dropout(0.5),
#             Dense(15, activation="softmax")
#         ]

#         # Don't change the below:
#         self.vgg16 = tf.keras.Sequential(self.vgg16, name="vgg_base")
#         self.head = tf.keras.Sequential(self.head, name="vgg_head")

#     def call(self, x):
#         """ Passes the image through the network. """

#         x = self.vgg16(x)
#         x = self.head(x)

#         return x

#     @staticmethod
#     def loss_fn(labels, predictions):
#         """ Loss function for model. """

#         # TASK 3
#         # TODO: Select a loss function for your network (see the documentation
#         #       for tf.keras.losses)
#         #       Read the documentation carefully, some might not work with our 
#         #       model!

#         lossFunction = SparseCategoricalCrossentropy()
#         return lossFunction(labels, predictions)