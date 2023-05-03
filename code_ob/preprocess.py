import pickle
import random
import re
from PIL import Image
import tensorflow as tf
import numpy as np
import collections
from tqdm import tqdm
import cv2
import tensorflow as tf
from keras.layers import \
       Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization

from keras.optimizers import SGD
from keras.losses import SparseCategoricalCrossentropy


class VGGModel(tf.keras.Model):
    def __init__(self):
        super(VGGModel, self).__init__()

        self.optimizer = SGD(learning_rate=1e-3)

        self.vgg16 = [
            # Block 1
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv1"),
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv2"),
            MaxPool2D(2, name="block1_pool"),
            # Block 2
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv1"),
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv2"),
            MaxPool2D(2, name="block2_pool"),
            # Block 3
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv1"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv2"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv3"),
            MaxPool2D(2, name="block3_pool"),
            # Block 4
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv3"),
            MaxPool2D(2, name="block4_pool"),
            # Block 5
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv3"),
            MaxPool2D(2, name="block5_pool")
        ]
        for layer in self.vgg16:
            layer.trainable = False
        self.vgg16 = tf.keras.Sequential(self.vgg16, name="vgg_base")

    def call(self, x):
        """ Passes the image through the network. """
        x = self.vgg16(x)
        return x


def preprocess_captions(captions, window_size):
    """ This preprocesses the captions to be in a usable form for
    training.

    1. Convert the caption to lowercase, and then remove all special characters from it.
    2. Split the caption into separate words, and collect all words which are more than 
       one character and which contain only alphabets (ie. discard words with mixed alpha-numerics).
    3. Join those words into a string.
    4. Replace the old caption in the captions list with this new cleaned caption

    https://towardsdatascience.com/image-captions-with-attention-in-tensorflow-step-by-step-927dad3569fa
    """

    for step, caption in enumerate(captions):
        caption_nopunct = re.sub(r"[^a-zA-Z0-9]+", ' ', caption.lower())
        clean_words = [word for word in caption_nopunct.split() if ((len(word) > 1) and (word.isalpha()))]
        caption_new = ['<start>'] + clean_words[:window_size-1] + ['<end>'] 
        captions[step] = caption_new


def get_image_features(image_names, dir, feature_fn, vis_subset=100):
    """ This method extracts the features from the images using the given feature_fn.
    """

    image_features = []
    vis_images = []
    gap = tf.keras.layers.GlobalAveragePooling2D()
    pbar = tqdm(image_names)
    for i, image_name in enumerate(pbar):
        img_path = f'{dir}/Images/{image_name}'
        pbar.set_description(f"[({i+1}/{len(image_names)})] Processing '{img_path}' into GAP Vector")
        with Image.open(img_path) as img:
            img_array = np.array(img.resize((224,224)))
        img_in = tf.keras.applications.resnet50.preprocess_input(img_array)[np.newaxis, :]
        image_features += [gap(feature_fn(img_in))]
        if i < vis_subset:
            vis_images += [img_array]
    print()
    return image_features, vis_images


def get_image_feature(image_path, feature_fn):
    """ This method extracts the features from the images using
    either resnet50 or vgg16. 
    """

    with Image.open(image_path) as img:
        img_array = np.array(img.resize((224,224)))

    # convert to 3 channels if grayscale
    if (len(img_array.shape) < 3):  
        img_array = cv2.cvtColor(img_array[...,np.newaxis], cv2.COLOR_GRAY2RGB)

    img_in = tf.keras.applications.resnet50.preprocess_input(img_array)[np.newaxis, :]

    gap = tf.keras.layers.GlobalAveragePooling2D()
    image_features = gap(feature_fn(img_in))
    vis_images = img_array

    return np.array(image_features).flatten(), np.array(vis_images)


def preprocess(dir, feature_fn):
    """ This is the main function that preprocesses the input data.

    This may need some tweaking to use other datasets.
    """

    text_file_path = f'{dir}/captions.txt'

    with open(text_file_path) as file:
        examples = file.read().splitlines()[1:]
    
    # map each image name to a list containing all 5 of its captons
    image_names_to_captions = {}
    for example in examples:
        img_name, caption = example.split(',', 1)
        image_names_to_captions[img_name] = image_names_to_captions.get(img_name, []) + [caption]

    # randomly split examples into training and testing sets
    shuffled_images = list(image_names_to_captions.keys())
    random.seed(0)
    random.shuffle(shuffled_images)
    test_image_names = shuffled_images[:1000]
    train_image_names = shuffled_images[1000:]

    def get_all_captions(image_names):
        to_return = []
        for image in image_names:
            captions = image_names_to_captions[image]
            for caption in captions:
                to_return.append(caption)
        return to_return
    

    # get lists of all the captions in the train and testing set
    train_captions = get_all_captions(train_image_names)
    test_captions = get_all_captions(test_image_names)

    # remove special charachters and other nessesary preprocessing
    window_size = 20
    preprocess_captions(train_captions, window_size)
    preprocess_captions(test_captions, window_size)

    # count word frequencies and replace rare words with '<unk>'
    word_count = collections.Counter()
    for caption in train_captions:
        word_count.update(caption)

    def unk_captions(captions, minimum_frequency):
        for caption in captions:
            for index, word in enumerate(caption):
                if word_count[word] <= minimum_frequency:
                    caption[index] = '<unk>'


    unk_captions(train_captions, 50)
    unk_captions(test_captions, 50)

    # pad captions so they all have equal length
    def pad_captions(captions, window_size):
        for caption in captions:
            caption += (window_size + 1 - len(caption)) * ['<pad>'] 
    

    pad_captions(train_captions, window_size)
    pad_captions(test_captions,  window_size)

    # assign unique ids to every work left in the vocabulary
    word2idx = {}
    vocab_size = 0
    for caption in train_captions:
        for index, word in enumerate(caption):
            if word in word2idx:
                caption[index] = word2idx[word]
            else:
                word2idx[word] = vocab_size
                caption[index] = vocab_size
                vocab_size += 1
    for caption in test_captions:
        for index, word in enumerate(caption):
            caption[index] = word2idx[word] 
    
    # use ResNet50 to extract image features
    print("Getting training embeddings")
    train_image_features, train_images = get_image_features(train_image_names, dir, feature_fn)
    print("Getting testing embeddings")
    test_image_features,  test_images  = get_image_features(test_image_names, dir, feature_fn)

    return dict(
        train_captions          = np.array(train_captions),
        test_captions           = np.array(test_captions),
        train_image_features    = np.array(train_image_features),
        test_image_features     = np.array(test_image_features),
        train_images            = np.array(train_images),
        test_images             = np.array(test_images),
        word2idx                = word2idx,
        idx2word                = {v:k for k,v in word2idx.items()},
    )


if __name__ == "__main__":
    """ When this file is run, it will preprocess images in the specified
    foler using an accompanied captions.txt file. 

    There may be some issues with this and using different file formats.
    """

    # # resnet
    # feature_fn = tf.keras.applications.ResNet50(False)  ## Produces Bx7x7x2048

    # vgg
    img_in_shape = (1, 224, 224, 3)
    model = VGGModel()
    path = "../code/vgg16_imagenet.h5"
    model.build(img_in_shape)
    model.vgg16.load_weights(path, by_name=True)
    feature_fn = model

    dir = "../data/data_vgg.p"

    with open(f'{dir}', 'wb') as pickle_file:
        pickle.dump(preprocess(dir, feature_fn), pickle_file)
    print(f'Data has been dumped!')