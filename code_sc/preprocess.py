import pickle
import random
import re
from PIL import Image
import tensorflow as tf
import numpy as np
import collections
from tqdm import tqdm


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


def get_image_features(image_names, dir, vis_subset=100):
    """ This method extracts the features from the images using Resnet50.
    
    This function could also be easily modified to use vgg.
    """

    image_features = []
    vis_images = []
    resnet = tf.keras.applications.ResNet50(False)  ## Produces Bx7x7x2048
    gap = tf.keras.layers.GlobalAveragePooling2D()  ## Produces Bx2048
    pbar = tqdm(image_names)
    for i, image_name in enumerate(pbar):
        img_path = f'{dir}/Images/{image_name}'
        pbar.set_description(f"[({i+1}/{len(image_names)})] Processing '{img_path}' into 2048-D ResNet GAP Vector")
        with Image.open(img_path) as img:
            img_array = np.array(img.resize((224,224)))
        img_in = tf.keras.applications.resnet50.preprocess_input(img_array)[np.newaxis, :]
        image_features += [gap(resnet(img_in))]
        if i < vis_subset:
            vis_images += [img_array]
    print()
    return image_features, vis_images


def preprocess(dir):
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
    train_image_features, train_images = get_image_features(train_image_names, dir)
    print("Getting testing embeddings")
    test_image_features,  test_images  = get_image_features(test_image_names, dir)

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

    dir = ""
    with open(f'{dir}/data.p', 'wb') as pickle_file:
        pickle.dump(preprocess(dir), pickle_file)
    print(f'Data has been dumped into {dir}/data.p!')
