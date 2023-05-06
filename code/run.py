import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf
import pyttsx3 as tts

from PIL import Image

import hyperparameters_sc as hp_sc
import hyperparameters_obj as hp_obj
#Will have to rework this file to include which hyperparameters are being used
#Scene hyperparameters are the default right now


from model_sc import YourModel_sc, VGGModel_sc
from model_obj import YourModel_obj, VGGModel_obj

from preprocess_sc import Datasets_sc
from preprocess_obj import Datasets_obj

from skimage.transform import resize
from tensorboard_utils import \
        ImageLabelingLogger, ConfusionMatrixLogger, CustomModelSaver

from skimage.io import imread
# from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np

from main import generate_caption
import text_to_speech as speech

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--task',
        required=True,
        choices=['1','2' ,'3','4','5', '6'],
        help='''Which task of the assignment to run -
        training from scratch (1), or fine tuning VGG-16 (3).''')
    parser.add_argument(
        '--data',
        default='..'+os.sep+'data'+os.sep,
        help='Location where the dataset is stored.')
    parser.add_argument(
        '--imagePath',
        default='..'+os.sep+'image'+os.sep,
        help='Location where the input image is stored.')
    parser.add_argument(
        '--load-vgg',
        default='vgg16_imagenet.h5',
        help='''Path to pre-trained VGG-16 file (only applicable to
        task 3).''')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.''')
    parser.add_argument(
        '--confusion',
        action='store_true',
        help='''Log a confusion matrix at the end of each
        epoch (viewable in Tensorboard). This is turned off
        by default as it takes a little bit of time to complete.''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')
    # parser.add_argument(
    #     '--lime-image',
    #     default='test/Bedroom/image_0003.jpg',
    #     help='''Name of an image in the dataset to use for LIME evaluation.''')

    return parser.parse_args()


# def LIME_explainer(model, path, preprocess_fn, timestamp):
#     """
#     This function takes in a trained model and a path to an image and outputs 4
#     visual explanations using the LIME model
#     """

#     save_directory = "lime_explainer_images" + os.sep + timestamp
#     if not os.path.exists("lime_explainer_images"):
#         os.mkdir("lime_explainer_images")
#     if not os.path.exists(save_directory):
#         os.mkdir(save_directory)
#     image_index = 0

#     def image_and_mask(title, positive_only=True, num_features=5,
#                        hide_rest=True):
#         nonlocal image_index

#         temp, mask = explanation.get_image_and_mask(
#             explanation.top_labels[0], positive_only=positive_only,
#             num_features=num_features, hide_rest=hide_rest)
#         plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
#         plt.title(title)

#         image_save_path = save_directory + os.sep + str(image_index) + ".png"
#         plt.savefig(image_save_path, dpi=300, bbox_inches='tight')
#         plt.show()

#         image_index += 1

#     # Read the image and preprocess it as before
#     image = imread(path)
#     if len(image.shape) == 2:
#         image = np.stack([image, image, image], axis=-1)
#     image = resize(image, (hp_sc.img_size, hp_sc.img_size, 3), preserve_range=True)
#     image = preprocess_fn(image)
    

#     explainer = lime_image.LimeImageExplainer()

#     explanation = explainer.explain_instance(
#         image.astype('double'), model.predict, top_labels=5, hide_color=0,
#         num_samples=1000)

#     # The top 5 superpixels that are most positive towards the class with the
#     # rest of the image hidden
#     image_and_mask("Top 5 superpixels", positive_only=True, num_features=5,
#                    hide_rest=True)

#     # The top 5 superpixels with the rest of the image present
#     image_and_mask("Top 5 with the rest of the image present",
#                    positive_only=True, num_features=5, hide_rest=False)

#     # The 'pros and cons' (pros in green, cons in red)
#     image_and_mask("Pros(green) and Cons(red)",
#                    positive_only=False, num_features=10, hide_rest=False)

#     # Select the same class explained on the figures above.
#     ind = explanation.top_labels[0]
#     # Map each explanation weight to the corresponding superpixel
#     dict_heatmap = dict(explanation.local_exp[ind])
#     heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
#     plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
#     plt.colorbar()
#     plt.title("Map each explanation weight to the corresponding superpixel")

#     image_save_path = save_directory + os.sep + str(image_index) + ".png"
#     plt.savefig(image_save_path, dpi=300, bbox_inches='tight')
#     plt.show()


def train(model, datasets, checkpoint_path, logs_path, init_epoch, task):
    """ Training routine. """

    #Scene model
    if task == '1':
        # Keras callbacks for training
        callback_list = [
            tf.keras.callbacks.TensorBoard(
                log_dir=logs_path,
                update_freq='batch',
                profile_batch=0),
            ImageLabelingLogger(logs_path, datasets),
            CustomModelSaver(checkpoint_path, ARGS.task, hp_sc.max_num_weights)
        ]

        # Include confusion logger in callbacks if flag set
        if ARGS.confusion:
            callback_list.append(ConfusionMatrixLogger(logs_path, datasets))

        # Begin training
        model.fit(
            x=datasets.train_data,
            validation_data=datasets.test_data,
            epochs=hp_sc.num_epochs,
            batch_size=None,            # Required as None as we use an ImageDataGenerator; see preprocess.py get_data()
            callbacks=callback_list,
            initial_epoch=init_epoch,
        )
        model.load_weights("your_weights.h5")
        

    # Object Model
    elif task == '2':
        # Keras callbacks for training
        callback_list = [
            tf.keras.callbacks.TensorBoard(
                log_dir=logs_path,
                update_freq='batch',
                profile_batch=0),
            ImageLabelingLogger(logs_path, datasets),
            CustomModelSaver(checkpoint_path, ARGS.task, hp_obj.max_num_weights)
        ]

        # Include confusion logger in callbacks if flag set
        if ARGS.confusion:
            callback_list.append(ConfusionMatrixLogger(logs_path, datasets))

        # Begin training
        model.fit(
            x=datasets.train_data,
            validation_data=datasets.test_data,
            epochs=hp_obj.num_epochs,
            batch_size=None,            # Required as None as we use an ImageDataGenerator; see preprocess.py get_data()
            callbacks=callback_list,
            initial_epoch=init_epoch,
        )

    #VGG SCENE
    elif task == '3':
        # Keras callbacks for training
        callback_list = [
            tf.keras.callbacks.TensorBoard(
                log_dir=logs_path,
                update_freq='batch',
                profile_batch=0),
            ImageLabelingLogger(logs_path, datasets),
            CustomModelSaver(checkpoint_path, ARGS.task, hp_sc.max_num_weights)
        ]

        # Include confusion logger in callbacks if flag set
        if ARGS.confusion:
            callback_list.append(ConfusionMatrixLogger(logs_path, datasets))

        # Begin training
        model.fit(
            x=datasets.train_data,
            validation_data=datasets.test_data,
            epochs=hp_sc.num_epochs,
            batch_size=None,            # Required as None as we use an ImageDataGenerator; see preprocess.py get_data()
            callbacks=callback_list,
            initial_epoch=init_epoch,
        )

    # VGG OBJECT
    elif task == '4':
        # Keras callbacks for training
        callback_list = [
            tf.keras.callbacks.TensorBoard(
                log_dir=logs_path,
                update_freq='batch',
                profile_batch=0),
            ImageLabelingLogger(logs_path, datasets),
            CustomModelSaver(checkpoint_path, ARGS.task, hp_obj.max_num_weights)
        ]

        # Include confusion logger in callbacks if flag set
        if ARGS.confusion:
            callback_list.append(ConfusionMatrixLogger(logs_path, datasets))

        # Begin training
        model.fit(
            x=datasets.train_data,
            validation_data=datasets.test_data,
            epochs=hp_obj.num_epochs,
            batch_size=None,            # Required as None as we use an ImageDataGenerator; see preprocess.py get_data()
            callbacks=callback_list,
            initial_epoch=init_epoch,
        )



    else: 
        #DO BOTH... Need to figure out what that means
        pass

def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=1,
    )

#Placeholder for processing an image
def predict_scene(model, image_path, preprocess_fn):
    """Predict the scene in the image using a pre-trained model."""

    # Load the image and resize it to the model's expected input size
    input_image = Image.open(image_path)
    input_image = np.array(input_image)
    
    image = resize(input_image, (hp_sc.img_size, hp_sc.img_size, 3), preserve_range=True, mode='reflect', anti_aliasing=True)

    # Preprocess the image and expand the dimensions to match the model's input shape
    input_image = preprocess_fn(image)
    input_image = np.expand_dims(input_image, axis=0)

    # Predict the scene using the model
    predictions = model(input_image)
    

    # Get the class with the highest probability
    predicted_class = np.argmax(predictions)

    # Get the sorted class names from the test data folder
    class_names = get_class_names("../data/15_Scene/test")

    # Convert the predicted class number to the class name
    predicted_class_name = scene_num_to_scene_name(predicted_class, class_names)

    # Return the predicted class name
    return predicted_class_name

def get_class_names(folder_path):
    """Get a sorted list of class names from the folder path."""

    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"Invalid folder path: {folder_path}")
        return None

    # Get the subdirectories in the folder
    class_dirs = [entry.name for entry in os.scandir(folder_path) if entry.is_dir()]

    # Sort the class names
    class_names = sorted(class_dirs)

    return class_names

def scene_num_to_scene_name(scene_num, class_names):
    """Convert a class number to the corresponding class name."""

    if scene_num < 0 or scene_num >= len(class_names):
        print(f"Invalid scene number: {scene_num}")
        return None

    return class_names[scene_num]


def main():
    """ Main function. """

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    # If loading from a checkpoint, the loaded checkpoint's directory
    # will be used for future checkpoints
    if ARGS.load_checkpoint is not None:
        ARGS.load_checkpoint = os.path.abspath(ARGS.load_checkpoint)

        # Get timestamp and epoch from filename
        regex = r"(?:.+)(?:\.e)(\d+)(?:.+)(?:.h5)"
        init_epoch = int(re.match(regex, ARGS.load_checkpoint).group(1)) + 1
        timestamp = os.path.basename(os.path.dirname(ARGS.load_checkpoint))

    # If paths provided by program arguments are accurate, then this will
    # ensure they are used. If not, these directories/files will be
    # set relative to the directory of run.py
    if os.path.exists(ARGS.data):
        ARGS.data = os.path.abspath(ARGS.data)
    if os.path.exists(ARGS.load_vgg):
        ARGS.load_vgg = os.path.abspath(ARGS.load_vgg)

    # Run script from location of run.py
    os.chdir(sys.path[0])

    datasets = None

    if ARGS.task == '1':
        datasets = Datasets_sc(ARGS.data, ARGS.task)
        model = YourModel_sc()
        model(tf.keras.Input(shape=(hp_sc.img_size, hp_sc.img_size, 3)))
        checkpoint_path = "checkpoints" + os.sep + \
            "your_model" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "your_model" + \
            os.sep + timestamp + os.sep

        # Print summary of model
        model.summary()

    elif ARGS.task == '2':
        datasets = Datasets_obj(ARGS.data, ARGS.task)
        model = YourModel_obj()
        model(tf.keras.Input(shape=(hp_obj.img_size, hp_obj.img_size, 3)))
        checkpoint_path = "checkpoints" + os.sep + \
            "your_model" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "your_model" + \
            os.sep + timestamp + os.sep

        # Print summary of model
        model.summary()

    elif ARGS.task == '3':
        datasets = Datasets_sc(ARGS.data, ARGS.task)
        model = VGGModel_sc()
        checkpoint_path = "checkpoints" + os.sep + \
            "vgg_model" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "vgg_model" + \
            os.sep + timestamp + os.sep
        model(tf.keras.Input(shape=(224, 224, 3)))

        # Print summaries for both parts of the model
        model.vgg16.summary()
        model.head.summary()

        # Load base of VGG model
        model.vgg16.load_weights(ARGS.load_vgg, by_name=True)

    elif ARGS.task == '4':
        datasets = Datasets_sc(ARGS.data, ARGS.task)
        model = VGGModel_obj()
        checkpoint_path = "checkpoints" + os.sep + \
            "vgg_model" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "vgg_model" + \
            os.sep + timestamp + os.sep
        model(tf.keras.Input(shape=(224, 224, 3)))

        # Print summaries for both parts of the model
        model.vgg16.summary()
        model.head.summary()

        # Load base of VGG model
        model.vgg16.load_weights(ARGS.load_vgg, by_name=True)


    elif ARGS.task == '5':
        ## SCENE CLASSIFICATION
        datasets = Datasets_sc(ARGS.data, ARGS.task)
        # Load the pre-trained model for scene classification
        model_path = "your_weights.h5"
        model = YourModel_sc()
        model(tf.keras.Input(shape=(hp_sc.img_size, hp_sc.img_size, 3)))
        model.load_weights(model_path)

        # Provide the path to the input image
        # input_image_path = "../data/15_Scene/test/Office/image_0001.jpg"
        input_image_path = ARGS.imagePath

        # Predict the scene in the input image using the pre-trained model
        predicted_class = predict_scene(model, input_image_path, datasets.preprocess_fn)

        #OBJECT CAPTION GENERATION

        captionOutput =generate_caption(input_image_path, "resnet")

        # Print the predicted class
        print("Predicted class: ", predicted_class)
        # Print the caption
        print("Caption test: ", captionOutput)
        speechString = captionOutput + " " + predicted_class
        speaker = speech.TTS(135, 1.0, 0)
        speaker.speak(speechString)
        print(speechString)
        

    elif ARGS.task == '6':
        datasets = Datasets_sc(ARGS.data, ARGS.task)
        # Load the pre-trained model for scene classification
        model_path = "vgg_weights.h5"
        model = VGGModel_sc()
        model(tf.keras.Input(shape=(224, 224, 3)))
        # model.load_weights(model_path)

        # Load base of VGG model
        model.vgg16.load_weights(ARGS.load_vgg, by_name=True)
        model.head.load_weights(model_path, by_name=True)

        # Provide the path to the input image
        # input_image_path = "../data/15_Scene/test/Office/image_0001.jpg"
        input_image_path = ARGS.imagePath

        # Predict the scene in the input image using the pre-trained model
        predicted_class = predict_scene(model, input_image_path, datasets.preprocess_fn)

        # Print the predicted class
        print("Predicted class: ", predicted_class)

    else:
        model = None #REPLACE WITH BOTH
        checkpoint_path = "checkpoints" + os.sep + \
            "vgg_model" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "vgg_model" + \
            os.sep + timestamp + os.sep
        model(tf.keras.Input(shape=(224, 224, 3)))

        # Print summaries for both parts of the model
        model.vgg16.summary()
        model.head.summary()

        # Load base of VGG model
        model.vgg16.load_weights(ARGS.load_vgg, by_name=True)

    # Load checkpoints
    if ARGS.load_checkpoint is not None:
        if ARGS.task == '1':
            model.load_weights(ARGS.load_checkpoint, by_name=False)
        else:
            model.head.load_weights(ARGS.load_checkpoint, by_name=False)

    # # Make checkpoint directory if needed
    if ARGS.task == '1':
        if not ARGS.evaluate and not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
    if ARGS.task == '2':
        if not ARGS.evaluate and not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
    if ARGS.task == '3':
        if not ARGS.evaluate and not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
    if ARGS.task == '4':
        if not ARGS.evaluate and not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

    # Compile model graph
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])

    if ARGS.evaluate:
        test(model, datasets.test_data)

        # TODO: change the image path to be the image of your choice by changing
        # the lime-image flag when calling run.py to investigate
        # i.e. python run.py --evaluate --lime-image test/Bedroom/image_003.jpg

    # Commented out to take out LIME stuff for now
        # path = ARGS.data + os.sep + ARGS.lime_image
        # LIME_explainer(model, path, datasets.preprocess_fn, timestamp)
    else:
        if ARGS.task == '5':
            checkpoint_path = None
            logs_path = None
            train(model, datasets, checkpoint_path, logs_path, init_epoch, ARGS.task)
        if ARGS.task == '6':
            checkpoint_path = None
            logs_path = None
            train(model, datasets, checkpoint_path, logs_path, init_epoch, ARGS.task)
        else: 
            train(model, datasets, checkpoint_path, logs_path, init_epoch, ARGS.task)


# Make arguments global
ARGS = parse_args()

main()