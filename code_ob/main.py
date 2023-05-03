import argparse
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt


from model import ImageCaptionModel, accuracy_function, loss_function
from preprocess import get_image_feature
from decoder import TransformerDecoder 
import transformer
from preprocess import VGGModel


def parse_args(args=None):
    """ 
    Perform command-line argument parsing (other otherwise parse arguments with defaults). 
    To parse in an interative context (i.e. in notebook), add required arguments.
    These will go into args and will generate a list that can be passed in.
    For example: 
        parse_args('--type', 'rnn', ...)
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--task',           required=True,              choices=['train', 'test', 'both', 'single'],  help='Task to run')
    parser.add_argument('--data',           required=True,              help='File path to the assignment data file.')

    parser.add_argument('--epochs',         type=int,   default=3,      help='Number of epochs used in training.')
    parser.add_argument('--lr',             type=float, default=1e-3,   help='Model\'s learning rate')
    parser.add_argument('--optimizer',      type=str,   default='adam', choices=['adam', 'rmsprop', 'sgd'], help='Model\'s optimizer')
    parser.add_argument('--batch_size',     type=int,   default=100,    help='Model\'s batch size.')
    parser.add_argument('--hidden_size',    type=int,   default=256,    help='Hidden size used to instantiate the model.')
    parser.add_argument('--window_size',    type=int,   default=20,     help='Window size of text entries.')
    parser.add_argument('--chkpt_path',     default='',                 help='where the model checkpoint is')
    parser.add_argument('--check_valid',    default=True,               action="store_true",  help='if training, also print validation after each epoch')
    parser.add_argument('--image_path',     type=str,   default='',     help='image path for single image testing.')
    if args is None: 
        return parser.parse_args()      ## For calling through command line
    return parser.parse_args(args)      ## For calling through notebook.


def main(args):

    ##############################################################################
    ## Data Loading
    with open(args.data, 'rb') as data_file:
        data_dict = pickle.load(data_file)

    # feat_prep = lambda x: np.repeat(np.array(x).reshape(-1, 2048), 5, axis=0)
    feat_prep = lambda x: np.repeat(np.array(x).reshape(-1, 512), 5, axis=0)
    img_prep  = lambda x: np.repeat(x, 5, axis=0)
    train_captions  = np.array(data_dict['train_captions'])
    test_captions   = np.array(data_dict['test_captions'])
    train_img_feats = feat_prep(data_dict['train_image_features'])
    test_img_feats  = feat_prep(data_dict['test_image_features'])
    word2idx        = data_dict['word2idx']

    ##############################################################################
    ## Training Task
    if args.task in ('train', 'both'):
        
        ##############################################################################
        ## Model Construction
        decoder_class = TransformerDecoder
        decoder = decoder_class(
            vocab_size  = len(word2idx), 
            hidden_size = args.hidden_size, 
            window_size = args.window_size
        )
        model = ImageCaptionModel(decoder)
        
        compile_model(model, args)
        train_model(
            model, train_captions, train_img_feats, word2idx['<pad>'], args, 
            valid = (test_captions, test_img_feats)
        )
        if args.chkpt_path: 
            ## Save model to run testing task afterwards
            save_model(model, args)
                
    ##############################################################################
    ## Testing Task
    if args.task in ('test', 'both'):
        if args.task != 'both': 
            ## Load model for testing. Note that architecture needs to be consistent
            model = load_model(args)
        if not (args.task == 'both' and args.check_valid):
            test_model(model, test_captions, test_img_feats, word2idx['<pad>'], args)

    ##############################################################################
    ## Single Image Task
    if args.task in ('single'):
        model = load_model(args)

        img_in_shape = (1, 224, 224, 3)
        vgg_model = VGGModel()
        path = "../code/vgg16_imagenet.h5"
        vgg_model.build(img_in_shape)
        vgg_model.vgg16.load_weights(path, by_name=True)
        feature_fn = vgg_model

        image_feat, image = get_image_feature(args.image_path, feature_fn)

        temperature = 0.5
        output = gen_caption_temperature(model, image_feat, word2idx, word2idx['<pad>'], temperature, args.window_size)
        print(output)
        plt.imshow(image)
        plt.text(50, -10, output)
        plt.show()
        
    ##############################################################################

##############################################################################
## UTILITY METHODS

def save_model(model, args):
    '''Loads model based on arguments'''
    tf.keras.models.save_model(model, args.chkpt_path)
    print(f"Model saved to '{args.chkpt_path}'")


def load_model(args):
    '''Loads model by reference based on arguments. Also returns said model'''
    model = tf.keras.models.load_model(
        args.chkpt_path,
        custom_objects=dict(
            AttentionHead           = transformer.AttentionHead,
            AttentionMatrix         = transformer.AttentionMatrix,
            TransformerBlock        = transformer.TransformerBlock,
            PositionalEncoding      = transformer.PositionalEncoding,
            TransformerDecoder      = TransformerDecoder,
            ImageCaptionModel       = ImageCaptionModel
        ),
    )
    ## Saving is very nuanced. Might need to set the custom components correctly.
    ## Functools.partial is a function wrapper that auto-fills a selection of arguments. 
    ## so in other words, the first argument of ImageCaptionModel.test is model (for self)
    from functools import partial
    model.test    = partial(ImageCaptionModel.test,    model)
    model.train   = partial(ImageCaptionModel.train,   model)
    model.compile = partial(ImageCaptionModel.compile, model)
    compile_model(model, args)
    print(f"Model loaded from '{args.chkpt_path}'")
    return model


def compile_model(model, args):
    '''Compiles model by reference based on arguments'''
    optimizer = tf.keras.optimizers.get(args.optimizer).__class__(learning_rate = args.lr)
    model.compile(
        optimizer   = optimizer,
        loss        = loss_function,
        metrics     = [accuracy_function]
    )


def train_model(model, captions, img_feats, pad_idx, args, valid):
    '''Trains model and returns model statistics'''
    stats = []
    try:
        for epoch in range(args.epochs):
            stats += [model.train(captions, img_feats, pad_idx, batch_size=args.batch_size)]
            if args.check_valid:
                model.test(valid[0], valid[1], pad_idx, batch_size=args.batch_size)
    except KeyboardInterrupt as e:
        if epoch > 1:
            print("Key-value interruption. Trying to early-terminate. Interrupt again to not do that!")
        else: 
            raise e
        
    return stats


def test_model(model, captions, img_feats, pad_idx, args):
    '''Tests model and returns model statistics'''
    perplexity, accuracy = model.test(captions, img_feats, pad_idx, batch_size=args.batch_size)
    return perplexity, accuracy


def gen_caption_temperature(model, image_embedding, wordToIds, padID, temp, window_length):
    """
    Function used to generate a caption using an ImageCaptionModel given
    an image embedding. 
    """
    idsToWords = {id: word for word, id in wordToIds.items()}
    unk_token = wordToIds['<unk>']
    caption_so_far = [wordToIds['<start>']]
    while len(caption_so_far) < window_length and caption_so_far[-1] != wordToIds['<end>']:
        caption_input = np.array([caption_so_far + ((window_length - len(caption_so_far)) * [padID])])
        logits = model(np.expand_dims(image_embedding, 0), caption_input)
        logits = logits[0][len(caption_so_far) - 1]
        probs = tf.nn.softmax(logits / temp).numpy()
        next_token = unk_token
        attempts = 0
        while next_token == unk_token and attempts < 5:
            next_token = np.random.choice(len(probs), p=probs)
            attempts += 1
        caption_so_far.append(next_token)
    return ' '.join([idsToWords[x] for x in caption_so_far][1:-1])


## END UTILITY METHODS
##############################################################################

if __name__ == '__main__':
    main(parse_args())

    # TODO:
        # integrate with brennan's stuff...