import numpy as np
import tensorflow as tf


class AttentionMatrix(tf.keras.layers.Layer):

    def __init__(self, *args, use_mask=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mask = use_mask

    def call(self, inputs):
        """

        :param K: is [batch_size x window_size_keys x embedding_size]
        :param Q: is [batch_size x window_size_queries x embedding_size]
        :return: attention matrix
        """
        K, Q = inputs
        window_size_queries = Q.get_shape()[1]  # window size of queries
        window_size_keys    = K.get_shape()[1]  # window size of keys

        mask_vals = np.triu(np.ones((window_size_queries, window_size_keys)) * np.NINF, k=1)
        mask = tf.convert_to_tensor(value=mask_vals, dtype=tf.float32)
        atten_mask = tf.tile(tf.reshape(mask, [-1, window_size_queries, window_size_keys]), [tf.shape(input=K)[0], 1, 1])

        Q = tf.cast(Q, dtype=tf.float32)
        K = tf.cast(K, dtype=tf.float32)
        atten = tf.matmul(Q, K, transpose_b=True)
        atten = tf.divide(atten, tf.sqrt(tf.cast(window_size_keys, dtype=tf.float32)))
        if self.use_mask: atten += atten_mask
        atten = tf.nn.softmax(atten)

        return atten


class AttentionHead(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, is_self_attention, **kwargs):
        super(AttentionHead, self).__init__(**kwargs)
        self.use_mask = is_self_attention

        def make_variables(*dims, initializer=tf.random.truncated_normal): 
            return tf.Variable(initializer(dims, stddev=.1))

        self.K = make_variables(input_size, output_size)
        self.V = make_variables(input_size, output_size)
        self.Q = make_variables(input_size, output_size)

        self.attn_mtx = AttentionMatrix(use_mask=self.use_mask)


    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """

        :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
        """ 

        K = tf.tensordot(inputs_for_keys, self.K, axes=1)
        V = tf.tensordot(inputs_for_values, self.V, axes=1)
        Q = tf.tensordot(inputs_for_queries, self.Q, axes=1)

        inputs = K, Q
        atten = self.attn_mtx(inputs)
        Z = tf.matmul(atten, V)

        return Z


class TransformerBlock(tf.keras.layers.Layer):
    """
    """

    def __init__(self, emb_sz, MultiHeadedAttention=False, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)

        self.ff_layer = tf.keras.layers.Dense(emb_sz)

        self.self_atten         = AttentionHead(emb_sz, emb_sz, True)
        self.self_context_atten = AttentionHead(emb_sz, emb_sz, False)
        self.layer_norm = tf.keras.layers.LayerNormalization()

        self.add = tf.keras.layers.Add()


    @tf.function
    def call(self, inputs, context_sequence):
        """
        This functions calls a transformer block.

        :param inputs: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :param context_sequence: tensor of shape [BATCH_SIZE x CONTEXT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :return: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        """

        masked_atten = self.self_atten(inputs, inputs, inputs)
        masked_res_norm = self.layer_norm(self.add([masked_atten, inputs]))
        
        unmasked_atten = self.self_context_atten(context_sequence, context_sequence, inputs)
        unmasked_res_norm = self.layer_norm(self.add([unmasked_atten, masked_res_norm]))

        ff_result = self.ff_layer(unmasked_res_norm)
        res_norm = self.layer_norm(self.add([ff_result, unmasked_res_norm]))
        output = tf.nn.relu(res_norm)

        return output
    

def positional_encoding(length, depth):
    ## Source: https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer

    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]    # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth  # (1, depth)
    angle_rates = 1 / (10000**depths)               # (1, depth)
    angle_rads = positions * angle_rates            # (pos, depth)
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    """
    """

    def __init__(self, vocab_size, embed_size, window_size):
        """
        """
        
        super().__init__()

        self.embed_size = embed_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.pos_encoding = positional_encoding(window_size, embed_size)

    def call(self, x):
        """
        """

        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embed_size, tf.float32))
        x  = x + self.pos_encoding[tf.newaxis, :length, :]
        
        return x