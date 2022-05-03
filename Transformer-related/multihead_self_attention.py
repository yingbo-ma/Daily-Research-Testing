import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class PositionalEncoding(layers.Layer):

    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def get_angles(self, pos, i, d_model): # pos: (seq_length, 1) i: (1, d_model)
        angles = 1 / np.power( 1000.,  2 * (i // 2) / np.float32(d_model))
        return pos * angles # output shape is (seq_length, d_model)

    def call(self, inputs):
        # print("input", inputs)
        seq_length = inputs.shape[2]
        d_model = inputs.shape[1]
        angles = self.get_angles(
            np.arange(seq_length)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        angles = np.transpose(angles)
        # print("angles", angles.shape)
        pos_encoding = angles[np.newaxis, ...]
        # print("encoded", pos_encoding.shape)
        return inputs + tf.cast(pos_encoding, dtype='float32')

def scaled_dot_product_attention(queries, keys, values, mask=None):
    product = tf.matmul(queries, keys, transpose_b=True)
    keys_dim = tf.cast(tf.shape(keys)[-1], dtype='float32') # tf.shape(keys)[-1] = keys.shape[2], which represents the embedding dimension
    scaled_product = product / tf.math.sqrt(keys_dim)

    attention_weights = tf.nn.softmax(scaled_product, axis=-1)
    attention_scores = tf.matmul(attention_weights, values)

    return attention_weights, attention_scores

class MultiHeadAttention(layers.Layer):

    def __init__(self, nb_proj):
        self.nb_proj = nb_proj
        super(MultiHeadAttention, self).__init__()

    def build(self, input_shape):
        self.dimension = input_shape[-1] # last dimension of the input shape, which is the dimension of each embedding dimension
        assert self.dimension % self.nb_proj == 0 # make sure multihead is correct
        self.d_proj = self.dimension // self.nb_proj

        self.query_lin = layers.Dense(units=self.dimension)
        self.key_lin = layers.Dense(units=self.dimension)
        self.value_lin = layers.Dense(units=self.dimension)
        self.final_lin = layers.Dense(units=self.dimension)

    def split_proj(self, inputs, batch_size): # inputs: (batch_size, seq_length, d_model)
        shape = (batch_size,
                 -1,
                 self.nb_proj,
                 self.d_proj)
        splited_inputs = tf.reshape(inputs, shape=shape) # (batch_size, seq_length, nb_proj, d_proj)
        return tf.transpose(splited_inputs, perm=[0, 2, 1, 3]) # (batch_size, nb_proj, seq_length, d_proj)

    def call(self, queries, keys, values):
        batch_size = tf.shape(queries)[0] # batch size is the 1st dimension of the tensor

        queries = self.query_lin(queries)
        keys = self.key_lin(keys)
        values = self.value_lin(values)

        queries = self.split_proj(queries, batch_size)
        keys = self.split_proj(keys, batch_size)
        values = self.split_proj(values, batch_size)

        scaled_scores, attention = scaled_dot_product_attention(queries, keys, values)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, shape=(batch_size, -1, self.dimension))

        outputs = self.final_lin(concat_attention)

        return outputs


P = PositionalEncoding()

list = [[[1., 2., 3., 4, 5, 6, 7, 8], [2., 3., 4., 5, 6, 7, 8, 9]]]
inputs_sequence = np.asarray(list)
output = P(inputs_sequence)
print(output)

M = MultiHeadAttention(nb_proj=4)
scores, outputs = M(output, output, output)
print(outputs)
print(scores)