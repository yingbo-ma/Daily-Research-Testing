import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

from MultiHeadSelfAttention import PositionalEncoding, scaled_dot_product_attention, MultiHeadAttention

class EncoderLayer(layers.Layer):
    def __init__(self, ffn_units, nb_proj, droupout):
        super(EncoderLayer, self).__init__()
        self.ffn_units = ffn_units
        self.nb_proj = nb_proj
        self.droupout = droupout

    def build(self, input_shape):
        self.d_model = input_shape[-1]

        self.multi_head_attention = MultiHeadAttention(self.nb_proj)
        self.dropout_1 = layers.Dropout(rate=self.droupout)
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)

        self.dense_1 = layers.Dense(units=self.ffn_units, activation="relu")
        self.dense_2 = layers.Dense(units=self.d_model, activation="relu")

        self.dropout_2 = layers.Dropout(rate=self.droupout)
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training):
        attention = self.multi_head_attention(inputs, inputs, inputs)
        attention = self.dropout_1(attention, training=training)
        attention = self.norm_1(attention + inputs)

        outputs = self.dense_1(attention)
        outputs = self.dense_2(outputs)
        outputs = self.dropout_2(outputs)
        outputs = self.norm_2(outputs + attention)

        return outputs

class Encoder(layers.Layer):

    def __init__(self, nb_layers, ffn_units, nb_proj, dropout, name="encoder"):
        super(Encoder, self).__init__(name=name)
        self.nb_layers = nb_layers
        self.ffn_units = ffn_units
        self.nb_proj = nb_proj
        self.dropout = dropout

        self.pos_encoding = PositionalEncoding()
        self.dropout = layers.Dropout(rate=dropout)
        self.enc_layers = [EncoderLayer(ffn_units, nb_proj, dropout) for _ in range(nb_layers)]

    def build(self, input_shape):
        self.d_model = input_shape[-1]
        self.embedding = layers.Embedding(self.d_model, self.d_model)

    def call(self, inputs):
        # outputs = self.embedding(inputs)
        # outputs *= tf.math.sqrt(tf.cast(self.d_model, dtype="float32")) # whether pre-embedding here is useful or not is not ture

        outputs = self.pos_encoding(inputs)
        outputs = self.dropout(outputs)

        for i in range(self.nb_layers):
            outputs = self.enc_layers[i](outputs, True)

        return outputs

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, TimeDistributed, LSTM, Bidirectional, Dense, Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

GENERATE_SQUARE = 64
IMAGE_CHANNELS = 3
num_timesteps = 5
input_shape = (None, num_timesteps, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS)

cnn = Sequential()

cnn.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
cnn.add(BatchNormalization(momentum=0.9))
cnn.add(LeakyReLU(alpha=0.2))
cnn.add(Dropout(0.2))

cnn.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
cnn.add(BatchNormalization(momentum=0.9))
cnn.add(LeakyReLU(alpha=0.2))
cnn.add(Dropout(0.2))

cnn.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
cnn.add(BatchNormalization(momentum=0.9))
cnn.add(LeakyReLU(alpha=0.2))
cnn.add(Dropout(0.2))

cnn.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
cnn.add(BatchNormalization(momentum=0.9))
cnn.add(LeakyReLU(alpha=0.2))
cnn.add(Dropout(0.2))
cnn.add(Flatten())

transform = Sequential()
transform.add(TimeDistributed(cnn))
transform.add(Encoder(2, 1024, 2, 0.2))
transform.add(Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01)))
transform.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
transform.build(input_shape=input_shape)
transform.summary()

list = [[[1., 2., 3., 4, 5, 6, 7, 8], [2., 3., 4., 5, 6, 7, 8, 9]]]
inputs_sequence = np.asarray(list)

E = EncoderLayer(ffn_units=2048, nb_proj=2, droupout=0.2)
outputs = E(inputs_sequence, training=True)
print(outputs)
D = layers.Dense(units=1, activation="sigmoid")
o = D(outputs)
print(o)