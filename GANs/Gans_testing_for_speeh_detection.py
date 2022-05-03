import xlrd
import os
from PIL import Image
import numpy as np

label_path = r"E:\Research Data\Data_UF\Yingbo_LD2_PKYonge_Class1_Mar142019_B\new_label.xlsx"
DATA_PATH = r"E:\Research Data\Data_UF\Yingbo_LD2_PKYonge_Class1_Mar142019_B\Image_Data"

GENERATE_SQUARE = 64
IMAGE_CHANNELS = 3
TRAIN_PERC = 0.75
CLASS_NUM = 3
BATCH_SIZE = 60
n_samples = int(BATCH_SIZE / CLASS_NUM)
latent_dim = 100
IMAGE_NUM = 2574
BATCH_NUM = int(IMAGE_NUM / BATCH_SIZE) + 1
ITERATIONS = 2000


### get the all data for 3 classes ######################################################################################################
def excel_data(file_path):
    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_index(0)

    nrows = table.nrows
    ncols = table.ncols

    excel_list = []
    for row in range(0, nrows):
        for col in range(ncols):
            cell_value = int(table.cell(row, col).value)
            excel_list.append(cell_value)
    return excel_list


print("Start reading Image & Label data...")
list = excel_data(label_path)

list_0 = []
for i, j in enumerate(list):
    if j == 0:
        list_0.append(i)

list_1 = []
for i, j in enumerate(list):
    if j == 1:
        list_1.append(i)

list_2 = []
for i, j in enumerate(list):
    if j == 2:
        list_2.append(i)

X_with_Class_0_Num = len(list_0)
X_with_Class_1_Num = len(list_1)
X_with_Class_2_Num = len(list_2)

print(X_with_Class_0_Num)
print(X_with_Class_1_Num)
print(X_with_Class_2_Num)

ALL_DATA_NUM = X_with_Class_0_Num + X_with_Class_1_Num + X_with_Class_2_Num

X_with_Class_0_Train_Num = int(X_with_Class_0_Num * TRAIN_PERC)
X_with_Class_1_Train_Num = int(X_with_Class_1_Num * TRAIN_PERC)
X_with_Class_2_Train_Num = int(X_with_Class_2_Num * TRAIN_PERC)

class_0_data = []
for index in range(X_with_Class_0_Num):
    path = os.path.join(DATA_PATH, str(list_0[index]) + ".jpg")
    image = Image.open(path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
    class_0_data.append(np.asarray(image))

class_0_data = np.reshape(class_0_data, (-1, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))

class_1_data = []
for index in range(X_with_Class_1_Num):
    path = os.path.join(DATA_PATH, str(list_1[index]) + ".jpg")
    image = Image.open(path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
    class_1_data.append(np.asarray(image))

class_1_data = np.reshape(class_1_data, (-1, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))

class_2_data = []
for index in range(X_with_Class_2_Num):
    path = os.path.join(DATA_PATH, str(list_2[index]) + ".jpg")
    image = Image.open(path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
    class_2_data.append(np.asarray(image))

class_2_data = np.reshape(class_2_data, (-1, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))

######split data into training and testing##################################################################

np.random.shuffle(class_0_data)
np.random.shuffle(class_1_data)
np.random.shuffle(class_2_data)

class_0_training_data = class_0_data[0: X_with_Class_0_Train_Num]
class_0_testing_data = class_0_data[X_with_Class_0_Train_Num:]
ix = np.random.randint(0, len(class_0_testing_data), 20)
class_0_testing_trim_data = np.asarray(class_0_testing_data[ix])


class_1_training_data = class_1_data[0: X_with_Class_1_Train_Num]
class_1_testing_data = class_1_data[X_with_Class_1_Train_Num:]
ix = np.random.randint(0, len(class_1_testing_data), 20)
class_1_testing_trim_data = np.asarray(class_1_testing_data[ix])

class_2_training_data = class_2_data[0: X_with_Class_2_Train_Num]
class_2_testing_data = class_2_data[X_with_Class_2_Train_Num:]
ix = np.random.randint(0, len(class_2_testing_data), 20)
class_2_testing_trim_data = np.asarray(class_2_testing_data[ix])

print("Length of class_0 test data: ", len(class_0_testing_data))
print("Length of class_1 test data: ", len(class_1_testing_data))
print("Length of class_2 test data: ", len(class_2_testing_data))

X_test = np.concatenate((class_0_testing_data, class_1_testing_data, class_2_testing_data), axis=0)
y_test = np.concatenate((np.zeros((len(class_0_testing_data), 1)), np.ones((len(class_1_testing_data), 1)),
                         2 * np.ones((len(class_2_testing_data), 1))), axis=0)

X_test_trim = np.concatenate((class_0_testing_trim_data, class_1_testing_trim_data, class_2_testing_trim_data), axis=0)
y_test_trim = np.concatenate((np.zeros((len(class_0_testing_trim_data), 1)), np.ones((len(class_1_testing_trim_data), 1)),
                         2 * np.ones((len(class_2_testing_trim_data), 1))), axis=0)

########################################################################################################

print("Start building networks...")
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import Dropout
from sklearn.metrics import classification_report
from keras.layers import Lambda
from keras.layers import Activation
import matplotlib.pyplot as plt
from keras import backend


# define supervised and unsupervised discriminator models
def define_discriminator(in_shape=(64, 64, 3), n_classes=3):
    # image input
    in_image = Input(shape=in_shape)
    # downsample
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(in_image)
    fe = BatchNormalization(momentum=0.9)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)

    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = BatchNormalization(momentum=0.9)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)

    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = BatchNormalization(momentum=0.9)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)

    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = BatchNormalization(momentum=0.9)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)

    # flatten feature maps
    fe = Flatten()(fe)

    # dropout
    fe = Dropout(0.4)(fe)

    d_out_layer = Dense(1, activation="sigmoid")(fe)
    d_model = Model(in_image, d_out_layer)
    d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

    c_out_layer = Dense(n_classes, activation="softmax")(fe)
    c_model = Model(in_image, c_out_layer)
    c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

    return d_model, c_model


# define the generator model
def define_generator(latent_dim):
    # image generator input
    in_lat = Input(shape=(latent_dim,))

    n_nodes = 128 * 8 * 8
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((8, 8, 128))(gen)

    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    out_layer = Conv2D(3, (7, 7), activation='tanh', padding='same')(gen)

    # define model
    model = Model(in_lat, out_layer)
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect image output from generator as input to discriminator
    gan_output = d_model(g_model.output)
    # define gan model as taking noise and outputting a classification
    model = Model(g_model.input, gan_output)

    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


# create the discriminator models
d_model, c_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)

print("Start training...")

epoch = 0

for i in range(ITERATIONS):
    ####generate supervised real data
    ix = np.random.randint(0, len(class_0_training_data), n_samples)
    X_supervised_samples_class_0 = np.asarray(class_0_training_data[ix])
    Y_supervised_samples_class_0 = np.zeros((n_samples, 1))

    ix = np.random.randint(0, len(class_1_training_data), n_samples)
    X_supervised_samples_class_1 = np.asarray(class_1_training_data[ix])
    Y_supervised_samples_class_1 = np.ones((n_samples, 1))

    ix = np.random.randint(0, len(class_2_training_data), n_samples)
    X_supervised_samples_class_2 = np.asarray(class_2_training_data[ix])
    Y_supervised_samples_class_2 = 2 * np.ones((n_samples, 1))

    Xsup_real = np.concatenate(
        (X_supervised_samples_class_0, X_supervised_samples_class_1, X_supervised_samples_class_2), axis=0)
    ysup_real = np.concatenate(
        (Y_supervised_samples_class_0, Y_supervised_samples_class_1, Y_supervised_samples_class_2), axis=0)
    ####generate unsupervised real data
    ix = np.random.randint(0, len(class_0_training_data), n_samples)
    X_unsupervised_samples_class_0 = np.asarray(class_0_training_data[ix])

    ix = np.random.randint(0, len(class_1_training_data), n_samples)
    X_unsupervised_samples_class_1 = np.asarray(class_1_training_data[ix])

    ix = np.random.randint(0, len(class_2_training_data), n_samples)
    X_unsupervised_samples_class_2 = np.asarray(class_2_training_data[ix])

    X_real = np.concatenate(
        (X_unsupervised_samples_class_0, X_unsupervised_samples_class_1, X_unsupervised_samples_class_2), axis=0)
    y_real = np.ones((BATCH_SIZE, 1))
    ##generate fake data
    seed = np.random.normal(0, 1, (BATCH_SIZE, latent_dim))
    X_fake = g_model.predict(seed)
    y_fake = np.zeros((BATCH_SIZE, 1))

    # update supervised discriminator (c)
    c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)
    # update unsupervised discriminator (d)
    d_loss1 = d_model.train_on_batch(X_real, y_real)
    d_loss2 = d_model.train_on_batch(X_fake, y_fake)
    # update generator (g)
    g_loss = gan_model.train_on_batch(seed, y_real)

    if (i + 1) % (BATCH_NUM * 1) == 0:
        epoch += 1
        print(f"Epoch {epoch}, c model accuarcy on training data: {c_acc}")
        _, test_acc = c_model.evaluate(X_test, y_test, verbose=0)
        print(f"Epoch {epoch}, c model accuarcy on test data: {test_acc}")
        y_pred = c_model.predict(X_test, batch_size=60, verbose=0)
        y_pred_bool = np.argmax(y_pred, axis=1)
        print(y_pred_bool)
        print(classification_report(y_test, y_pred_bool))