from basic_functions import read_label_excel, read_data, recursive_data_label_prepare
from Encoder import Encoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ReLU, LeakyReLU, Dropout, Flatten, TimeDistributed, LSTM, Bidirectional, Dense, Input, MaxPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

import numpy as np
import matplotlib.pyplot as plt

GENERATE_SQUARE = 64
IMAGE_CHANNELS = 3
BATCH_SIZE = 64
num_timesteps = 5
input_shape = (None, num_timesteps, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS)

test_label_path = r"E:\Research Data\Data_UF\Jule_LD14_PKYonge_Class1_Mar142019\binary_label.xlsx"
test_DATA_PATH = r"E:\Research Data\Data_UF\Jule_LD14_PKYonge_Class1_Mar142019\Image_Data"

train_label_path = r"E:\Research Data\Data_UF\Yingbo_LD2_PKYonge_Class1_Mar142019_B\binary_label.xlsx"
train_DATA_PATH = r"E:\Research Data\Data_UF\Yingbo_LD2_PKYonge_Class1_Mar142019_B\Image_Data"

# test_label_path = r"E:\Research Data\Data_UF\Yingbo_LD2_PKYonge_Class1_Mar142019_B\binary_label.xlsx"
# test_DATA_PATH = r"E:\Research Data\Data_UF\Yingbo_LD2_PKYonge_Class1_Mar142019_B\Image_Data"
#
# train_label_path = r"E:\Research Data\Data_UF\Jule_LD14_PKYonge_Class1_Mar142019\binary_label.xlsx"
# train_DATA_PATH = r"E:\Research Data\Data_UF\Jule_LD14_PKYonge_Class1_Mar142019\Image_Data"

train_data, train_target = recursive_data_label_prepare(train_label_path, train_DATA_PATH, num_timesteps, GENERATE_SQUARE, IMAGE_CHANNELS)
test_data, test_target = recursive_data_label_prepare(test_label_path, test_DATA_PATH, num_timesteps, GENERATE_SQUARE, IMAGE_CHANNELS)
origin_test_target = read_label_excel(test_label_path)


cnn = Sequential()

cnn.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
# cnn.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')) Maxpooling deteriorate accuracy
cnn.add(BatchNormalization(momentum=0.9))
cnn.add(LeakyReLU(alpha=0.2))
# cnn.add(ReLU())
cnn.add(Dropout(0.2))

cnn.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same'))
# cnn.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
cnn.add(BatchNormalization(momentum=0.9))
cnn.add(LeakyReLU(alpha=0.2))
# cnn.add(ReLU())
cnn.add(Dropout(0.2))

cnn.add(Conv2D(16, (3, 3), strides=(2, 2), padding='same'))
# cnn.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
cnn.add(BatchNormalization(momentum=0.9))
cnn.add(LeakyReLU(alpha=0.2))
# cnn.add(ReLU())
cnn.add(Dropout(0.2))

cnn.add(Conv2D(8, (3, 3), strides=(2, 2), padding='same'))
# cnn.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
cnn.add(BatchNormalization(momentum=0.9))
cnn.add(LeakyReLU(alpha=0.2))
# cnn.add(ReLU())
cnn.add(Dropout(0.2))

cnn.add(Flatten())

transform = Sequential()
transform.add(TimeDistributed(cnn))
transform.add(Encoder(2, 256, 2, 0.2))
# transform.add(Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01)))
transform.add(Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.1)))
transform.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.5), metrics=['accuracy'])
transform.build(input_shape=input_shape)
transform.summary()

history = transform.fit(
    train_data,
    train_target,
    batch_size = BATCH_SIZE,
    epochs = 200,
    shuffle = True,
    validation_data = (test_data, test_target)
)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Start Testing

_, test_acc = transform.evaluate(
    test_data,
    test_target
)
print('Accuracy: %.2f' % (test_acc * 100))

y_pred = transform.predict(
    test_data,
    verbose=0
)

pred_list = y_pred.tolist()

for i in range(len(pred_list)):
    for j in range(num_timesteps):
        if pred_list[i][j] > [0.5]:
            pred_list[i][j] = [1]
        else:
            pred_list[i][j] = [0]

final_pred_list = []
for i in range(len(pred_list)):
    final_pred_list.append(pred_list[i][0][0])
final_pred_list.append(pred_list[i][1][0])
final_pred_list.append(pred_list[i][2][0])
final_pred_list.append(pred_list[i][3][0])
final_pred_list.append(pred_list[i][4][0])

final_pred = np.asarray(final_pred_list)
print(classification_report(origin_test_target, final_pred_list))