import os
import pandas as pd
import numpy as np
from scipy import stats

FEATURE_NUM = 3
AVERAGE_N_ELEMENTS = 30 # video is 30 frames per second. our video is 60 fps, so 60 frames/second. now we want 2 frames per second, so every 30 frames we take an average.

if FEATURE_NUM == 1:
    FEATURE = "\\eye_gaze"
    NUM_LOOP = 288 # how many feaature points does this feature extract? for facial aus is 35, for head posi is 6, for eye-gaze is 120.
    # for eye-gaze, openface generate 112 2D landmarks, 228 3D landmarks, in total there are 288 eye-gaze related features per detected face.
elif FEATURE_NUM == 2:
    FEATURE = "\\facial_aus"
    NUM_LOOP = 35
elif FEATURE_NUM == 3:
    FEATURE = "\\head_movement"
    NUM_LOOP = 6
else:
    print("Please indicate feature numbers!")

# load feature for speaker 1
feature_corpus = r'E:\Github Repos\Project-Code\Regular-Workspace\Satisfaction_Modeling\Facial_Features' + FEATURE
speaker_1 = 's1'
feature_list = os.listdir(feature_corpus)

Feature_List = []

for feature_file in feature_list:
    Feature_List_per_session = []

    features = pd.read_csv(feature_corpus + "\\" + feature_file + "\\" + speaker_1 + '.csv', delimiter='\t', header=None).values.tolist()
    for utterance in features:
        feature = utterance[0].strip('][').split(', ')

        if (feature[0] == 'nan'):
            Feature_List_per_session.append(Feature_List_per_session[-1])
        else:
            temp_list = []
            # remove any brakets in num_strings
            for num_string in feature:
                if ('[' in num_string):
                    num_string = num_string.replace('[', '')
                if (']' in num_string):
                    num_string = num_string.replace(']', '')
            # extract feature numbers
                feature_num = float(num_string)
                temp_list.append(feature_num)
            temp_array = np.array(temp_list).reshape(-1, NUM_LOOP)
            if (FEATURE_NUM == 1):
                temp_array = temp_array[:, 0:8] # 8 eye direction features
            residue = len(temp_array) % AVERAGE_N_ELEMENTS # need to cut more when it comes to eye gaze, memory out
            temp_array = temp_array[:len(temp_array) - residue]
            if (len(temp_array) == 0): # not sure why this is happening, need to check later
                temp_array = temp_array_saved
            split_array = np.array_split(temp_array, (len(temp_array) / AVERAGE_N_ELEMENTS))
            aver_list = []

            for array in split_array:
                aver_arr = np.mean(array, axis=0).tolist()
                aver_list += aver_arr
            temp_array_saved = temp_array
            # aver_list = stats.zscore(np.array(aver_list)).tolist()
            Feature_List_per_session.append(aver_list)

    Feature_List.append(Feature_List_per_session)

# load feature for speaker 2
feature_corpus = r'E:\Github Repos\Project-Code\Regular-Workspace\Satisfaction_Modeling\Facial_Features' + FEATURE
speaker_1 = 's2'
feature_list = os.listdir(feature_corpus)

for feature_file in feature_list:
    Feature_List_per_session = []

    features = pd.read_csv(feature_corpus + "\\" + feature_file + "\\" + speaker_1 + '.csv', delimiter='\t', header=None).values.tolist()
    for utterance in features:
        feature = utterance[0].strip('][').split(', ')

        if (feature[0] == 'nan'):
            Feature_List_per_session.append(Feature_List_per_session[-1])
        else:
            temp_list = []
            # remove any brakets in num_strings
            for num_string in feature:
                if ('[' in num_string):
                    num_string = num_string.replace('[', '')
                if (']' in num_string):
                    num_string = num_string.replace(']', '')
            # extract feature numbers
                feature_num = float(num_string)
                temp_list.append(feature_num)

            temp_array = np.array(temp_list).reshape(-1, NUM_LOOP)
            if (FEATURE_NUM == 1):
                temp_array = temp_array[:, 0:8]
            residue = len(temp_array) % AVERAGE_N_ELEMENTS
            temp_array = temp_array[:len(temp_array) - residue]
            if (len(temp_array) == 0): # not sure why this is happening, need to check later
                temp_array = temp_array_saved
            split_array = np.array_split(temp_array, (len(temp_array) / AVERAGE_N_ELEMENTS))
            aver_list = []

            for array in split_array:
                aver_arr = np.mean(array, axis=0).tolist()
                aver_list += aver_arr
            temp_array_saved = temp_array
            # aver_list = stats.zscore(np.array(aver_list)).tolist()
            Feature_List_per_session.append(aver_list)

    Feature_List.append(Feature_List_per_session)

################## round each item in a list of floats to 2 decimal places
for session in Feature_List:
    for utterance in session:
        utterance = ['%.1f' % elem for elem in utterance] #

MAX_SEQ = 0 # MAX sequence per utterance time length
MAX_UTT = 0 # MAX utterance per session

for session in Feature_List: # this funtion gets the max sequence length of each utterance
    for utterance in session:
        if len(utterance) > MAX_SEQ:
            MAX_SEQ = len(utterance)
        else:
            MAX_SEQ = MAX_SEQ

for session in Feature_List:
    if len(session) > MAX_UTT:
        MAX_UTT = len(session)
    else:
        MAX_UTT = MAX_UTT

print("MAX utterance per session", MAX_UTT)
print("MAX sequence per utterance time length", MAX_SEQ)

for session in Feature_List:
    if (len(session) < MAX_UTT):
        session += ([[0.0]*MAX_SEQ] * (MAX_UTT - len(session)))

for session in Feature_List:
    for utterance in session:
        if (len(utterance) < MAX_SEQ):
            utterance += [0.0] * (MAX_SEQ - len(utterance))

print("The shape of the feature list is: SESSION_NUM * SESSION_UTTERANCE * UTTERANCE_LENGTH", len(Feature_List), len(Feature_List[0]), len(Feature_List[0][0]))

# Actual Satisfaction Score: Score Order: Dec 11. Dec 4. Feb 2019, in folder order
Score_List_Speaker_1 = [4.3, 4.2, 3.7, 4.5, 4.3, 3.8, 4.8, 4.3, 4.5, 3.8, 5, 4.2, 4.3, 5.0, 4.7, 3.8, 4.3, 4.7, 4.2, 4.8, 2.5, 3.7, 4.2]
Score_List_Speaker_2 = [4.0, 4.8, 4.3, 5.0, 4.8, 5.0, 4.8, 3.8, 4.8, 4.3, 4.7, 4.8, 5, 5.0, 4.3, 4.2, 4.8, 5.0, 5.0, 3.7, 2.2, 3.7, 4]
Score_List = Score_List_Speaker_2 + Score_List_Speaker_1 # should be rated by their partner, so the score is reversed for training
print("Score_List", len(Score_List))
## build the LSTM network, the input of the network would be Lound_Feature_List
## LSTM input shape: batch = SESSION_NUM, time_steps = SESSION_UTTERANCE, dim = UTTERANCE_LENGTH
Normalized_Score_List = []
for item in Score_List:
    ave_item = (item - min(Score_List)) / (max(Score_List) - min(Score_List))
    Normalized_Score_List.append(ave_item)
print("Normalized_Score_List", Normalized_Score_List)

########### shuffle data ##################
## prepare random shuffle data
shuffled_index_list = [16, 37, 26, 32, 29, 10, 8, 31, 6, 39, 28, 7, 5, 41, 25, 36, 17, 15, 38, 20, 0, 23, 1, 34, 2, 11, 12, 22, 27, 33, 35, 4, 40, 19, 44, 21, 42, 24, 43, 3, 30, 18, 9, 14, 13]
train_index = shuffled_index_list[0:36]
vali_index = shuffled_index_list[36:]

feature_train = []
feature_vali = []
score_train = []
score_vali = []

for index in train_index:
    feature_train.append(Feature_List[index])
    score_train.append(Normalized_Score_List[index])

for index in vali_index:
    feature_vali.append(Feature_List[index])
    score_vali.append(Normalized_Score_List[index])

print("score_train", score_train)
print("score_vali", score_vali)
################################################################

import tensorflow as tf
from tensorflow.keras.layers import Input, BatchNormalization, Dropout, Dense, LSTM, Dropout, RNN, GRU, Layer
from tensorflow.keras import Sequential, Model, backend
from tensorflow.keras.regularizers import l2

def creatModel():
    global model
    model = Sequential([
        Input(shape=(MAX_UTT, MAX_SEQ)),
        GRU(128, return_sequences=True),
        GRU(128, return_sequences=True),
        GRU(128, return_sequences=True),
        GRU(128, return_sequences=True),
        GRU(128, return_sequences=False),
        Dense(units=64, kernel_initializer='glorot_uniform', activation='relu'),
        # Dense(units=1),
        Dense(units=1, activation='sigmoid', kernel_regularizer=l2(0.001))
    ])
    model.build(input_shape=(None, MAX_UTT, MAX_SEQ))
    model.compile(loss='mean_absolute_error', optimizer=tf.optimizers.Adam(lr=0.001))
    # model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.SGD(learning_rate=0.05))
    print(model.summary())

creatModel()

print("start training...")
history = model.fit(
    feature_train,
    score_train,
    batch_size=len(Feature_List),
    epochs=300,
    validation_data=(feature_vali, score_vali),
    verbose=1
)

print("\n**************************************************************************************************")
print("Curves plotting...")
import matplotlib.pyplot as plt

# list all data in history
print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

predict_list = model.predict(feature_vali)
arr_list = np.array(predict_list).reshape(len(predict_list)).tolist()
print(arr_list)
print(score_vali)

# 5-fold cross validation results
print("Gettting Model Layers Output...")

for layer_index in range(len(model.layers)):
    print("Layer Index: ", layer_index)
    intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[layer_index].output)
    print(intermediate_layer_model.predict(feature_vali))
    print("##################################################################")