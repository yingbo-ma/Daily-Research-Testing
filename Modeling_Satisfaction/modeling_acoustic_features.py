import os
import pandas as pd
import numpy as np

FEATURE_NUM = 4

if FEATURE_NUM == 1:
    FEATURE = "\\loudness"
elif FEATURE_NUM == 2:
    FEATURE = "\\specflux"
elif FEATURE_NUM == 3:
    FEATURE = "\\jitter"
elif FEATURE_NUM == 4:
    FEATURE = "\\shimmer"
else:
    print("Please indicate feature numbers!")

feature_corpus = 'E:\Github Repos\Project-Code\Regular-Workspace\Satisfaction_Modeling\Audio_Features' + FEATURE
speaker_1 = 's1'
feature_list = os.listdir(feature_corpus)

Feature_List = []
AVERAGE_N_ELEMENTS = 96 # for loudness and specflux, every sample frame we get 1 point; for mfcc, every sample frame we get 4 points; this parameter decides our feature sample scale

# load features from speaker 1
for feature_file in feature_list:
    Feature_List_per_session = []

    loud_list = pd.read_csv(feature_corpus + "\\" + feature_file + "\\" + speaker_1 + '_remove_eou_silence.csv.csv', delimiter='\t', header=None).values.tolist()
    # loud_list = pd.read_csv(feature_corpus + "\\" + feature_file + "\\" + speaker_1 + '_remove_eou_silence.csv', delimiter='\t', header=None).values.tolist()
    for list in loud_list:
        list_num = list[0].strip('][').split(', ')
        if(list_num[0][0] == 'n'): # ['nan']
            # print("ERROR! [nan] feature reported due to the audio clip being too short! Recommend to use the latest feature instead!")
            Feature_List_per_session.append(Feature_List_per_session[-1])
        else:
            temp_list = []
            for num in list_num:
                num = float(num)
                temp_list.append(num)
                if (len(temp_list) == len(list_num)):
                    if(len(temp_list) < AVERAGE_N_ELEMENTS): # if number of values is less than 96, we still take average of all values and convert it to an one-second clip
                        append_list = [np.average(temp_list)]
                        Feature_List_per_session.append(append_list)
                    else:
                        residue = len(temp_list) % AVERAGE_N_ELEMENTS
                        temp_list = temp_list[:len(temp_list)-residue]
                        temp_list = np.average(np.array(temp_list).reshape(-1, AVERAGE_N_ELEMENTS), axis=1) # if we set the AVERAGE_N_ELEMENTS is too large, when the audio is too short, will encounter the DivisionbyZero error
                        temp_list = temp_list.tolist()
                        Feature_List_per_session.append(temp_list)
    Feature_List.append(Feature_List_per_session)

# load features from speaker 2
feature_corpus = 'E:\Github Repos\Project-Code\Regular-Workspace\Satisfaction_Modeling\Audio_Features' + FEATURE
speaker_2 = 's2'
feature_list = os.listdir(feature_corpus)

# load features from speaker 1
for feature_file in feature_list:
    Feature_List_per_session = []

    loud_list = pd.read_csv(feature_corpus + "\\" + feature_file + "\\" + speaker_2 + '_remove_eou_silence.csv.csv', delimiter='\t', header=None).values.tolist()
    # loud_list = pd.read_csv(feature_corpus + "\\" + feature_file + "\\" + speaker_2 + '_remove_eou_silence.csv', delimiter='\t', header=None).values.tolist()
    for list in loud_list:
        list_num = list[0].strip('][').split(', ')
        if (list_num[0][0] == 'n'):  # ['nan']
            # print("ERROR! [nan] feature reported due to the audio clip being too short! Recommend to use the latest feature instead!")
            Feature_List_per_session.append(Feature_List_per_session[-1])
        else:
            temp_list = []
            for num in list_num:
                num = float(num)
                temp_list.append(num)
                if (len(temp_list) == len(list_num)):
                    if (len(
                            temp_list) < AVERAGE_N_ELEMENTS):  # if number of values is less than 96, we still take average of all values and convert it to an one-second clip
                        append_list = [np.average(temp_list)]
                        Feature_List_per_session.append(append_list)
                    else:
                        residue = len(temp_list) % AVERAGE_N_ELEMENTS
                        temp_list = temp_list[:len(temp_list) - residue]
                        temp_list = np.average(np.array(temp_list).reshape(-1, AVERAGE_N_ELEMENTS), axis=1)  # if we set the AVERAGE_N_ELEMENTS is too large, when the audio is too short, will encounter the DivisionbyZero error
                        temp_list = temp_list.tolist()
                        Feature_List_per_session.append(temp_list)
    Feature_List.append(Feature_List_per_session)

#########################################################################################################################################

MAX_SEQ = 0 # MAX sequence per utterance time length

for session in Feature_List: # this funtion gets the max sequence length of each utterance
    for utterance in session:
        if len(utterance) > MAX_SEQ:
            MAX_SEQ = len(utterance)
        else:
            MAX_SEQ = MAX_SEQ

MAX_UTT = 0 # MAX utterance per session

for session in Feature_List:
    if len(session) > MAX_UTT:
        MAX_UTT = len(session)
    else:
        MAX_UTT = MAX_UTT

print("MAX utterance per session", MAX_UTT)
print("MAX sequence per utterance time length", MAX_SEQ)

for session in Feature_List: # zero-post padding to each speaking utterance
    # zero late padding for each session upto the maximum utterance number
    if (len(session) < MAX_UTT):
        session += ([[0.0]*MAX_SEQ] * (MAX_UTT - len(session)))
for session in Feature_List: # zero-post padding to each speaking utterance
    # zero late padding for each utterance upto the maximum utterance time length
    for utterance in session:
        if (len(utterance) < MAX_SEQ):
            utterance += [0.0] * (MAX_SEQ - len(utterance))
print("The shape of the feature list is: SESSION_NUM * SESSION_UTTERANCE * UTTERANCE_LENGTH", len(Feature_List), len(Feature_List[0]), len(Feature_List[0][0]))

##################### generating random numbers for the baseline model ################################
# import random
#
# New_Feature_List = []
# for session in Feature_List:
#     new_session = []
#     for utterance in session:
#         new_utterance = []
#         for unit in utterance:
#             new_item = random.uniform(0, 1)
#             new_utterance.append(new_item)
#         new_session.append(new_utterance)
#     New_Feature_List.append(new_session)
######################################################################################################

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
# shuffled_index_list = [36, 4, 0, 17, 35, 32, 22, 19, 5, 39, 10, 2, 33, 23, 38, 44, 3, 31, 18, 45, 16, 1, 43, 8, 21, 15, 14, 37, 24, 11, 27, 41, 42, 30, 7, 6, 9, 13, 40, 25, 28, 12, 20, 34, 26, 29]
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
from tensorflow.keras.layers import Input, BatchNormalization, Dropout, Dense, LSTM, GRU
from tensorflow.keras import Sequential, Model
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split

def creatModel():
    global model
    model = Sequential([
        Input(shape=(MAX_UTT, MAX_SEQ)),
        LSTM(units=128, return_sequences=True),
        LSTM(units=128, return_sequences=False),
        Dense(units=64, kernel_initializer='glorot_uniform', activation='relu'),
        Dense(units=1, activation='sigmoid', kernel_regularizer=l2(0.001))
    ])
    model.build(input_shape=(None, MAX_UTT, MAX_SEQ))
    model.compile(loss='mean_absolute_error', optimizer=tf.optimizers.Adam(lr=0.001))
    print(model.summary())

creatModel()

print("start training...")
history = model.fit(
    feature_train,
    score_train,
    batch_size=len(feature_train),
    epochs=500,
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
    output = intermediate_layer_model.predict(feature_vali)
    print(output)
    print("##################################################################")
    if (layer_index == 3):
        new_list = []
        for item in output:
            new_list.append(item[0])
        print(new_list)