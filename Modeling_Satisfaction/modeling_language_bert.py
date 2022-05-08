import pickle
import numpy as np

with open('./Language_Features/BERT_Embeddings/bert_embeddings.pkl', 'rb') as pickle_load:
    BERT_Embeddings_List = pickle.load(pickle_load)

MAX_UTT = 0
MAX_SEQ = 768

for session in BERT_Embeddings_List:
    if len(session) > MAX_UTT:
        MAX_UTT = len(session)
    else:
        MAX_UTT = MAX_UTT

for session in BERT_Embeddings_List: # zero-post padding to each speaking utterance
    if (len(session) < MAX_UTT):
        session += ([[0.0]*MAX_SEQ] * (MAX_UTT - len(session)))

print("The shape of the feature list is: SESSION_NUM * SESSION_UTTERANCE * UTTERANCE_LENGTH", len(BERT_Embeddings_List), len(BERT_Embeddings_List[0]), len(BERT_Embeddings_List[0][0]))
print("The shape of the feature list is: SESSION_NUM * SESSION_UTTERANCE * UTTERANCE_LENGTH", len(BERT_Embeddings_List), len(BERT_Embeddings_List[1]), len(BERT_Embeddings_List[1][0]))
print("The shape of the feature list is: SESSION_NUM * SESSION_UTTERANCE * UTTERANCE_LENGTH", len(BERT_Embeddings_List), len(BERT_Embeddings_List[2]), len(BERT_Embeddings_List[2][0]))
print("The shape of the feature list is: SESSION_NUM * SESSION_UTTERANCE * UTTERANCE_LENGTH", len(BERT_Embeddings_List), len(BERT_Embeddings_List[3]), len(BERT_Embeddings_List[3][0]))
print("The shape of the feature list is: SESSION_NUM * SESSION_UTTERANCE * UTTERANCE_LENGTH", len(BERT_Embeddings_List), len(BERT_Embeddings_List[4]), len(BERT_Embeddings_List[4][0]))

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
    feature_train.append(BERT_Embeddings_List[index])
    score_train.append(Normalized_Score_List[index])

for index in vali_index:
    feature_vali.append(BERT_Embeddings_List[index])
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