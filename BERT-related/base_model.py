from nltk.tokenize import sent_tokenize

import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import pandas as pd
import bert
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

MAX_SEQ_LEN = 40
CLASS_NUM = 2

print("\n**************************************************************************************************")
print("create bert layer...")
def createBertLayer(max_seq_length):
    global bert_layer
    currentDir = os.path.dirname(os.path.realpath(__file__))
    bertDir = os.path.join(currentDir, "models", "uncased_L-12_H-768_A-12")
    bert_params = bert.params_from_pretrained_ckpt(bertDir)
    bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert")
    model_layer = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name='input_ids'),
        bert_layer
    ])
    model_layer.build(input_shape=(None, max_seq_length))
    bert_layer.apply_adapter_freeze() # use this to use pre-trained BERT weights; otherwise the model will train the BERT model from the scratch
    # bert_layer.trainable = False

createBertLayer(MAX_SEQ_LEN)
print("done!")

print("\n**************************************************************************************************")
print("load bert checkpoint...")
def loadBertCheckpoint():
    currentDir = os.path.dirname(os.path.realpath(__file__))
    bertDir = os.path.join(currentDir, "models", "uncased_L-12_H-768_A-12")
    checkpointName = os.path.join(bertDir, "bert_model.ckpt")
    bert.load_stock_weights(bert_layer, checkpointName)

loadBertCheckpoint()
print("done!")

print("\n**************************************************************************************************")
print("create model...")
def createModel():
    global model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype='int32', name='input_ids'),
        bert_layer,
        tf.keras.layers.Lambda(lambda x: x[:, 0, :]),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(768, activation=tf.nn.leaky_relu),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    model.build(input_shape=(None, MAX_SEQ_LEN))
    model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(lr=0.00001), metrics=['accuracy'])
    print(model.summary())

createModel()
print("done!")

print("create tokenizer...")
def createTokenizer():
    currentDir = os.path.dirname(os.path.realpath(__file__))
    modelsFolder = os.path.join(currentDir, "models", "uncased_L-12_H-768_A-12")
    vocab_file = os.path.join(modelsFolder, "vocab.txt")
    tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case=True)
    return tokenizer

tokenizer = createTokenizer()
print("done!")

print("\n**************************************************************************************************")
print("loading textual data...")
# this data loading process is questionable, should we load them individually? because we don't have audio for the switch text between pairs
# or we create random audio fetures for the switch text between pairs?
data_list = pd.read_csv('./csv_data/data/data.csv', delimiter='\t', header=None).values.tolist()
label_list = pd.read_csv('./csv_data/label/label.csv', delimiter='\t', header=None).values.tolist()
label_list = label_list[:len(label_list)-1]

utterance_tokens = []
pair_tokens = []

for sentences in data_list:
    sentence_list = sent_tokenize(sentences[0])
    sentences_tokens = []
    for sentence in sentence_list:
        words = tokenizer.tokenize(sentence)
        words.append('[SEP]')
        sentences_tokens += words
    sentences_tokens += ['[EOT]']
    utterance_tokens.append(sentences_tokens)

for token_index in range(len(utterance_tokens)-1):
    current_token = utterance_tokens[token_index]
    next_token = utterance_tokens[token_index+1]
    turn_token = ['[CLS]'] + current_token + next_token
    turn_token = turn_token[:-1]
    pair_tokens.append(turn_token)

tokens_ids = [tokenizer.convert_tokens_to_ids(token) for token in pair_tokens]
padded_token_ids = pad_sequences(tokens_ids, maxlen=MAX_SEQ_LEN, dtype="long", truncating="post", padding="post")

print("\n**************************************************************************************************")
print("under-sampling majority class...")
undersample = RandomUnderSampler(sampling_strategy='majority')
X_train, y_train = padded_token_ids[:4000], label_list[:4000]
X_test, y_test = padded_token_ids[4001:], label_list[4001:]
X_under, y_under = undersample.fit_resample(X_train, y_train)
X_train, X_vali, y_train, y_vali = train_test_split(X_under, y_under, test_size=0.1)

X_train = np.array(X_train).reshape(-1, MAX_SEQ_LEN)
X_test = np.array(X_test).reshape(-1, MAX_SEQ_LEN)
y_train = np.array(y_train)
y_test = np.array(y_test)
print("done!")

print("\n**************************************************************************************************")
print("start training...")

history = model.fit(
    X_train,
    y_train,
    batch_size=16,
    epochs=10,
    validation_data=(X_test, y_test),
    verbose=1
)
print("Done!")

print("\n**************************************************************************************************")
print("Curves plotting...")
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

print("\n**************************************************************************************************")
print("confusion matrix on training set...")
y_pred = model.predict(X_train,verbose=0)
y_pred_list = []
for pred_index in range(len(y_pred)):
    if y_pred[pred_index][0] < 0.5:
        y_pred_list.append(0)
    elif y_pred[pred_index][0] > 0.5:
        y_pred_list.append(1)
    else:
        print("ERROR! pred probability equals to 0.5!")
print(classification_report(y_train, y_pred_list))

print("\n**************************************************************************************************")
print("confusion matrix on test set...")
y_pred = model.predict(X_test,verbose=0)
y_pred_list = []
for pred_index in range(len(y_pred)):
    if y_pred[pred_index][0] < 0.5:
        y_pred_list.append(0)
    elif y_pred[pred_index][0] > 0.5:
        y_pred_list.append(1)
    else:
        print("ERROR! pred probability equals to 0.5!")
print(classification_report(y_test, y_pred_list))

print(y_pred_list)


