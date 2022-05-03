import os
import pandas as pd
import re

import gensim
from gensim.models import Word2Vec

text_corpus_path = '.\\csv_data\\data\\'
label_corpus_path = '.\\csv_data\\label\\'

text_file_list = os.listdir(text_corpus_path)
label_file_list = os.listdir(label_corpus_path)

# prepare turn exchanges
turn_pairs = []
label_list = []

for text_file in text_file_list:

    turn_list = pd.read_csv(text_corpus_path + text_file, delimiter='\t', header=None).values.tolist()

    for turn_index in range(len(turn_list) - 1):
        cur_turn_text = turn_list[turn_index][0]
        nex_turn_text = turn_list[turn_index + 1][0]
        turn_pair_text = cur_turn_text + ' ' + nex_turn_text
        turn_pairs.append(turn_pair_text)

###### feature selection 1: sentence length, and sentiment ######
clean_corpus = []

for turn_exchange in turn_pairs:

    # remove important punctuations such as '.' and '?'
    turn_exchange = re.sub(r'[,]', '', turn_exchange)
    turn_exchange = re.sub(r'[.]', '', turn_exchange)
    turn_exchange = re.sub(r'[?]', '', turn_exchange)
    turn_exchange = turn_exchange.split()
    clean_corpus.append(turn_exchange)

# train model

model = Word2Vec(clean_corpus, min_count=1)
print(model.most_similar(positive=['flag'], topn=10))
# summarize the loaded model

feature_vector_list = []
MAX_SEQ = 0

for sentence in clean_corpus:
    sentence_feature = []
    for word in sentence:
        word_vector_score = model[word]
        sentence_feature.extend(word_vector_score)
    if(MAX_SEQ < len(sentence_feature)):
        MAX_SEQ = len(sentence_feature)
    feature_vector_list.append(sentence_feature)

print("MAX_SEQ", MAX_SEQ)
print(len(feature_vector_list))


for label_file in label_file_list:
    cur_label_list = pd.read_csv(label_corpus_path + label_file, delimiter='\t', header=None).values.tolist()
    cur_label_list = cur_label_list[:-1]
    label_list += cur_label_list

print(len(label_list))

AVE_SEQ = 0
SEQ_SUM = 0
for list in feature_vector_list:
    SEQ_SUM += len(list)
AVE_SEQ = int(SEQ_SUM / len(feature_vector_list))
print("AVE_SEQ", AVE_SEQ)

MAX_SEQ = 2000
New_feature_vector_list = []

for list in feature_vector_list:
    if (len(list) < MAX_SEQ):
        list += [0.0] * (MAX_SEQ - len(list))
    elif (len(list) > MAX_SEQ):
        list = list[:MAX_SEQ]
    else:
        print("No Action Needed!")

    New_feature_vector_list.append(list)

# ##### build a simple SVM model #####

train_data = New_feature_vector_list[:4000]
train_label = label_list[:4000]

test_data = New_feature_vector_list[4000:]
test_label = label_list[4000:]

from imblearn.under_sampling import RandomUnderSampler
undersample = RandomUnderSampler(sampling_strategy='majority')
X_under, y_under = undersample.fit_resample(train_data, train_label)

print("Step 3")

from sklearn import svm
clf = svm.SVC()
clf.fit(X_under, y_under)

predict_train = clf.predict(X_under)
predict_test = clf.predict(test_data)
from sklearn.metrics import classification_report

print("the training result")
print(classification_report(y_under, predict_train))

print("the testing result")
print(classification_report(test_label, predict_test))




































