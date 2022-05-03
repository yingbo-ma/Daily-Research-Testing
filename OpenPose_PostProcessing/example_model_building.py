# a script for building a learning model from postprocessed features
import os
import pandas as pd
import numpy as np

loudness_feature_corpus = 'E:\\Github Repos\\Project-Code\\Regular-Workspace\\LAK\\Visual Models\\csv_data\\head_pose\\'
label_corpus = 'E:\\Github Repos\\Project-Code\\Regular-Workspace\\LAK\\Acoustic Models\\csv_data\\label\\'

feature_list = os.listdir(loudness_feature_corpus)
label_list = os.listdir(label_corpus)

Head_Pose_Feature_List = []
Label_List = []

MAX_SEQ = 0

print("Step 1")

for feature_file in feature_list:
    eye_gaze_list = pd.read_csv(loudness_feature_corpus + feature_file, delimiter='\t', header=None).values.tolist()
    for list in eye_gaze_list:
        new_list = list[0].strip('][').split(', ')
        if (len(new_list) == 1):  # in case the visual feature list is empty
            Head_Pose_Feature_List.append(Head_Pose_Feature_List[-1])  # if empty we append the lastest one directly
        else:
            temp_list = []
            for number in new_list:
                if ("'" in number):
                    number = float(number[1:-1])
                else:
                    number = float(number)
                temp_list.append(number)
            list_len = len(new_list)
            if list_len > MAX_SEQ:
                MAX_SEQ = list_len
            else:
                MAX_SEQ = MAX_SEQ
            Head_Pose_Feature_List.append(temp_list)

print("MAX_SEQ", MAX_SEQ)
print(len(Head_Pose_Feature_List))

AVE_SEQ = 0
SEQ_SUM = 0
for list in Head_Pose_Feature_List:
    SEQ_SUM += len(list)
AVE_SEQ = int(SEQ_SUM / len(Head_Pose_Feature_List))
print("AVE_SEQ", AVE_SEQ)

MAX_SEQ = AVE_SEQ
New_Head_Pose_Feature_List = []

for list in Head_Pose_Feature_List:
    if (len(list) < MAX_SEQ):
        list += [0.0] * (MAX_SEQ - len(list))
    elif (len(list) > MAX_SEQ):
        list = list[:MAX_SEQ]
    else:
        print("No Action Needed!")

    New_Head_Pose_Feature_List.append(list)

for label_file in label_list:
    tag_list = pd.read_csv(label_corpus + label_file, delimiter='\t', header=None).values.tolist()
    for list in tag_list:
        Label_List.append(list)

print(len(Label_List))

# ##### build a simple SVM model #####

train_data = New_Head_Pose_Feature_List[:4000]
train_label = Label_List[:4000]

test_data = New_Head_Pose_Feature_List[4000:]
test_label = Label_List[4000:]

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