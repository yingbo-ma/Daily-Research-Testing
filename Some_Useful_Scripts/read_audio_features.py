import csv
import os

import warnings
warnings.filterwarnings(action='ignore')

pair_info = "\\Feb2019_G43"
acoustic_feature_path = r"E:\Research Data\ENGAGE\ENGAGE Recordings" + pair_info + "\\clean_data" + "\\audio_features"
file_list = os.listdir(acoustic_feature_path)
saved_feature_path = "./acoustic_feature_data" + pair_info + ".csv"

feature_list = []

for file_index in range(0, len(file_list)):
    # attention! we can't use [for file in file_list], because this is not on the time sequential order but from numeric order like 0, 1, 10, 11, 12, ...
    print(file_index)
    file_name = str(file_index) + ".csv"
    acoustic_feature_list = []
    f = open(acoustic_feature_path+"\\"+file_name)
    csv_reader_object = csv.reader(f)
    for line in csv_reader_object: # we set the average time for audios are 10 seconds
        acoustic_feature_list.extend(line)
    acoustic_feature_list = acoustic_feature_list[:10*128]
    ## add extra '0' to make list length consistent 1280
    if (len(acoustic_feature_list) < 10*128):
        acoustic_feature_list = acoustic_feature_list + [0] * (10*128 - len(acoustic_feature_list))
    feature_list.append(acoustic_feature_list)

with open(saved_feature_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(feature_list))