import csv
import os

import warnings
warnings.filterwarnings(action='ignore')
import statistics

import numpy as np

pair_info = "\\Feb2019_G43"
audio_feature_path = r"E:\Research Data\ENGAGE\ENGAGE Recordings" + pair_info + "\\clean_data" + "\\video_features"
file_list = os.listdir(audio_feature_path)
saved_feature_path = "./visual_feature_data" + pair_info + ".csv"

visual_feature_list = []

for file_index in range(0, len(file_list)):
    # attention! we can't use [for file in file_list], because this is not on the time sequential order but from numeric order like 0, 1, 10, 11, 12, ...
    print(file_index)
    file_name = str(file_index) + ".csv"
    f = open(audio_feature_path + "\\" + file_name)
    csv_reader_object = csv.reader(f)
    next(csv_reader_object)

    feature_list = []
    frame_list = []

    current_frame = 1
    stored_faces = 0

    for line in csv_reader_object: # 30 frames for 1 second

        if (int(line[0]) == current_frame and stored_faces < 2):

            visual_feature = []

            head_pose_feature = line[293:298]
            intensity_of_AUs = line[679:695]
            presence_of_AUs = line[696:713]

            head_pose_feature = list(map(float, head_pose_feature)) # convert data type from str to float, we need to calculate the average number later
            intensity_of_AUs = list(map(float, intensity_of_AUs))
            presence_of_AUs = list(map(float, presence_of_AUs))

            visual_feature.append(line[0])
            visual_feature.append(head_pose_feature)  # head movement features
            visual_feature.append(intensity_of_AUs)  # intensity (from 0 to 5) of 17 AUs
            visual_feature.append(presence_of_AUs)  # the presense (0 absent, 1 present) of 18 AUs

            print(len(visual_feature))

            stored_faces += 1
            if(stored_faces == 2):
                current_frame += 1
                stored_faces = 0

            feature_list.append(visual_feature)

        elif (int(line[0]) > current_frame and stored_faces == 1):
            # print("ERROR! Only one faces detected in the last frame")
            feature_list.append(feature_list[-1]) # duplicate last facial features to make last frame intact

            visual_feature = []

            head_pose_feature = line[293:298]
            intensity_of_AUs = line[679:695]
            presence_of_AUs = line[696:713]

            head_pose_feature = list(map(float, head_pose_feature))
            intensity_of_AUs = list(map(float, intensity_of_AUs))
            presence_of_AUs = list(map(float, presence_of_AUs))

            visual_feature.append(line[0])
            visual_feature.append(head_pose_feature)  # head movement features
            visual_feature.append(intensity_of_AUs)  # intensity (from 0 to 5) of 17 AUs
            visual_feature.append(presence_of_AUs)  # the presense (0 absent, 1 present) of 18 AUs

            feature_list.append(visual_feature) # add current first facial features for the current frame

            stored_faces = 1 # for the current frame we already stored one face now
            current_frame += 1

        elif (int(line[0]) > current_frame and stored_faces == 0):
            # print("ERROE! The frames are not consecutive!")
            feature_list = feature_list

        elif (int(line[0]) < current_frame):
            # print("Non-Related Faces! Discarded Automatically!")
            feature_list = feature_list

    while (len(feature_list) & 2 != 0):
        feature_list.append(feature_list[-1])

    # print("Frame Number is: ", len(feature_list) / 2)

    pair_feature_list = []

    for frame_index in range(0, len(feature_list), 2):
        pair_feature = feature_list[frame_index][1:] + feature_list[frame_index + 1][1:] # pair two face ids into one single feature list, abandon frame id here
        pair_feature_list.append(pair_feature)

    fps = 30  # default fps value, 30
    final_feature_list = [pair_feature_list[i * fps:(i + 1) * fps] for i in
                          range((len(pair_feature_list) + fps - 1) // fps)]

    total_flattened_feature_list = []  # this is the flattened feature lists frame-by-frame
    for index in range(len(final_feature_list)):
        feature_list_in_frame = final_feature_list[index]
        flattened_frame_feature_list = []  # flattened feature lists for 30 frames
        for frame_index in range(len(feature_list_in_frame)):
            frame_feature_list = feature_list_in_frame[frame_index]
            flattened_frame_feature = []
            for features in frame_feature_list:
                flattened_frame_feature += features
            flattened_frame_feature_list.append(flattened_frame_feature)
        total_flattened_feature_list.append(flattened_frame_feature_list)

    # calculate avaraged values of the feature list
    avareaged_feature_lists = []
    for features in total_flattened_feature_list:
        avareaged_list = list(map(statistics.mean, zip(*features)))
        mid_np = np.array(avareaged_list)  # only save 2 numbers after the point
        mid_np_2f = np.round(mid_np, 2)  #
        avareaged_list_new = list(mid_np_2f)
        avareaged_feature_lists.extend(avareaged_list_new)

    # print("Averaged Facial Feature List Length is: ", len(avareaged_feature_lists))

    print(avareaged_feature_lists)
    visual_feature_list.append(avareaged_feature_lists)

    # with open(saved_feature_path, 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(zip(visual_feature_list))