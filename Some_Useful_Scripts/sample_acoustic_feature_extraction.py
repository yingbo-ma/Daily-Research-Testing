import csv
import os
import audiofile
import opensmile

pair_info = "\\Feb2019_G43"
speaker = "\\s2"
audio_corpus_path = r"E:\Research Data\ENGAGE\ENGAGE Recordings" + pair_info + "\\clean_data\\individual_satisf_study" + speaker + "\\audio_clips_remove_eou_silence\\"

saved_audio_loudness_path = "./Audio_Features/loudness" + pair_info + speaker + "_remove_eou_silence.csv"
saved_audio_spectralflux_path = "./Audio_Features/specflux" + pair_info + speaker + "_remove_eou_silence.csv"
saved_audio_mfcc_path = "./Audio_Features/mfcc" + pair_info + speaker + "_remove_eou_silence.csv"
saved_audio_jitter_path = "./Audio_Features/jitter" + pair_info + speaker + "_remove_eou_silence.csv"
saved_audio_shimmer_path = "./Audio_Features/shimmer" + pair_info + speaker + "_remove_eou_silence.csv"

import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

audio_file_list = os.listdir(audio_corpus_path)
audio_file_list = sorted_alphanumeric(audio_file_list)

Loudness_feature_list = []
Spectralflux_feature_list = []
MFCC_list = []
Jitter_feature_list = []
Shimmer_feature_list = []

for audio_index in range(len(audio_file_list)):

    audio_file_name = audio_file_list[audio_index]
    audio_file_path = audio_corpus_path + audio_file_name

    print("reading " + audio_file_name + " info!")

    signal, sampling_rate = audiofile.read(audio_file_path)

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors
    )

    X = smile.process_signal(signal, sampling_rate).values.tolist()

    print(len())
    print(len(X[0]))

    Loudness = []
    SpectralFlux = []
    Mfcc = []
    Jitter = []
    Shimmer = []

    for index in range(len(X)):
        Loudness_sma3 = X[index][0]
        Loudness.append(Loudness_sma3)

        jitterLocal_sma3nz = X[index][11]
        Jitter.append(jitterLocal_sma3nz)

        shimmerLocaldB_sma3nz = X[index][12]
        Shimmer.append(shimmerLocaldB_sma3nz)

        spectralFlux_sma3 = X[index][5]
        SpectralFlux.append(spectralFlux_sma3)

        mfcc1 = X[index][6]
        mfcc2 = X[index][7]
        mfcc3 = X[index][8]
        mfcc4 = X[index][9]

        Mfcc.append(mfcc1); Mfcc.append(mfcc2); Mfcc.append(mfcc3); Mfcc.append(mfcc4)

    Loudness_feature_list.append(Loudness)
    Spectralflux_feature_list.append(SpectralFlux)
    MFCC_list.append(Mfcc)
    Jitter_feature_list.append(Jitter)
    Shimmer_feature_list.append(Shimmer)

    print("read " + audio_file_name + " info done!")

with open(saved_audio_loudness_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(Loudness_feature_list))

with open(saved_audio_spectralflux_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(Spectralflux_feature_list))

with open(saved_audio_mfcc_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(MFCC_list))

with open(saved_audio_jitter_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(Jitter_feature_list))

with open(saved_audio_shimmer_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(Shimmer_feature_list))

print(len(Jitter_feature_list))
print(len(Shimmer_feature_list))
print(len(Loudness_feature_list))
print(len(Spectralflux_feature_list))
print(len(MFCC_list))