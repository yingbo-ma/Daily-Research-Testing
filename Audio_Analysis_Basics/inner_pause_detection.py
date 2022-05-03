import os
from pydub import AudioSegment, silence

pair_info = "\\Feb2019_G43"
speaker = "\\s2"
audio_corpus_path = r"E:\Research Data\ENGAGE\ENGAGE Recordings" + pair_info + "\\clean_data\\individual_satisf_study" + speaker + "\\audio_clips\\"
audio_out_path = r"E:\Research Data\ENGAGE\ENGAGE Recordings" + pair_info + "\\clean_data\\individual_satisf_study" + speaker + "\\audio_clips_remove_eou_silence-6\\"
# read acoustic feature
import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

audio_file_list = os.listdir(audio_corpus_path)
audio_file_list = sorted_alphanumeric(audio_file_list)

for audio_index in range(len(audio_file_list)):

    audio_file_name = audio_file_list[audio_index]
    audio_file_path = audio_corpus_path + audio_file_name

    print("reading " + audio_file_name + " info!")
    myaudio = AudioSegment.from_wav(audio_file_path)
    speech = silence.detect_nonsilent(myaudio, min_silence_len=200, silence_thresh=-6)
    speech = [((start / 1000), (stop / 1000)) for start, stop in speech]  # convert to sec
    if (len(speech) == 0): # if no speech was detected
        myaudio.export(audio_out_path + "\{}.wav".format(audio_index), format="wav")
        print("exporting edited " + audio_file_name + " done!")
    else:
        first_speech_starting_point = list(speech[0])[0]
        last_speech_ending_point = list(speech[-1])[1]
        speech_remove_eou = myaudio[first_speech_starting_point * 1000 : last_speech_ending_point * 1000]
        speech_remove_eou.export(audio_out_path + "\{}.wav".format(audio_index), format="wav")
        print("exporting edited " + audio_file_name + " done!")