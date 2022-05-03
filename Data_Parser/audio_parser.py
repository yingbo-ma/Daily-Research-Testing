from pydub import AudioSegment
import xlrd

if __name__ == '__main__':

    audio_source_file = r"E:\Research Data\ENGAGE\ENGAGE Recordings\Dec4-2019 - t012 t065\raw_data\9_Group.wav"
    audio_out_path = r"E:\Research Data\ENGAGE\ENGAGE Recordings\Dec4-2019 - t012 t065\clean_data\audio_clips"
    transcription_path = r"E:\Research Data\ENGAGE\ENGAGE Recordings\Dec4-2019 - t012 t065\clean_data\Dec4-2019 - t012 t065_final_version.xlsx"

    audio = AudioSegment.from_wav(audio_source_file)
    # prepare the segment time stamps according to the ground truth time stamps, read raw data from .xlsx file
    Raw_TimeStep_List = []
    Speaker_List = []

    book = xlrd.open_workbook(transcription_path)
    sheet = book.sheet_by_index(0)

    for row_index in range(1, sheet.nrows):  # skip heading and 1st row
        time, speaker, text = sheet.row_values(row_index, end_colx=3)
        Raw_TimeStep_List.append(time)
        Speaker_List.append(speaker)

    for index in range(len(Speaker_List)-1):
        speaker_1 = Speaker_List[index]
        speaker_2 = Speaker_List[index+1]
        if(speaker_1 == speaker_2):
            print("ERROE! Repreated Speaker! Error Index is " + str(index+2))

    print("Speaker Information Checked!")

    segRange = []  # a list of starting/ending frame indices pairs
    for time_index in range(len(Raw_TimeStep_List) - 2):
        temp_list = []
        str_start_time = Raw_TimeStep_List[time_index]
        str_end_time = Raw_TimeStep_List[time_index + 2]

        start_time_split = str_start_time.split(':')
        start_time = 60 * int(start_time_split[0]) + int(float(start_time_split[1]))

        end_time_split = str_end_time.split(':')
        end_time = 60 * int(end_time_split[0]) + int(float(end_time_split[1]))

        temp_list.append(start_time*1000)
        temp_list.append(end_time*1000)
        segRange.append(temp_list)
    print(segRange)
    print(len(segRange))

    for idx, [begin, end] in enumerate(segRange):
        print("Parsing " + str(idx) + " clip...")
        audio_chunk = audio[begin:end]
        audio_chunk.export(audio_out_path+"\{}.wav".format(idx), format="wav")
        begin = end