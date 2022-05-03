import cv2
import xlrd

if __name__ == '__main__':

    # set video input and output path
    vidPath = r'E:\Research Data\ENGAGE\ENGAGE Recordings\Feb2019_G43\raw_data\G43.mp4'
    shotsPath = r'E:\Research Data\ENGAGE\ENGAGE Recordings\Feb2019_G43\clean_data\video_clips\%d.avi' # output path (must be avi, otherwize choose other codecs)
    transcription_path = r"E:\Research Data\ENGAGE\ENGAGE Recordings\Feb2019_G43\clean_data\Copy of Feb2019_G43_final_version.xlsx"

    # opencv prepare reading the video
    cap = cv2.VideoCapture(vidPath)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # prepare the segment time stamps according to the ground truth time stamps, read raw data from .xlsx file
    Raw_TimeStep_List = []
    Speaker_List = []

    book = xlrd.open_workbook(transcription_path)
    sheet = book.sheet_by_index(0)

    for row_index in range(1, sheet.nrows):  # skip heading and 1st row
        time, speaker, text = sheet.row_values(row_index, end_colx=3)
        Raw_TimeStep_List.append(time)
        Speaker_List.append(speaker)

    print(Raw_TimeStep_List)

    for index in range(len(Speaker_List)-1):
        speaker_1 = Speaker_List[index]
        speaker_2 = Speaker_List[index+1]
        if(speaker_1 == speaker_2):
            print("ERROE! Repreated Speaker! Error Index is " + str(index+2))

    print("Speaker Information Checked!")

    segRange = [] # a list of starting/ending frame indices pairs
    for time_index in range(len(Raw_TimeStep_List) - 2):
        temp_list = []
        str_start_time = Raw_TimeStep_List[time_index]
        str_end_time = Raw_TimeStep_List[time_index + 2] # because the timestamp is only the start time, therefore index should increase 2 instead of 1 to include a whole conversation, otherwise it is only the first utterance

        start_time_split = str_start_time.split(':')
        start_time = 60 * int(start_time_split[0]) + int(float(start_time_split[1]))
        end_time_split = str_end_time.split(':')
        end_time = 60 * int(end_time_split[0]) + int(float(end_time_split[1]))

        temp_list.append(start_time*fps)
        temp_list.append(end_time*fps)
        segRange.append(temp_list)
    print(segRange)
    print(len(segRange))

    # start parsing videos
    for idx,[begFidx,endFidx] in enumerate(segRange):
        print("Parsing " + str(idx) + " clip...")
        writer = cv2.VideoWriter(shotsPath%idx,fourcc,fps,size)
        cap.set(cv2.CAP_PROP_POS_FRAMES,begFidx)
        ret = True # has frame returned
        while(cap.isOpened() and ret and writer.isOpened()):
            ret, frame = cap.read()
            frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
            if frame_number < endFidx:
                writer.write(frame)
            else:
                break
        writer.release()
    print('Video Parsing Done!')