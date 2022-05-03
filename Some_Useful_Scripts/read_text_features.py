import xlrd
import numpy as np
import csv

import re

from support_functions import decontracted

import warnings
warnings.filterwarnings(action='ignore')

pair_info = "\\Feb2019_G43"
csv_file_name = pair_info+"_final_version.xlsx"
corpus_path = r"E:\Research Data\ENGAGE\ENGAGE Recordings" + pair_info + "\\clean_data" + csv_file_name
saved_text_path = "./csv_data/data" + pair_info + ".csv"
saved_label_path = "./csv_data/label" + pair_info + ".csv"

# read raw data from .xlsx file
Raw_Text_List = []
Clean_Text_List = []

tag_list = []

book = xlrd.open_workbook(corpus_path)
sheet = book.sheet_by_index(0)

for row_index in range(1, sheet.nrows-1): # skip heading and 1st row, because the audio parse code, we need to discard the last one sentence
    time, speaker, text, tag = sheet.row_values(row_index, end_colx=4)

    if speaker == 'Other':
        print('row index of', row_index, 'is removed becaused of Other Speaker.')
    elif text == '(())':
        print('row index of ', row_index, 'is removed becaused of Null Text.')
    elif type(text) == int or type(text) == float:
        text = str(text)
        Raw_Text_List.append(text)
        if ('Impasse' in tag):
            tag_list.append(1)
        else:
            tag_list.append(0)
    else:
        Raw_Text_List.append(text)
        if ('Impasse' in tag):
            tag_list.append(1)
        else:
            tag_list.append(0)

# start preprocessing
for text_index in range(len(Raw_Text_List)):

    utterance = Raw_Text_List[text_index]

    # step 1: replace all digital numbers with consistent string of 'number'
    utterance = re.sub(r'\d+', 'number', utterance)
    # step 2: convert text to lowercase
    utterance = utterance.lower()
    # step 3: solve contractions
    utterance = decontracted(utterance)
    # step 4: remove '()' several times. '((())) happens in text corpus'
    utterance = re.sub('[()]', '', utterance)
    # step 5: remove '[]' and contents inside it
    utterance = re.sub("([\[]).*?([\]])", "", utterance)
    # step 6: remove specific symbols, keep ? and . because it conveys the question and answer info
    utterance = re.sub(r'[--]', '', utterance) # '--'
    utterance = re.sub(r'[\\]', '', utterance)  # '\'
    utterance = re.sub(r'[\']', '', utterance)  # '''
    utterance = re.sub(r'[!]', '', utterance)  # '!'
    utterance = re.sub(r'["]', '', utterance)  # '"'
    utterance = re.sub(r'[{]', '', utterance)  # '{"}'
    utterance = re.sub(r'[}]', '', utterance)  # '}'
    # step 7: remove '...'
    if '...' in utterance:
        utterance = utterance.replace('...', " ") # Do we also want to remove . in each sentence? because we need to tell the network where is the switch point of the utterance-pair
    # step 8: remove white space
    utterance = utterance.strip()
    Clean_Text_List.append(utterance)

tag_list = tag_list[:len(Clean_Text_List)]

with open(saved_text_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(Clean_Text_List))

with open(saved_label_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(tag_list))

print(len(Clean_Text_List)*2-1)
print(len(tag_list)*2-1)