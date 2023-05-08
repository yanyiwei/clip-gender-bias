from os import listdir
from collections import Counter

SOURCE_DIR = f'\\clip_32_captions'
#Each SOBEM image has a text file with 1,000 captions
targets = listdir(SOURCE_DIR)

#Iterate through each caption file
for idx,target in enumerate(targets):
    with open(f'{SOURCE_DIR}\\{target}','r',encoding='utf8') as reader:
        text = reader.read().split('\n')
    
    #Lowercase text, remove periods, create sublist of words for each sentence
    sublists = [sentence.lower().replace('.','').replace(',','').split(' ') for sentence in text]

    #Join sublists into one list of words for the file
    joined = [item for sublist in sublists for item in sublist]

    #Get word counts in a Counter and add to total
    current_counts = Counter(joined)
    if idx == 0:
        counts = current_counts
    else:
        counts += current_counts

#Get all words occurring 100 or more times in captions
words = [key for key, val in counts.items() if val >= 100]

with open(f'C:\\Users\\wolfe\\Documents\\Research\\words.txt','w') as writer:
    writer.write('\n'.join(words))