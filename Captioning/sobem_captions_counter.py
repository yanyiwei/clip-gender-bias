from os import listdir
from collections import Counter

SOURCE_DIR = f'\\clip_32_captions'
#Each SOBEM image has a text file with 1,000 captions
targets = listdir(SOURCE_DIR)

#Nonobjectified individuals are denoted with NO in their filenames; Objectified individuals with O
non_objectified = [i for i in targets if '_NO_' in i]
objectified = [i for i in targets if '_O_' in i]

obj_conditions = [non_objectified,objectified]
obj_descriptions = ['Nonobjectified','Objectified']

#File names contain the emotions of the individual - AN = Anger, SA = Sadness, HA = Happiness, 1 = Low Emotion, 2 = High Emotion
emotions = ['AN1','AN2','SA1','SA2','HA1','HA2']

#Text stimuli denoting each emotion, including words added after examining the most frequent output
anger = ['frowning','frown','frowns','frowny','serious','unhappy','anger','angry','grimace','grimacing','scowl','scowling']
sadness = ['frown','frowning','frowns','frowny','crying','sad','sadness','unhappy','grimace','grimacing','serious','upset']
happiness = ['happy','smile','smiling','smiles','smiley','laughing']

emotes = [anger,anger,sadness,sadness,happiness,happiness]

#Iterate over nonobjectified, objectified states

for n, condition in enumerate(obj_conditions):
    current_condition = obj_descriptions[n]

    #Iterate through emotional states and select words for detection
    for k,emotion in enumerate(emotions):
        emote = emotes[k]

        #Select caption files for SOBEM individuals who display the current emotional condition
        subtargets = [i for i in condition if emotion in i]

        #Iterate through each caption file for an individual displaying the current emotional condition
        for idx,target in enumerate(subtargets):
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

        #Iterate over target words and get a count for the number of descriptions of the target emotion
        emotion_count = 0
        for word in emote:
            if word in counts:
                emotion_count += counts[word]

        #Print emotion, total count for emotion, and count per 1000 captions (20 files with 1000 captions each for each objectification + emotional condition)
        print(f'{current_condition} {emotion} Count: {emotion_count} Count Per 1000: {emotion_count/20}')