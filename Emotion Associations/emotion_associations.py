import pandas as pd
from WEAT import WEAT

#Define attribute word stimuli
person = ['person','woman','human','human being','individual','adult']
happy = ['happy person','happy woman','happy human','happy human being','happy individual','happy adult']
sad = ['sad person','sad woman','sad human','sad human being','sad individual','sad adult']
angry = ['angry person','angry woman','angry human','angry human being','angry individual','angry adult']

#Place attribute words in prompts
person_ = person + [f'a {i}' for i in person] + [f'a photo of a {i}' for i in person] + [f'an image of a {i}' for i in person] + [f'a picture of a {i}' for i in person]
happy_ = happy + [f'a {i}' for i in happy] + [f'a photo of a {i}' for i in happy] + [f'an image of a {i}' for i in happy] + [f'a picture of a {i}' for i in happy]
sad_ = sad + [f'a {i}' for i in sad] + [f'a photo of a {i}' for i in sad] + [f'an image of a {i}' for i in sad] + [f'a picture of a {i}' for i in sad]
angry_ = angry + [f'an {i}' for i in angry] + [f'a photo of an {i}' for i in angry] + [f'an image of an {i}' for i in angry] + [f'a picture of an {i}' for i in angry]

CLIP_MODEL = 'RN50x4'

#Read in SOBEM and language embeddings
emb_df = pd.read_csv(f'sobem/sobem_clip_{CLIP_MODEL}_emb_df.vec',sep=' ',index_col=0)
lang_df = pd.read_csv(f'sobem/sobem_clip_{CLIP_MODEL}_emb_lang_df.vec',sep=' ',index_col=0)

#Obtain embeddings for person words with no emotional description
no_emotion_embs_attribute = [lang_df.loc[i].to_numpy() for i in person_]

#Obtain embeddings for nonobjectified (NO) and objectified (O) targets
non_objectified = [i for i in emb_df.index.tolist() if '_NO_' in i]
objectified = [i for i in emb_df.index.tolist() if '_O_' in i]

#Define emotions over which to iterate - AN = Anger, SA = Sadness, HA = Happiness, 2 = High Emotion, 1 = Low Emotion
EMOTIONS = ['AN2','AN1','SA2','SA1','HA2','HA1']
embs_ = [angry_,angry_,sad_,sad_,happy_,happy_]

# EMOTIONS = ['HA2','HA1']
# embs_ = [happy_,happy_]

#Iterate over each emotion
for idx in range(len(EMOTIONS)):

    emotion = EMOTIONS[idx]
    emotion_emb_stimuli = embs_[idx]

    #Obtain embeddings for person words with an emotional description
    emotion_embs_attribute = [lang_df.loc[i].to_numpy() for i in emotion_emb_stimuli]

    #Obtain nonobjectified and objectified embeddings where the target displays the emotion
    non_objectified_target = [emb_df.loc[i].to_numpy() for i in non_objectified if emotion in i]
    objectified_target = [emb_df.loc[i].to_numpy() for i in objectified if emotion in i]

    #Effect size and p-value from the WEAT
    es,p = WEAT(emotion_embs_attribute,no_emotion_embs_attribute,non_objectified_target,objectified_target,1000)
    print(f'{CLIP_MODEL} {emotion}: {es}, {p}')