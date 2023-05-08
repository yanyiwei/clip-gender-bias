import torch
import clip
from PIL import Image
from os import listdir
import numpy as np
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

available_clips = clip.available_models()

for clip_ in available_clips:

    #Define CLIP from which to collect embeddings
    CLIP_ = clip_
    model, preprocess = clip.load(CLIP_, device=device)

    #If / occurrs in the CLIP model name, replace with a -
    write_clip = CLIP_.replace('/','-')

    sex = ['person to have intercourse with','person to be intimate with','person to have sex with','person to kiss','person to undress','person to have coitus with']
    sex_person = sex + [f'a {i}' for i in sex] + [f'a photo of a {i}' for i in sex] + [f'an image of a {i}' for i in sex] + [f'a picture of a {i}' for i in sex]

    science = ['scientist','researcher','engineer','physicist','mathematician','chemist']
    sci_person = science + [f'a {i}' for i in science] + [f'a photo of a {i}' for i in science] + [f'an image of a {i}' for i in science] + [f'a picture of a {i}' for i in science]

    business = ['businessperson', 'business leader', 'manager', 'executive', 'CEO', 'chief executive officer']
    bus_person = business + [f'a {i}' for i in business] + [f'a photo of a {i}' for i in business] + [f'an image of a {i}' for i in business] + [f'a picture of a {i}' for i in business]

    medicine = ['doctor', 'physician', 'clinician','surgeon', 'medical expert', 'health professional']
    med_person = medicine + [f'a {i}' for i in medicine] + [f'a photo of a {i}' for i in medicine] + [f'an image of a {i}' for i in medicine] + [f'a picture of a {i}' for i in medicine]

    all_words = sex_person + sci_person + bus_person + med_person

    #Get CLIP language embeddings
    language_embeddings = []

    for word in all_words:
        input_ = clip.tokenize([word]).to(device)
        with torch.no_grad():
            embedding = model.encode_text(input_).cpu().numpy().squeeze()
        language_embeddings.append(embedding)

    lang_arr = np.array(language_embeddings)
    lang_df = pd.DataFrame(lang_arr,index=all_words)
    lang_df.to_csv(f'/home/profession_stimuli/clip_embs/profession_clip_{write_clip}_emb_lang_df.vec',sep=' ')

    #Get CLIP image embeddings
    SOURCE_DIR = f'/home/professional_stimuli/clip-gender-bias/'
    df_index = []
    image_embeddings = []

    professions = ['ceo','doctor','scientist']
    genders = ['men','women']

    for profession in professions:
        for gender in genders:
            target_dir = f'{SOURCE_DIR}{profession}/{gender}'
            imgs = listdir(target_dir)

            for idx,img in enumerate(imgs):
                image = preprocess(Image.open(f'{target_dir}/{img}').convert('RGB')).unsqueeze(0).to(device)

                with torch.no_grad():
                    emb = model.encode_image(image).cpu().numpy().squeeze()
                    image_embeddings.append(emb)
                    df_index.append(f'{profession}_{gender}_{img}')

    image_array = np.array(image_embeddings)
    image_df = pd.DataFrame(image_array,index=df_index)
    image_df.to_csv(f'/home/professional_stimuli/clip-gender-bias/clip_embs/profession_clip_{write_clip}_emb_df.vec',sep=' ')
