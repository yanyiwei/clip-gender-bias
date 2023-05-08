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
    print(CLIP_)

    #If / occurrs in the CLIP model name, replace with a -
    write_clip = CLIP_.replace('/','-')

    #Define text stimuli
    female = ['female', 'woman', 'girl', 'sister', 'she', 'her', 'hers', 'daughter']
    male = ['male', 'man', 'boy', 'brother', 'he', 'him', 'his', 'son']
    all_words = female + male

    #Get CLIP language embeddings
    language_embeddings = []

    for word in all_words:
        input_ = clip.tokenize([word]).to(device)
        with torch.no_grad():
            embedding = model.encode_text(input_).cpu().numpy().squeeze()
        language_embeddings.append(embedding)

    lang_arr = np.array(language_embeddings)
    lang_df = pd.DataFrame(lang_arr,index=all_words)
    lang_df.to_csv(f'/home/oasis/new_/clip_{write_clip}_emb_df_language.vec',sep=' ')

    #Get CLIP image embeddings
    SOURCE_DIR = f'/home/oasis/Images'
    imgs = listdir(SOURCE_DIR)

    image_embeddings = []

    for idx,img in enumerate(imgs):
        image = preprocess(Image.open(f'{SOURCE_DIR}/{img}').convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(image).cpu().numpy().squeeze()
            image_embeddings.append(emb)

    image_arr = np.array(image_embeddings)
    image_df = pd.DataFrame(image_arr,index=imgs)
    image_df.to_csv(f'/home/oasis/new_/clip_{write_clip}_emb_df.vec',sep=' ')
