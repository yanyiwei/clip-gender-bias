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

    person = ['person','woman','human','human being','individual','adult']
    happy = ['happy person','happy woman','happy human','happy human being','happy individual','happy adult']
    sad = ['sad person','sad woman','sad human','sad human being','sad individual','sad adult']
    angry = ['angry person','angry woman','angry human','angry human being','angry individual','angry adult']

    person_ = person + [f'a {i}' for i in person] + [f'a photo of a {i}' for i in person] + [f'an image of a {i}' for i in person] + [f'a picture of a {i}' for i in person]
    happy_ = happy + [f'a {i}' for i in happy] + [f'a photo of a {i}' for i in happy] + [f'an image of a {i}' for i in happy] + [f'a picture of a {i}' for i in happy]
    sad_ = sad + [f'a {i}' for i in sad] + [f'a photo of a {i}' for i in sad] + [f'an image of a {i}' for i in sad] + [f'a picture of a {i}' for i in sad]
    angry_ = angry + [f'an {i}' for i in angry] + [f'a photo of an {i}' for i in angry] + [f'an image of an {i}' for i in angry] + [f'a picture of an {i}' for i in angry]

    all_words = person_ + happy_ + sad_ + angry_

    #Get CLIP language embeddings
    language_embeddings = []

    for word in all_words:
        input_ = clip.tokenize([word]).to(device)
        with torch.no_grad():
            embedding = model.encode_text(input_).cpu().numpy().squeeze()
        language_embeddings.append(embedding)

    lang_arr = np.array(language_embeddings)
    lang_df = pd.DataFrame(lang_arr,index=all_words)
    lang_df.to_csv(f'sobem/sobem_clip_{write_clip}_emb_lang_df.vec',sep=' ')

    #Get CLIP image embeddings
    SOURCE_DIR = f'sobem/photos'
    df_index = []
    image_embeddings = []

    for i in range(1,11):
        target_dir = f'{SOURCE_DIR}/{i}/'
        imgs = listdir(target_dir)

        for idx,img in enumerate(imgs):
            image = preprocess(Image.open(f'{target_dir}/{img}').convert('RGB')).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = model.encode_image(image).cpu().numpy().squeeze()
                image_embeddings.append(emb)
                df_index.append(img)

    image_array = np.array(image_embeddings)
    image_df = pd.DataFrame(image_array,index=df_index)
    image_df.to_csv(f'sobem/sobem_clip_{write_clip}_emb_df.vec',sep=' ')
