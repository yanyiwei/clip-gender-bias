import pandas as pd
from os import listdir
from collections import Counter
from nsfw_detector import predict

model = predict.load_model('D:\\nsfw_mobilenet2.224x224.h5')

#Source directory of generated images for females
#SOURCE = f'D:\\vqgan_gender_bias_females'
#Source directory of generated images for males
SOURCE = f'D:\\vqgan_gender_bias_males'

#Catalogue target images
targets = listdir(SOURCE)
imgs = [f'{SOURCE}{target}' for target in targets]

#Obtain probabilities that target images are pornographic or sexualized
prediction = predict.classify(model,SOURCE)

#Create a dataframe of probabilities and write to file
prediction_df = pd.DataFrame.from_dict(prediction,orient='index')
#prediction_df.to_csv(f'D:\\vqgan_gender_bias\\cnn_prediction_df_female.csv')
prediction_df.to_csv(f'D:\\vqgan_gender_bias\\cnn_prediction_df_male.csv')

#Update index to remove directory and just keep the file name
current_index = prediction_df.index.tolist()
#new_index = [i[75:] for i in current_index]
new_index = [i[63:] for i in current_index]
prediction_df.index = new_index

#Get the age associated with each generated image based on the filename
age = [i[:2] for i in new_index]
prediction_df['age'] = age

age_counts = {'drawings':[],'hentai':[],'neutral':[],'porn':[],'sexy':[]}

#Select photos corresponding to a certain age
for age_ in range(12,18):
    sub_df = prediction_df.loc[prediction_df.age.isin([str(age_)])]

    #Drop the age column, take the argmax of the sexualized image categories, and count for the age range
    max_df = sub_df.drop(['age'],axis=1).idxmax(axis='columns')
    counts = Counter(max_df.tolist())

    #Add the count for the current classification category as corresponds to the current age
    for category in ['drawings','hentai','neutral','porn','sexy']:
        age_counts[category].append(counts[category])

#Print the dictionary of counts, which includes five categories with six ages from 12-17, each in order
print(age_counts)