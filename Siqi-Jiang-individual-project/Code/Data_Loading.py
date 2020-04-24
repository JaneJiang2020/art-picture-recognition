# Import librariest
from numpy.random import seed
seed(1)
import cv2
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as pl


artists = pd.read_csv('/home/ubuntu/Final_project/best-artworks-of-all-time/artists.csv')
print(artists.shape)

# Sort artists by number of paintings
artists = artists.sort_values(by=['paintings'], ascending=False)

# Check on artists having more than 250 paintings
artists_top = artists[artists['paintings'] >= 250].reset_index()
artists_top = artists_top[['name', 'paintings']]

#Create class_weight and check on the imbalance
artists_top['class_weight'] = artists_top["paintings"].sum() / (artists_top.shape[0] * artists_top.paintings)
print(artists_top)
class_weights = artists_top['class_weight'].to_dict()
# print(class_weights)

# Explore images of top artists
images_dir = '/home/ubuntu/Final_project/best-artworks-of-all-time/images/images'
artists_dirs = os.listdir(images_dir)
artists_top_name = artists_top['name'].str.replace(' ', '_').values
print(artists_top_name)

#Change the folder name of Albrecht_Durer cause for loop can't regornize
artists_top_name[4]= "Albrecht_Durer"
print(artists_top_name)

#load the imgs and label at the same time, resize the images
resize =(224,224)
imgs=[]
label=[]
for name in artists_top_name:
    for path in [f for f in os.listdir('/home/ubuntu/Final_project/best-artworks-of-all-time/images/images/'+ name)]:
        img = cv2.resize(cv2.imread('/home/ubuntu/Final_project/best-artworks-of-all-time/images/images/'+ name +"/" + path),resize)
        imgs.append(img)
        label.append(name)


x = np.array(imgs)
y = np.array(label)

print(x.shape)
print(y.shape)
print(y)

#check the size of each class
values, counts = np.unique(y, return_counts=True)
values = list(values)
counts = list(counts)
print(values)
print(counts)

#enlabel the target variable
le = LabelEncoder()
le.fit(['Vincent_van_Gogh','Edgar_Degas','Pablo_Picasso','Pierre-Auguste_Renoir',
 'Albrecht_Durer','Paul_Gauguin','Francisco_Goya','Rembrandt',
 'Alfred_Sisley','Titian'])
y = le.transform(y)
print(y)

values, counts = np.unique(y, return_counts=True)
values = list(values)
counts = list(counts)
print(values)
print(counts)

#save the images and labels as numpy files
np.save("x_images.npy", x)
np.save("y_labels.npy", y)




