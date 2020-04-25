# %%  Import Pakage----------------------------------------------
import os
import cv2
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder

# Import CSV file ----------------------------------------------
df1 = pd.read_csv('best-artworks-of-all-time/artists.csv')
print(df1.shape)

# Check Missing value
df1_missing = df1.isna()
df1_num_missing = df1_missing.sum()
print(df1_num_missing)

# Drop uninterested columns
df1.drop(["bio","wikipedia", "id"], axis=1, inplace=True)

# Display images from their owner
top5 = df1.sort_values(by=["paintings"], ascending=False).head(5)

# plot painting distribution
plt.figure(figsize=(15,10))
sns.set(style="whitegrid")
ax = sns.distplot(df1['paintings'], color='yellow')
plt.title('Number of Paintings Distribution')
plt.xlabel('# of Painting')
plt.ylabel('Rate')
plt.show()

# %%  EDA ----------------------------------------------------------------------
# ******************************************************************************
# Calculating Age & Age analysis
df1_year = pd.DataFrame(df1.years.str.split(' ',2).tolist(),columns = ['birth','-','death'])
df1_year.drop(["-"], axis=1,inplace=True)
df1["birth"] = df1_year.birth
df1["death"] = df1_year.death
df1["age"] = df1["death"].apply(lambda x: int(x)) - df1["birth"].apply(lambda x: int(x))
print("Age Distribution: \n", df1.age.describe())

# plot painting distribution
plt.figure(figsize=(20,5))
sns.set(style="whitegrid")
sns.barplot(x=df1['age'].value_counts().index, y=df1['age'].value_counts().values, color='green')
plt.title('Age Distribution 1')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(15,10))
sns.set(style="whitegrid")
sns.distplot(df1['age'], color='green')
plt.title('Age Distribution 2')
plt.xlabel('Age')
plt.ylabel('Rate')
plt.show()

# Nationality Analysis
df2 = pd.DataFrame(df1['nationality'].str.split(',',2).tolist(), columns = ['n1','n2','n3'])
# print(df2)
df1 = pd.concat([df1, df2], axis=1)
print("All Nations", pd.unique(df1[['n1', 'n2', 'n3']].values.ravel('K')))

# At first, I planed to break one record with multiple nationalities into 3 ones to calculate frequence, but later I realized multi-nationalities should be count as one special situation
frame1 = [df2.n1, df2.n2, df2.n3]
df_n = pd.concat(frame1).dropna()
print(df_n)

# plot Nationality Frequency
sns.set(style="whitegrid")
plt.figure(figsize=(20,5))
sns.barplot(x=df1['nationality'].value_counts().index, y=df1['nationality'].value_counts().values, color="salmon")
plt.title('Nationality')
plt.xticks(rotation=60)
plt.xlabel('Nation')
plt.ylabel('Frequency')
plt.show()

# Genre Analysis
df3 = pd.DataFrame(df1['genre'].str.split(',',2).tolist(), columns = ['g1','g2','g3'])
# print(df3)
df1 = pd.concat([df1, df3], axis=1)
print("All Genres", pd.unique(df1[['g1', 'g2', 'g3']].values.ravel('K')))

# Same as nationality, no need to break multi-genre into several records.
frame2 = [df3.g1, df3.g2, df3.g3]
df_g = pd.concat(frame2).dropna()
# print(df_g)

plt.figure(figsize=(20,5))
sns.barplot(x=df1['genre'].value_counts().index, y=df1['genre'].value_counts().values)
plt.title('Genre Count')
plt.xticks(rotation=60)
plt.xlabel('Genre')
plt.ylabel('Frequency')
plt.show()

# Cross Analysis
bins=[18,40,65,80,98]
group = ["Teenage","Young Adult","Adult","elderly"]
df1['age_structure'] = pd.cut(df1['age'], bins, labels = group)

# of painting in each age group by Genre
plt.figure(figsize=(20,10))
sns.barplot(x="genre", y="paintings", hue="age_structure", data=df1)
plt.title('Number of painting in each age group by Genre')
plt.xticks(rotation=60)
plt.xlabel('Genre')
plt.ylabel('Number of Paintings')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

print("Genre Distribution: \n", df1.genre.describe())

# of painting in each genre group by age group
plt.figure(figsize=(20,10))
sns.barplot(x="age_structure", y="paintings", hue="genre", data=df1)
plt.title('Number of painting in each genre group by age')
plt.xticks(rotation=60)
plt.xlabel('Age Group')
plt.ylabel('Number of Paintings')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

# Age Distribution
plt.figure(figsize=(20,10))
sns.set(color_codes=True)
sns.set(style="whitegrid")
sns.lmplot(x='age', y='paintings', markers=["x"], data=df1, palette="Set2")
plt.title('Age & Paintings')
plt.xlabel('Age')
plt.ylabel('# of Paintings')
plt.show()

# Painting distribution
plt.figure(figsize=(16,10))
sns.violinplot(df1['paintings'], color = 'purple')
plt.xlabel('paintings')
plt.ylabel('Frequency')
plt.title('Paintings distribution')
plt.show()

print("Painting Distribution: \n",df1.paintings.describe())

# Number of Painting by age group
plt.figure(figsize=(20,10))
sns.set(style="whitegrid")
sns.violinplot(x=df1['age_structure'],y=df1['paintings'])
plt.title('Number of Painting by age group')
plt.xlabel('Age Group')
plt.ylabel('Number of Painting')
plt.show()

# Number of Painting in each Nation by Genre
plt.figure(figsize=(20,10))
sns.violinplot(df1['genre'][:15],df1['paintings'][:15],hue=df1['nationality'][:15],dodge=False)
plt.xticks(rotation=60)
plt.title('Number of Painting in each Nation by Genre')
plt.xlabel('Nation')
plt.ylabel('Number of Painting')
plt.legend(bbox_to_anchor=(1.05, 2), loc='upper left', borderaxespad=0.)
plt.show()

# Number of Painting in each age group
plt.figure(figsize=(20,10))
sns.boxenplot(x="age_structure", y="paintings", color="g", scale="linear", data=df1)
plt.title('Number of Painting in each age group')
plt.xlabel('Age Group')
plt.ylabel('Number of Painting')
plt.show()

# Number of Painting in each genre
plt.figure(figsize=(20,10))
sns.boxenplot(x="genre", y="paintings", color="m", scale="linear", data=df1)
plt.title('Number of Painting in each genre')
plt.xlabel('Age Group')
plt.ylabel('Number of Painting')
plt.xticks(rotation=60)
plt.show()

# Number of Painting in each genre
plt.figure(figsize=(20,10))
sns.boxenplot(x="nationality", y="paintings", color="y", scale="linear", data=df1)
plt.title('Number of Painting in each nation')
plt.xlabel('Age Group')
plt.ylabel('Number of Painting')
plt.xticks(rotation=60)
plt.show()

# Number of Painting by Genre
plt.figure(figsize=(20,10))
sns.swarmplot(x=df1['genre'],y=df1['paintings'], color = "black")
plt.xticks(rotation=60)
plt.title('Number of Painting by Genre')
plt.xlabel('Genre')
plt.ylabel('Number of Painting')
plt.show()

# Number of Painting by Nation
plt.figure(figsize=(20,10))
sns.swarmplot(x=df1['nationality'], y=df1['paintings'], color = "red")
plt.xticks(rotation=60)
plt.title('Number of Painting by Nation')
plt.xlabel('Nation')
plt.ylabel('Number of Painting')
plt.show()

print("Nation Distribution:  \n", df1.nationality.describe())

# Summary
print("Summary")
print("Total Artists  : ", df1["name"].nunique())
print("Total Nations  : ", df_n.nunique())
print("Total Genre    : ", df_g.nunique())
print("Total Nations (original)  : ", df1["nationality"].nunique())
print("Total Genre (original)    : ", df1["genre"].nunique())
print("Genre by nation  :", df1.groupby(["nationality"])["genre"].nunique().reset_index())
print("Nation by genre  :", df1.groupby(["genre"])["nationality"].nunique().reset_index())

# Create new dataset
# Pick up top 10 to create a new data frame and use weight to salve unbalance data set
top10 = df1.sort_values(by=["paintings"], ascending=False).head(10).reset_index()
top10.iloc[4, 1] = "Albrecht_Dürer".replace("_", " ")
top10.iloc[3, 1] = "Pierre-Auguste Renoir".replace("_", " ")
top10['weight'] = top10.paintings.sum() / (top10.shape[0] * top10.paintings)
top10_name = top10['name'].str.replace(' ', '_').values
print("Top 10 Artists : ", top10)
print("Top 10 Artists Name: ", top10_name)

pic_dir = 'best-artworks-of-all-time/images/images'

# # Print few random paintings
fig, axes = plt.subplots(1, 6, figsize=(20,5))

for j in range(5):
    j=0
    for i in range(6):
        artist = random.choice(top10_name)
        image = random.choice(os.listdir(os.path.join(pic_dir, artist)))
        image_file = os.path.join(pic_dir, artist, image)
        image = plt.imread(image_file)
        axes[i].imshow(image)
        axes[i].set_title(artist.replace('_', ' '))
        axes[i].axis('off')
    j += 1
    plt.show()


# %%  Load Data ----------------------------------------------------------------
# ******************************************************************************
top10_name=['Vincent_van_Gogh', 'Edgar_Degas' ,'Pablo_Picasso', 'Pierre-Auguste_Renoir','Albrecht_Dürer',
            'Paul_Gauguin', 'Francisco_Goya', 'Rembrandt', 'Alfred_Sisley', 'Titian']

resize = (224,224)
imgs = []
label = []
for name in top10_name:
    for path in [f for f in os.listdir("/home/ubuntu/Final Project/best-artworks-of-all-time/images/images/" + name)]:
        img = cv2.resize(cv2.imread("/home/ubuntu/Final Project/best-artworks-of-all-time/images/images/" + name + "/" + path), resize)
        imgs.append(img)
        label.append(name)
x = np.array(imgs)
y = np.array(label)
print("First checking:", x.shape, y.shape)

le = LabelEncoder()
le.fit(['Vincent_van_Gogh', 'Edgar_Degas' ,'Pablo_Picasso', 'Pierre-Auguste_Renoir','Albrecht_Dürer', 'Paul_Gauguin', 'Francisco_Goya', 'Rembrandt', 'Alfred_Sisley', 'Titian'])
y = le.transform(y)
print(x.shape, y.shape)

# # %%  Model ----------------------------------------------------------------------
# # ********************************************************************************
import os
import random
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import cohen_kappa_score, f1_score


# %%  Set-Up ------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %%  Hyper Parameters --------------------
LR = 1e-3
N_NEURONS = 400
N_EPOCHS = 100
BATCH_SIZE = 512
DROPOUT = 0.2

# %%  Data Prep ---------------------------
print("Double checking shapes", x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=SEED, test_size=0.2, stratify=y)
x_train, x_test = x_train.reshape(len(x_train), -1), x_test.reshape(len(x_test), -1)
x_train, x_test = x_train/255, x_test/255
y_train, y_test = to_categorical(y_train, num_classes=10), to_categorical(y_test, num_classes=10)
print("Double checking shapes", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# %%  Model  ----------------
model = Sequential([
    Dense(N_NEURONS, input_dim = 7500, activation="relu"),
    Dense(10, activation="softmax")
])
model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test),
          callbacks=[ModelCheckpoint("finalproject.hdf5", monitor="val_loss", save_best_only=True)])

print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")
print("Cohen Kappa", cohen_kappa_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1)))
print("F1 score", f1_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test, axis=1), average = 'macro'))


# %%  Prediction ----------------------------------------------------------
# *************************************************************************
from keras.models import load_model
import cv2

def predict(paths):
    x = []
    for path in paths:
        x.append(cv2.resize(cv2.imread(path), resize))

    x = np.array(x)
    x = x.reshape(len(x), -1)
    x = x / 255

    model = load_model('finalproject.hdf5')
    y_pred = np.argmax(model.predict(x), axis=1)
    return y_pred, model