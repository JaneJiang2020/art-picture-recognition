import os
import numpy as np
from PIL import Image, ImageOps
from numpy import expand_dims
from keras.preprocessing.image import ImageDataGenerator
import random
import tensorflow as tf
from keras.initializers import glorot_uniform, he_normal
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import cohen_kappa_score, f1_score
from keras.optimizers import SGD
from matplotlib import pyplot
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder


x, y = np.load('x_images.npy', allow_pickle=True), np.load('y_labels.npy', allow_pickle=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2, stratify=y, shuffle=True)

x_train, x_test = x_train.reshape(len(x_train), -1), x_test.reshape(len(x_test), -1)
x_train, x_test = x_train/255, x_test/255
y_train, y_test = to_categorical(y_train, num_classes=10), to_categorical(y_test, num_classes=10)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
weight_init = glorot_uniform(seed=SEED)

LR = 0.001
N_NEURONS = (256, 64,64)  #128
N_EPOCHS = 150 #200 #150
BATCH_SIZE = 64 #64
DROPOUT = 0.1 #0.1

model = Sequential([
    Dense(N_NEURONS[0], input_dim=150528, kernel_initializer='uniform'),
    Activation("relu"),
    Dropout(DROPOUT),
    BatchNormalization()])
for n_neurons in N_NEURONS[1:]:
    model.add(Dense(n_neurons, activation="relu", kernel_initializer='uniform'))
    model.add(Dropout(DROPOUT, seed=SEED))
    model.add(BatchNormalization())
model.add(Dense(10, activation="softmax", kernel_initializer=weight_init))
model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test))
# model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test),
#           callbacks=[ModelCheckpoint("mlp_siqijiang_test4.hdf5", monitor="val_loss", save_best_only=True)])

model.summary()

print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")
print("Cohen Kappa", cohen_kappa_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1)))
print("F1 score", f1_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1), average = 'macro'))
