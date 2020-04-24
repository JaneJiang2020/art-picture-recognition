import os
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from numpy import expand_dims
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.callbacks import *

# Sort artists by number of paintings
artists = pd.read_csv('/home/ubuntu/Final_project/best-artworks-of-all-time/artists.csv')
print(artists.shape)
artists = artists.sort_values(by=['paintings'], ascending=False)

# check on the artists having more than 250 paintings(take them for our model)
artists_top_ten = artists[artists['paintings'] >= 250].reset_index()
artists_top_ten = artists_top_ten[['name', 'paintings']]

#get the weight of each class
artists_top_ten['class_weight'] = artists_top_ten.paintings.sum() / (artists_top_ten.shape[0] * artists_top_ten.paintings)
artists_top_ten
artists_class_weights = artists_top_ten['class_weight'].to_dict()
artists_class_weights

# Explore images of top artists
images_dir = '/home/ubuntu/Final_project/best-artworks-of-all-time/images/images'
artists_dirs = os.listdir(images_dir)
artists_top_name = artists_top_ten['name'].str.replace(' ', '_').values

# Perform Image Augmentation
batch_size = 16
image_size = (224, 224)
model_input_shape=(224, 224,3)
n_classes = artists_top_ten.shape[0]

#Generate batches of tensor image data with real-time data augmentation
datagen = ImageDataGenerator(rescale=1./255.,
                             validation_split=0.2,
                             # rotation_range=50,
                             # width_shift_range=0.2,
                             # height_shift_range=0.2,
                             shear_range=5,
                             horizontal_flip=True,
                             vertical_flip=True)

#Setup the data generators
#Takes the path to a directory & generates batches of augmented data
train_generator = datagen.flow_from_directory(directory=images_dir,          #directory must be set to the path where your ‘n’ classes of folders are present.
                                                    class_mode='categorical', #Set “binary” if you have only two classes to predict, if not set to“categorical”
                                                    target_size=image_size,  #the size of your input images, every image will be resized to this size.
                                                    batch_size=batch_size,   #No. of images to be yielded from the generator per batch
                                                    subset="training",       #Set True if you want to shuffle the order of the image that is being yielded
                                                    shuffle=True,
                                                    classes=artists_top_name.tolist(),
                                                    color_mode="rgb"
                                                   )

valid_generator = datagen.flow_from_directory(directory=images_dir,
                                                    class_mode='categorical',
                                                    target_size=image_size,
                                                    batch_size=batch_size,
                                                    subset="validation",
                                                    shuffle=True,
                                                    classes=artists_top_name.tolist(),
                                                    color_mode="rgb"
                                                   )
#Fitting/Training VGG19 model, with weights pre-trained on ImageNet.
STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
# print("Total number of batches =", STEP_SIZE_TRAIN, "and", STEP_SIZE_VALID)

# Load pre-trained model
pre_trained_model = VGG19(weights='imagenet', include_top=False, input_shape=model_input_shape)  # include_top: whether to include the 3 fully-connected layers at the top of the network.
                                                                                                 # weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).
#train with all layer
for layer in pre_trained_model.layers:
    layer.trainable = False

#In Keras, each layer has a parameter called “trainable”. For freezing the weights of a particular layer,
# we should set this parameter to False, indicating that this layer should not be trained.
# That’s it! We go over each layer and select which layers we want to train.

# Check the trainable status of the individual layers
for layer in pre_trained_model.layers:
    print(layer, layer.trainable)

# Add layers at the end
VGG_model = pre_trained_model.output
VGG_model = Flatten()(VGG_model)

N_NEURONS=(512,16)
for n_neurons in N_NEURONS[1:]:
    VGG_model = Dense(n_neurons, kernel_initializer='uniform')(VGG_model)
    VGG_model = BatchNormalization()(VGG_model)
    VGG_model = Activation('relu')(VGG_model)

output = Dense(n_classes, activation='softmax')(VGG_model)

model = Model(inputs=pre_trained_model.input, outputs=output)

optimizer = Adam(lr=0.0001)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# Show a summary of the model. Check the number of trainable parameters
model.summary()

n_epoch = 10

early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1,
                           mode='auto', restore_best_weights=True)

# Reduce learning rate when a metric has stopped improving.
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,
                              verbose=1, mode='auto')
# monitor: quantity to be monitored. factor: factor by which the learning rate will be reduced. new_lr = lr * factor. patience: number of epochs with no improvement after which learning rate will be reduced.
# verbose: int. 0: quiet, 1: update messages.in auto mode, the direction is automatically inferred from the name of the monitored quantity.

# Train the model - all layers
VGG_fit = model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=valid_generator, validation_steps=STEP_SIZE_VALID,
                              epochs=n_epoch,                                                 # Number of epochs to train the model. An epoch is an iteration over the entire data provided. 10
                              shuffle=True,
                              verbose=1,                 # 1, will show animated progress bar
                              callbacks=[reduce_lr],     # define and use a callback when to automate some tasks after every training/epoch that have controls over the training process.
                              use_multiprocessing=True,
                              workers=16,
                              class_weight=artists_class_weights  # This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
                             )


score = model.evaluate_generator(train_generator, verbose=1)
print("Prediction accuracy on train data =", 100*score[1], "%")

score = model.evaluate_generator(valid_generator, verbose=1)
print("Prediction accuracy on CV data =", 100*score[1], "%")

# print("Final accuracy on train set:", 100*model.evaluate_generator(valid_generator, verbose=1)[0], "%")
# print("Final accuracy on validations set:", 100*model.evaluate_generator(valid_generator, verbose=1)[1], "%")

#Check Performance
acc = VGG_fit.history['accuracy']
val_acc = VGG_fit.history['val_accuracy']
loss = VGG_fit.history['loss']
val_loss = VGG_fit.history['val_loss']
epochs = range(len(acc))

fig, axes = pyplot.subplots(1, 2, figsize=(15,5))
axes[0].plot(epochs, acc, 'b', label='Training acc')
axes[0].plot(epochs, val_acc, 'g', label='Validation acc')
axes[0].set_title('VGG19: Training and validation accuracy')
axes[0].legend()


axes[1].plot(epochs, loss, 'b', label='Training loss')
axes[1].plot(epochs, val_loss, 'g', label='Validation loss')
axes[1].set_title('VGG19: Training and validation loss')
axes[1].legend()

pyplot.savefig('performance_vgg.png')


#model.save('vgg_model.h5')


#Reference:https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/


