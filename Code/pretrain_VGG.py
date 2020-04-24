
############################################################################################################
#### This code is created by Yunyun Jiang, the purpose of this model is to build a pretrained VGG model ####
############################################################################################################



# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
import seaborn as sb
import cv2
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale
from collections import Counter
from sklearn.datasets import make_classification


import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets,transforms,models
from torch.utils.data.sampler import SubsetRandomSampler



#import nltk
#nltk.download('punkt')

if "images" not in os.listdir(os.getcwd()):
    try:
        os.system("wget -x --load-cookies cookiesKaggle.txt -nH https://www.kaggle.com/ikarus777/best-artworks-of-all-time/download")
        os.system("unzip best-artworks-of-all-time.csv.zip")
    except:
        print("There was a problem with the download!")
        # Download the Stanford Sentiment Treebank from https://gluebenchmark.com/tasks and unzip it in the current working dir
    if "best-artworks-of-all-time" not in os.listdir(os.getcwd()):
        print("There was a problem with the download!")
        import sys
        sys.exit()

DATA_DIR = os.getcwd ()+"/final_dataset/"
print(DATA_DIR)
RESIZE_TO = 50


############ define a function to transform the image data and
# split data into training and test #######

def load_split_train_test(DATA_DIR,test_train_split=0.9, val_train_split=0.2):
    train_transforms=transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize ([0.5,0.5,0.5],
                                                                [0.5,0.5,0.5])
                                          ])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize ([0.5,0.5,0.5],
                                                                  [0.5,0.5,0.5])
                                          ])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize ([0.5,0.5,0.5],
                                                                  [0.5,0.5,0.5])
                                          ])
    train_data = datasets.ImageFolder(DATA_DIR,
                    transform=train_transforms)
    valid_data = datasets.ImageFolder(DATA_DIR,
                    transform=valid_transforms)
    test_data = datasets.ImageFolder(DATA_DIR,
                    transform=test_transforms)


    dataset_size = len(train_data)
    indices = list ( range (dataset_size) )
    np.random.shuffle(indices)

    test_split = int(np.floor(test_train_split*dataset_size))
    train_indices_1, test_indices = indices [:test_split], indices [test_split:]
    train_size = len ( train_indices_1)
    validation_split=int(np.floor((1-val_train_split)*train_size))
    train_indices, val_indices = indices [ :validation_split], indices[validation_split:test_split]
    train_sampler = SubsetRandomSampler ( train_indices )
    test_sampler = SubsetRandomSampler ( test_indices )
    val_sampler = SubsetRandomSampler ( val_indices )
    trainloader=torch.utils.data.DataLoader(train_data,batch_size=64,sampler=train_sampler)
    valloader = torch.utils.data.DataLoader ( valid_data, batch_size = 32, sampler = test_sampler )
    testloader=torch.utils.data.DataLoader(test_data,sampler=val_sampler)
    return trainloader,valloader, testloader

trainloader, valloader, testloader = load_split_train_test(DATA_DIR, .8, .1)
print(trainloader.dataset.classes)
print(len(trainloader))
print(len(testloader))
print(len(valloader))

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 0.0001
N_EPOCHS = 30
DROPOUT = 0.5

# %% ----------------------------------- Helper Functions --------------------------------------------------------------
def acc(x, y, return_labels=False):
    """ Simple function to get the accuracy or the predicted labels. The label with the highest logit is chosen """
    with torch.no_grad():  # Explained on the training loop
        logits = model(x)  # (n_examples, n_labels) --> Need to operate on columns, so axis=1
        pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
    if return_labels:
        return pred_labels
    else:
        return 100*accuracy_score(y.cpu().numpy(), pred_labels)


# Build and train your network ##
# Transfer Learning
model = models.vgg16(pretrained=True)
model

# %% -------------------------------------- Backprogation ------------------------------------------------------------------
#  VGG16  network #:
for parameter in model.parameters():
    parameter.requires_grad = False
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 5000)),
                                        ('relu', nn.ReLU()),
                                        ('drop', nn.Dropout(p=0.5)),
                                        ('fc2', nn.Linear(5000, 102))]))
model.classifier = classifier


# %% -------------------------------------- Training Prep ----------------------------------------------------------
model.to(device)
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()


# Function for the validation pass
def validation(model, valloader, criterion):
    val_loss = 0
    accuracy = 0

    for images, labels in iter (valloader):
        images, labels = images.to (device), labels.to (device)

        output = model.forward ( images )
        val_loss += criterion ( output, labels ).item ()
        m = nn.Softmax (dim=1)
        probabilities = m(output)
        equality = (labels.data == probabilities.max ( dim = 1 ) [ 1 ])
        accuracy += equality.type ( torch.FloatTensor ).mean ()

    return val_loss, accuracy


# %% --------------------------------- define a train classifier -------------------------------------------------------

epochs = 10
steps = 0
print_every = 10
running_loss=0
train_losses,val_losses,val_accuracy=[],[],[]

for e in range ( epochs ):
    model.train()
    running_loss=0
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()  ####### every epoch needs to back to 0 ####
        output = model.forward(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if steps % print_every == 0:
            model.eval ()
            #val_loss=0
            accuracy=0
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad ():
                validation_loss, accuracy = validation (model,valloader,criterion )
            train_losses.append(running_loss/len(trainloader))
            val_losses.append ( validation_loss / len (valloader ) )
            val_accuracy.append(accuracy/len (valloader) )
            print ( "Epoch: {}/{}.. ".format ( e + 1, epochs ),
                    "Training Loss: {:.3f}.. ".format ( running_loss / print_every),
                    "Validation Loss: {:.3f}.. ".format (validation_loss/len (valloader)),
                    "Validation Accuracy: {:.3f}".format ( accuracy / len (valloader) ) )
            running_loss = 0
            model.train ()



#------------------ Train classifier -------------------------#
#train_classifier()
torch.save(model.state_dict(),'/home/ubuntu/Deep-Learning/Pytorch_/Final Project/models/VGG16_train.pt')



print(train_losses)
print(val_losses)
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.title('VGG-16: training/validation Loss')
plt.legend(frameon=False)
plt.savefig('/home/ubuntu/Deep-Learning/Pytorch_/Final Project/models/plots/VGG16_loss.png')
plt.show()

plt.plot(val_accuracy, label='Validation Accuracy')
#plt.plot(val_losses, label='Validation loss')
plt.title('VGG-16: Validation accuracy')
plt.legend(frameon=False)
plt.savefig('/home/ubuntu/Deep-Learning/Pytorch_/Final Project/models/plots/VGG16_valid_accuracy.png')
plt.show()


####  test trained work on test data ##########

def test_accuracy(model, testloader):
    # Do validation on the test set
    model.eval ()
    model.to (device)

    with torch.no_grad ():
        accuracy = 0
        target_labels,pred_labels=[],[]
        for images, labels in iter ( testloader ):
            images, labels = images.to (device), labels.to (device)
            output = model.forward(images)
            m = nn.Softmax ( dim = 1 )
            probabilities = m ( output )
            target_labels.append(labels.data)
            pred_labels.append(probabilities.max (dim = 1)[1])
            equality = (labels.data == probabilities.max ( dim = 1 ) [ 1 ])
            accuracy += equality.type ( torch.FloatTensor ).mean ()
        print ( "Test Accuracy: {}".format ( accuracy / len ( testloader )))
    return target_labels, pred_labels

target_labels, pred_labels= test_accuracy (model, testloader)

print(target_labels)
print(pred_labels)
print(len(testloader))

stacked=torch.stack(
    (
        target_labels
        ,pred_labels
    )
    ,dim=1
)

stacked.shape
print(stacked)
#stacked[0].tolist()
#j,k=stacked[0].tolist()

cmt=torch.zeros(10,10,dtype=torch.int64)
cmt

for p in stacked:
    tl,pl=p.tolist()
    cmt[tl,pl]=cmt[tl,pl]+1


import matplotlib.pyplot as plt
from sklearn.metrics import *
import itertools
import seaborn as sns
#from resources.plotcm import plot_confusion_matrix

cm=confusion_matrix(target_labels,pred_labels)
print(type(cm))




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    #plt.set ( xlim = (0, 12))
    #plt.xlim ( [ 0.0, 12.0 ] )
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

cnf_matrix = confusion_matrix(target_labels,pred_labels)
np.set_printoptions(precision=2)
class_names=['Albrecht_Du╠Иrer', 'Alfred_Sisley', 'Edgar_Degas', 'Francisco_Goya', 'Pablo_Picasso', 'Paul_Gauguin','Pierre_Auguste_Renoir','Rembrandt', 'Titian', 'Vincent_van_Gogh']
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='VGG-16: Confusion matrix')
plt.savefig('/home/ubuntu/Deep-Learning/Pytorch_/Final Project/models/plots/VGG16_confusion_matrix.png')
plt.show()