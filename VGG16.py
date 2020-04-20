# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data.dataset import Dataset

from sklearn.model_selection import train_test_split

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import csv
from PIL import Image
import time
import copy
import os

#print(os.listdir("../input"))
use_gpu = torch.cuda.is_available()

# Any results you write to the current directory are saved as output.
#csv_file = '../input/comp-551-w2019-project-3-modified-mnist/train_labels.csv'
#train_pics = '../input/data-thresh-84/train_data_processed_84_thresh.pkl'
#test_pics = '../input/data-thresh-84/test_data_processed_84_thresh.pkl'

csv_file = 'train_labels.csv'
train_pics = 'train_data_processed_84_thresh.pkl'
test_pics = 'test_data_processed_84_thresh.pkl'

train_images = pd.read_pickle(train_pics)
train_labels = pd.read_csv(csv_file)
train_images.shape

#%%

X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.01)


#%%
print(X_val.shape)
print(y_val.shape)

#%%
BATCH_SIZE = 64

class MNIST_Modified_Train(Dataset):
    def __init__(self, image_file, label_file, transform=None):
        """
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data = image_file
        self.labels = label_file.iloc[:, 1].values

        self.transforms = transform

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        single_image_label = torch.as_tensor(single_image_label)
        # Read each 784 pixels and reshape the 1D array ([784]) to 2D array ([28,28]) 
        img_as_np = np.asarray(self.data[index]).astype('uint8')
        # Covert single channel images to 3-channels
        img_as_np = np.stack([img_as_np]*3, axis=-1)

	    # Convert image from numpy array to PIL image, mode 'L' is for grayscale
        img_as_img = Image.fromarray(img_as_np)
        #img_as_img = img_as_img.convert('L')
        # Transform image to tensor
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        else:
            self.transforms = transforms.ToTensor()
            img_as_tensor = self.transforms(img_as_img)
        # Return image and the label
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data)
        
        
transformations = transforms.Compose([transforms.RandomRotation((-30, 30)), 
                                      transforms.RandomAffine(degrees=0, 
                                      translate =(0.1, 0.1), shear = (-30,30), fillcolor = 0), transforms.ToTensor()
                                    #, transforms.Normalize((train_images.mean()/255,), (train_images.std()/255,))
                                      ]) 
                                      
train_1 = MNIST_Modified_Train(X_train, y_train)
train_2 = MNIST_Modified_Train(X_train, y_train, transform = transformations)                               
train_3 = MNIST_Modified_Train(X_train[0:32000], y_train[0:32000], transform = transformations) 
train = torch.utils.data.ConcatDataset([train_1, train_2, train_3])
val = MNIST_Modified_Train(X_val, y_val)
print(len(train))
print(type(train))
train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = True)
val_loader = torch.utils.data.DataLoader(val, batch_size = BATCH_SIZE, shuffle = True)


#%%

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
#    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def show_databatch(inputs, classes):
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[x.item() for x in classes])

# Get a batch of training data
inputs, classes = next(iter(train_loader))
show_databatch(inputs, classes)

#%%
def visualize_model(vgg, num_images=6):
    was_training = vgg.training
    
    # Set model for evaluation
    vgg.train(False)
    vgg.eval() 
    
    images_so_far = 0
    
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels = data
            print(type(data))
            size = inputs.size()[0]
            
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            
            outputs = vgg(inputs)
            
            _, preds = torch.max(outputs.data, 1)
            predicted_labels = [preds[j] for j in range(inputs.size()[0])]
            
            print("Ground truth:")
            show_databatch(inputs.data.cpu(), labels.data.cpu())
            print("Prediction:")
            show_databatch(inputs.data.cpu(), predicted_labels)
            
            del inputs, labels, outputs, preds, predicted_labels
            torch.cuda.empty_cache()
            
            images_so_far += size
            if images_so_far >= num_images:
                break
        
    vgg.train(mode=was_training) # Revert model back to original training state

#%%

def eval_model(vgg, criterion):
    since = time.time()
    avg_loss = 0
    avg_acc = 0
    loss_test = 0
    acc_test = 0
    
    test_batches = len(val_loader)
    print("Evaluating model")
    print('-' * 10)
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if i % 100 == 0:
                print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)
    
            vgg.train(False)
            vgg.eval()
            inputs, labels = data
    
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
    
            outputs = vgg(inputs)
    
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
    
            loss_test += loss.item()
            acc_test += torch.sum(preds == labels.data)
    
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        
    avg_loss = loss_test / len(val)
    avg_acc = acc_test.item() / len(val)
    
    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)

    


#%%
# Load the pretrained model from pytorch

VGG_TYPES = {'vgg11' : torchvision.models.vgg11, 
             'vgg11_bn' : torchvision.models.vgg11_bn, 
             'vgg13' : torchvision.models.vgg13, 
             'vgg13_bn' : torchvision.models.vgg13_bn, 
             'vgg16' : torchvision.models.vgg16, 
             'vgg16_bn' : torchvision.models.vgg16_bn,
             'vgg19_bn' : torchvision.models.vgg19_bn, 
             'vgg19' : torchvision.models.vgg19}

class Custom_VGG(nn.Module):

    def __init__(self,
                 ipt_size=(84, 84), 
                 pretrained=True, 
                 vgg_type='vgg16', 
                 num_classes=10):
        super(Custom_VGG, self).__init__()

        # load convolutional part of vgg
        assert vgg_type in VGG_TYPES, "Unknown vgg_type '{}'".format(vgg_type)
        vgg_loader = VGG_TYPES[vgg_type]
        vgg = vgg_loader(pretrained=pretrained)
        self.features = vgg.features

        # init fully connected part of vgg
        test_ipt = Variable(torch.zeros(1,3,ipt_size[0],ipt_size[1]))
        test_out = vgg.features(test_ipt)
        self.n_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
        self.classifier = nn.Sequential(nn.Linear(self.n_features, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(),
                                        nn.Linear(4096, num_classes)
                                       )
        self._init_classifier_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _init_classifier_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
#____________________________________________________________________________________________________________________________
#vgg16 = models.vgg16_bn()
#vgg16 = models.vgg16()
vgg16 = Custom_VGG(ipt_size=(84, 84), pretrained=True)
#vgg16.load_state_dict(torch.load(vgg16bn))
print(vgg16.classifier[6].out_features) # 1000 


# Freeze training for all layers
for param in vgg16.features.parameters():
    param.require_grad = False
    
# Print out all the layers of the model 
print(vgg16)

#%%
if use_gpu:
    vgg16.cuda() #.cuda() will move everything to the GPU side
    
criterion = nn.CrossEntropyLoss()

#optimizer_ft = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)
optimizer_ft = optim.Adam(vgg16.parameters(), lr=0.001, eps=0.05)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)    


#%%

def train_model(vgg, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(vgg.state_dict())
    best_acc = 0.0
    
    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0
    
    train_batches = len(train_loader)
    val_batches = len(val_loader)
    
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print('-' * 10)
        
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        
        vgg.train(True)
        
        for i, data in enumerate(train_loader):
            if i % 100 == 0:

                print("\rTraining batch {}/{}".format(i, train_batches), end='', flush=True)
                
            inputs, labels = data
            
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            
            optimizer.zero_grad()
            
            outputs = vgg(inputs)
            
            _, preds = torch.max(outputs.data, 1)
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            loss_train += loss.item()
            acc_train += torch.sum(preds == labels.data)
            
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        
        print()
       
        avg_loss = loss_train / len(train)
        avg_acc = acc_train.item() / len(train)
        print('acc_train:', acc_train, 'len:',len(train))
        print('train loss:', avg_loss, 'train_acc:', avg_acc)
        vgg.train(False)
        vgg.eval()
        
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                if i % 100 == 0:
                    print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)
                    
                inputs, labels = data
                
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                
                optimizer.zero_grad()
                
                outputs = vgg(inputs)
                
                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)
                
                loss_val += loss.item()
                acc_val += torch.sum(preds == labels.data)
                
                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()
        
        avg_loss_val = loss_val / len(val)
        avg_acc_val = acc_val.item() / len(val)
        print('acc_val:', acc_val, 'len:',len(val))
        print('val loss:', avg_loss_val, 'val_acc:', avg_acc_val)
        
        print()
        print("Epoch {} result: ".format(epoch+1))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print('-' * 10)
        print()
        
        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(vgg.state_dict())
        
    elapsed_time = time.time() - since
    print()
    print('Elapsed Time: ', elapsed_time)
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))
    
    vgg.load_state_dict(best_model_wts)
    return vgg

#%%
vgg16 = train_model(vgg16, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=1)
torch.save(vgg16.state_dict(), 'VGG16_ModifiedMNIST.pt')   
#%%
eval_model(vgg16, criterion)
#visualize_model(vgg16, num_images=32) 

# Delete training data to clear memory
del train_images
del X_train
del train_1
del train_2
del train_3
del train

#Prepare the test loader
test_images = pd.read_pickle(test_pics)
X_test = test_images
X_test = np.stack([X_test]*3, axis=1)
test = torch.from_numpy(X_test).type(torch.float32)
print(test.shape)
test_loader = torch.utils.data.DataLoader(test, batch_size=100, shuffle=False)

#%%
def predict(vgg):
    print("Predicting...")
    since = time.time()
    #vgg.eval()
    test_batches = len(test_loader)
    pred = 0
    pred_all = 0
    type_pred_all = type(pred_all)
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if i % 100 == 0:
                print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)
                
        #for data in test_loader:
            vgg.eval()
            data = data.cuda()
            output = vgg(data)
            pred = output.data.max(1, keepdim=True)[1]
            #print(pred)
            #print(type(pred))
            if type(pred_all) == type_pred_all:
                pred_all = pred
            else:
                pred_all = torch.cat((pred_all, pred), 0)
            
    elapsed_time = time.time() - since
    print('Prediction Completed')
    return pred_all
    
pred = predict(vgg16)

def prediction_print(pred):
    count = 0
    with open('prediction_VGG_v4.csv', 'w', newline='', encoding='utf-8') as csv_file:
        print("Printing prediction to csv file... ")
        writer = csv.writer(csv_file)
        writer.writerow(['ID', 'Category'])
        for i in pred:
            writer.writerow([count, i.item()])
            count += 1
        
    csv_file.close()
    print('Printing finished!')

# Print predictions to file
#prediction_print(pred)

import matplotlib.cm as cm
def display_wrong(vgg):
    vgg.eval()
    with torch.no_grad():
        for data, target in val_loader:
            data = data.cuda()
            target = target.cuda()            
            output = vgg(data)
            pred = output.data.max(1, keepdim=True)[1]
            data = data.cpu()
            target = target.cpu() 

            for i in zip(target, pred, data):

                if i[0].item() != i[1].item():
                    print('Target: ', i[0].item(), 'Prediction: ', i[1].item())

                    img = i[2].numpy().transpose((1, 2, 0))

                    print(img.shape)

                    plt.axis('off')
                    plt.imshow(img, cmap=cm.Greys_r)
                    plt.show()
           
display_wrong(vgg16)