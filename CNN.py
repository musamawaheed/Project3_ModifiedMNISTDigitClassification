import csv
import numpy as np
import pandas as pd
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# DIRECTORIES
csv_file = '../input/group-46-data/train_labels.csv'
train_pics = '../input/group-46-data3/train_data_processed3b_thresh.pkl'
test_pics = '../input/group-46-data3/test_data_processed3b_thresh.pkl'


BATCH_SIZE = 64

train_images = pd.read_pickle(train_pics)
test_images = pd.read_pickle(test_pics)

label_file = pd.read_csv(csv_file)
train_labels = label_file['Category'].values

X_training, X_validation, y_training, y_validation = train_test_split(train_images, train_labels, test_size=0.2)

X_train = torch.from_numpy(X_training).type(torch.LongTensor)
y_train = torch.from_numpy(y_training).type(torch.LongTensor)

X_vali = torch.from_numpy(X_validation).type(torch.LongTensor)
y_vali = torch.from_numpy(y_validation).type(torch.LongTensor)

X_test = torch.from_numpy(test_images).type(torch.LongTensor)

# COUNT LABEL CLASSES
# print(y_train.bincount())

X_train = X_train.view(-1, 1, 28, 28).float()
X_vali = X_vali.view(-1, 1, 28, 28).float()
X_test = X_test.view(-1, 1, 28, 28).float()

train_1 = torch.utils.data.TensorDataset(X_train, y_train)
vali = torch.utils.data.TensorDataset(X_vali, y_vali)


class MNIST_Modified_Train(Dataset):
    def __init__(self, X_training, y_training, transform=None):
        self.data = X_training
        # self.labels = label_file.iloc[:, 1]
        self.labels = y_training
        # self.height = height
        # self.width = width
        self.transforms = transform

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        single_image_label = torch.as_tensor(single_image_label)
#        print(single_image_label)
#        print(type(single_image_label))
        # Read each 784 pixels and reshape the 1D array ([784]) to 2D array ([28,28])
        img_as_np = np.asarray(self.data[index]).astype('uint8')
        # img_as_np = np.stack([img_as_np]*3, axis=-1)
#        print(img_as_np.shape)
#        print(type(img_as_np))
        # Convert image from numpy array to PIL image, mode 'L' is for grayscale
        img_as_img = Image.fromarray(img_as_np)
        # img_as_img = img_as_img.convert('L')
        # Transform image to tensor
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        # Return image and the label
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data)


transformations = transforms.Compose([transforms.RandomAffine(degrees=(-30, 30), fillcolor=0), transforms.ToTensor(), transforms.Normalize((train_images.mean() / 255,), (train_images.std() / 255,))])
#transformations = transforms.Compose([transforms.RandomAffine(degrees=(-20, 20), shear=(-30, 30), fillcolor=0), transforms.ToTensor()])

train_2 = MNIST_Modified_Train(X_training, y_training, transform=transformations)

train = torch.utils.data.ConcatDataset([train_1, train_2])

train_loader = DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True)
vali_loader = DataLoader(dataset=vali, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=X_test, batch_size=10000, shuffle=False)


n_epochs = 50
learning_rate = 0.001
momentum = 0.3
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# class Net(nn.Module):
#    def __init__(self):
#        super(Net, self).__init__()
#        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
#        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, stride=1)
#        self.conv3 = nn.Conv2d(128, 512, kernel_size=3, stride=1)
#        self.conv4 = nn.Conv2d(512, 1024, kernel_size=3, stride=1)
#        self.fc1 = nn.Linear(4 * 4 * 1024, 2000)
#        self.fc2 = nn.Linear(2000, 500)
#        self.fc3 = nn.Linear(500, 10)
#    def forward(self, x):
#        x = F.relu(self.conv1(x))
#        #x = F.max_pool2d(x, 2, 2)
#        x = F.relu(self.conv2(x))
#        x = F.max_pool2d(x, 2, 2)
#        x = F.relu(self.conv3(x))
#        x = F.relu(self.conv4(x))
#        x = F.max_pool2d(x, 2, 2)
#        #print(x.shape)
#        x = x.view(-1, 4 * 4 * 1024)
#        #print(x.shape)
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = self.fc3(x)
#        return F.log_softmax(x, dim=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=2, stride=1)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1)
        self.conv5_bn = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=2, stride=1)
        self.conv6_bn = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(512, 1024, kernel_size=3, stride=1)
        self.conv7_bn = nn.BatchNorm2d(1024)
        self.conv8 = nn.Conv2d(1024, 1024, kernel_size=2, stride=1)
        self.conv8_bn = nn.BatchNorm2d(1024)
        self.fc1 = nn.Linear(4 * 4 * 1024, 4000)
        self.fc2 = nn.Linear(4000, 500)
        self.fc3 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.max_pool2d(x, 2, 2)
        #x = F.relu(self.conv3(x))
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)))
        #x = F.relu(self.conv4(x))
        x = F.dropout(x, p=0.3, training=self.training)
        #x = F.max_pool2d(x, 2, 2)
        x = F.leaky_relu(self.conv5_bn(self.conv5(x)))
        #x = F.relu(self.conv5(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.leaky_relu(self.conv6_bn(self.conv6(x)))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.leaky_relu(self.conv7_bn(self.conv7(x)))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.leaky_relu(self.conv8_bn(self.conv8(x)))
        x = F.dropout(x, p=0.3, training=self.training)
        # print(x.shape)
        x = x.view(-1, 4 * 4 * 1024)
        # print(x.shape)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


network = Net()
network.cuda()
#optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum, weight_decay=0.001)
optimizer = optim.Adam(network.parameters(), lr=learning_rate, eps=0.05)

train_losses = []
train_losses2 = []
train_counter = []
vali_losses = []
vali_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = network(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        train_losses2.append(loss.item())
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            #torch.save(network.state_dict(), '/results/model.pth')
            #torch.save(optimizer.state_dict(), '/results/optimizer.pth')
    print('\nTraining set: Avg. loss:', sum(train_losses2) / len(train_losses2))


def validation():
    network.eval()
    vali_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in vali_loader:
            data = data.cuda()
            target = target.cuda()
            output = network(data)
            vali_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    vali_loss /= len(vali_loader.dataset)
    vali_losses.append(vali_loss)
    print('\nValidation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(vali_loss, correct, len(vali_loader.dataset), 100. * correct / len(vali_loader.dataset)))


def display_wrong():
    network.eval()
    with torch.no_grad():
        for data, target in vali_loader:
            data = data.cuda()
            target = target.cuda()
            output = network(data)
            pred = output.data.max(1, keepdim=True)[1]
            data = data.cpu()
            target = target.cpu()
            for i in zip(target, pred, data):
                if i[0].item() != i[1].item():
                    print('Target: ', i[0].item(), 'Prediction: ', i[1].item())
                    # print(i[2])
                    img = i[2].numpy()
                    img = np.squeeze(img)
                    # print(img.shape)
                    plt.imshow(img, cmap=cm.Greys_r)
                    plt.show()


def display_rate():
    fig = plt.figure()
    plt.scatter(vali_counter, vali_losses, color='red', zorder=10)
    plt.plot(train_counter, train_losses, color='blue', zorder=0)
    plt.legend(['Train Loss', 'Validation Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('cross entropy loss')
    plt.show()


def test():
    network.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.cuda()
            output = network(data)
            pred = output.data.max(1, keepdim=True)[1]
    # print(pred)
    # print(pred.shape)
    return pred


validation()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    validation()

# display_wrong()
# display_rate()

#pred = test()


def prediction_print(pred):
    count = 0
    with open('prediction6_CNN.csv', 'w', newline='', encoding='utf-8') as csv_file:
        print("Printing prediction to csv file... ")
        writer = csv.writer(csv_file)
        writer.writerow(['ID', 'Category'])
        for i in pred:
            writer.writerow([count, i.item()])
            count += 1
    csv_file.close()

# prediction_print(pred)
