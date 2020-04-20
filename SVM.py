import math
import time
import csv
import numpy as np
import pandas as pd
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# DIRECTORIES
csv_file = 'Data/train_labels.csv'
train_pics = 'Data/train_data_processed3b_thresh.pkl'
test_pics = 'Data/test_data_processed3b_thresh.pkl'

train_images = pd.read_pickle(train_pics)
test_images = pd.read_pickle(test_pics)


test_images = test_images.reshape(len(test_images), -1)
train_images_reshaped = train_images.reshape(len(train_images), -1)

label_file = pd.read_csv(csv_file)
train_labels = label_file['Category'].values

X_training, X_validation, y_training, y_validation = train_test_split(train_images_reshaped, train_labels, test_size=0.2)

print('Training...')
svc_linear = SVC(C=10, kernel='poly', gamma='auto', random_state=0)
svc_linear.fit(X_training, y_training)

print('Predicting...')
pred_vali = svc_linear.predict(X_validation)
print("Accuracy is ", accuracy_score(y_validation, pred_vali) * 100)


pred_test = svc_linear.predict(test_images)

print(pred_test.shape)

# count = 0
# with open('prediction_logreg.csv', 'w', newline='', encoding='utf-8') as csv_file:
#     print("Printing prediction to csv file... ")
#     writer = csv.writer(csv_file)
#     writer.writerow(['ID', 'Category'])
#     for i in all_test_y_pred:
#         writer.writerow([count, i])
#         count += 1

# csv_file.close()
