import os
import numpy as np
import cv2
from sklearn import svm
import random
import csv

def get_data():

    train_labels = "./train_labels.txt"
    train_features = "./waste_fc7_features_vgg16.txt"
    train_images = "./waste_image_name_list.txt"

    X_train = np.loadtxt(open(train_features))

    print "Features got"

    train_image_dict = {}
    train_labels_dict = {}

    f = open(train_labels)

    for line in f:
        line = line.strip().split('|')
        name = line[0].strip()
        label = int(line[2].strip())
        if label == 5:
            label = 3

        train_labels_dict[name] = label

    f.close()

    f = open(train_images)

    for i, line in enumerate(f):
        name = line.strip()
        train_image_dict[i] = name

    f.close()

    y_train = []

    for i in range(X_train.shape[0]):
        name = train_image_dict[i]
        label = train_labels_dict[name]
        y_train.append(label)

    y_train = np.asarray(y_train)

    print "Labels got"
    print "X_train shape, y_train shape", X_train.shape, y_train.shape

    idx = range(0,X_train.shape[0])
    random.shuffle(idx)

    X_test = np.loadtxt(open("./waste_test_fc7_features_vgg16.txt"))
    test_im_names = {}

    f = open('./waste_test_image_name_list.txt')
    for i, name in enumerate(f):
        name = name.strip()
        test_im_names[i] = name
    f.close()

    print 'X_test shape', X_test.shape

    return X_train[idx], y_train[idx], X_test, test_im_names

def my_svm(X_train, y_train, X_val, y_val, X_test, test_im_names):

    print "X_train shape, X_val shape", X_train.shape, X_val.shape

    X_train = list(X_train)
    y_train = list(y_train)

    X_val = list(X_val)
    y_val = list(y_val)

    X_test = list(X_test)

    lin_clf = svm.LinearSVC(multi_class='ovr')
    print "SVM Training"
    lin_clf.fit(X_train, y_train) 
    print "SVM Trained"
    '''
    print "SVM predict"
    y_pred = lin_clf.predict(X_val)

    hit = 0.0
    total = len(y_val)

    for i in range(total):
        if y_val[i] == y_pred[i]:
            hit+=1.0
    print "Accuracy", (float(hit)/total)*100
    '''
    print "Working on test data"
    #Test data work
    y_test = lin_clf.predict(X_test)

    print "Test labels", y_test[:100]
        
    ds = {1:'dry_cans', 0:'dry_other', 4:'dry_paper', 2:'dry_plastic', 3:'wet'}
    f = open('test_ouput.csv', 'w')

    for i in range(len(y_test)):
        label = y_test[i]
        s = test_im_names[i]+","+ds[label]+"\n"
        f.write(s)
    f.close()

    return 



if __name__ == "__main__":
    
    x,y, x_test, test_im_names = get_data()
    split = int(1.0*x.shape[0])
    my_svm(x[:split, :], y[:split], x[split:, :], y[split:], x_test, test_im_names)
    exit(0)

    



