#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
import numpy as np

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

##### Make Training Set Smaller (1%) For Speed
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]
#####

# time measurement
t0 = time()

clf = SVC(kernel='rbf', C=10000.0) # OR: linear
clf.fit(features_train, labels_train)

# return time it took to train
print ("training time", round(time()-t0, 3), "s")

pred = clf.predict(features_test)
accuracy = accuracy_score(labels_test, pred)

print("accuracy", accuracy)
print classification_report(labels_test, pred)

print np.count_nonzero(pred == 1)


##### Print Out Specific Predictions
"""
tenth=pred[10]
twentysixth=pred[26]
fiftyeth=pred[50]
print "10th: ", tenth
print "26th: ", twentysixth
print "50th: ", fiftyeth
"""
#########################################################
