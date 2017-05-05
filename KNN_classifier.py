
# coding: utf-8

# In[24]:

import json
import numpy as np
import math
import tensorflow as tf
import subprocess
import sklearn
from sklearn import neighbors,datasets
from Feature_extraction_SiteSurvey import normalize_readings

tf.logging.set_verbosity(tf.logging.ERROR)

def getHiddenLayerUnits(fv_size,class_size):
    first_layer_size = class_size + int((fv_size - class_size) * 0.75)
    second_layer_size = class_size + int((fv_size - class_size) * 0.5)
    third_layer_size = class_size + int((fv_size - class_size) * 0.25)
    
    return [first_layer_size, second_layer_size, third_layer_size]


#FIELD_TRAINING = "Site_survey_training.csv"
#fieldtest = tf.contrib.learn.datasets.base.load_csv_without_header(filename=FIELD_TRAINING,target_dtype=np.int,features_dtype=np.float32)


macList = set()

training_label = []

#minRssi = -50

requestData = []

training_fvs = []
strength_reading = []
p=[]

# FOR SITE SURVEY
with open('SiteSurvey.txt', 'r') as data_file:    
    for line in data_file:
        lineSplits = line.rstrip().split(',')
        label =lineSplits[11]
        fv = []
        fv.append(lineSplits[1])
        fv.append(lineSplits[2])
        fv.append(lineSplits[3])
        fv.append(lineSplits[5])
        fv = np.array(fv, dtype=float)
        
        training_label.append(label)
        training_fvs.append(fv)
        
training_label = np.array(training_label, dtype=int)
training_fvs = np.array(training_fvs, dtype=float)

#normalize the readings 
#strength_training = normalize_readings(training_fvs,training_label)
#training_fvs[:,3] = strength_training

print(training_fvs)

test_fvs=[]
test_label = []

# FOR DATA COLLECTION
with open('DataCollection.txt', 'r') as data_file:    
    for line in data_file:
        lineSplits = line.rstrip().split(',')
        label =lineSplits[11]
        fv = []
        fv.append(lineSplits[1])
        fv.append(lineSplits[2])
        fv.append(lineSplits[3])
        fv.append(lineSplits[5])
        fv = np.array(fv, dtype=float)
        
        test_label.append(label)
        test_fvs.append(fv)
        
test_label = np.array(test_label, dtype=int)
test_fvs = np.array(test_fvs, dtype=float)

# convert test readings into intensities --- NOT NEEDED FOR NOW
# for i in range(0,len(test_label)-1):
#     z=test_fvs[i,0]
#     z1=test_fvs[i,1]
#     z2=test_fvs[i,2]
#     test_fvs[i,0]=math.sqrt(z*z + z1*z1 + z2*z2)
#     test_fvs[i,1]=math.sqrt(z*z + z1*z1)
    
#normalize test readings
#strength_test = normalize_readings(test_fvs,test_label)
#test_fvs[:,3] = strength_test

print(test_fvs)

#call KNN classifier
n_neighbors= 8


clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance', algorithm = 'ball_tree', p=1)
clf.fit(training_fvs,training_label)

result =[]
result_prob = []

for data in test_fvs:

    #print(data)
    Z = clf.predict(data)
    Z1 = clf.predict_proba(data)
    result.append(Z)
    result_prob.append(Z1)

print(result)

accuracy = clf.score(test_fvs,test_label)

print(accuracy)


