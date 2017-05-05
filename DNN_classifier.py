
# coding: utf-8

# In[24]:

import json
import numpy as np
import math
import tensorflow as tf
import subprocess

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

# both datasets have intensities converted. No need for further processing here

# FOR SITE SURVEY
with open('SiteSurvey_2sec.txt', 'r') as data_file:    
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

print(training_fvs)

test_fvs=[]
test_label = []

# FOR DATA COLLECTION
with open('TestData_2sec.txt', 'r') as data_file:    
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

# convert readings into intensities
# for i in range(0,len(test_label)-1):
#     z=test_fvs[i,0]
#     z1=test_fvs[i,1]
#     z2=test_fvs[i,2]
#     test_fvs[i,0]=math.sqrt(z*z + z1*z1 + z2*z2)
#     test_fvs[i,1]=math.sqrt(z*z + z1*z1)
    

print(test_fvs)

feature_columns = [tf.contrib.layers.real_valued_column("",dimension=4)]
# size of features
fv_size = 4

#number of class labels
class_size = 12

subprocess.call(["rm", "-rf","/tmp/field_model"])

# this command produces 50% accuracy using the function getHiddenLayerUnits
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units = getHiddenLayerUnits(fv_size,class_size), n_classes=class_size, model_dir="/tmp/field_model")

# this command produces 100% accuracy using hardcoded hidden_layer units: 10,20,10
#classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units = getHiddenLayerUnits(fv_size,class_size), n_classes=class_size, model_dir="/tmp/field_model_fifth_5")

# this command produces 100% accuracy
#classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units = getHiddenLayerUnits(fv_size,class_size), n_classes=class_size, model_dir="/tmp/field_model_4")




classifier.fit(x=training_fvs,y=training_label,steps=100)

accuracy_score = classifier.evaluate(x=test_fvs,y=test_label)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))


