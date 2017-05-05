from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

print(__doc__)
import numpy as np
import sklearn
from numpy import array
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import tensorflow as tf


#FIELD_TEST = "DataCollection_DATASET1.csv"
#FIELD_TRAINING = "Site_survey_training.csv"
FIELD_TRAINING = "Site_survey_training_D.csv"
FIELD_TEST = "DataCollection_DATASET2.csv"

fieldtrain = tf.contrib.learn.datasets.base.load_csv_without_header(filename=FIELD_TRAINING,target_dtype=np.int,features_dtype=np.float32)

fieldtesting = tf.contrib.learn.datasets.base.load_csv_without_header(filename=FIELD_TEST,target_dtype=np.int,features_dtype=np.float32)

# Normalize each reading of the test data
def normalize_readings(readings,labels):
	strength_reading = []

	p=[]

	for i in range(0,len(labels)-1):
		if labels[i] == labels[i+1]:
			p.append(readings[i,3])
		else:
			p.append(readings[i,3])		
			if i < len(labels)-2:
				strength_reading.append(p)
				p=[]

	p.append(readings[len(labels)-1, 3])
		
	strength_reading.append(p)
	c=0
	z=0

	strength_signature=[]
	#print(strength_reading)

	#perform mean of the strength of each label separately
	for i in range(0,len(strength_reading)):
		mean = np.mean(strength_reading[i])
		std = np.std(strength_reading[i])
		for j in strength_reading[i]:
			#print(j)
			val = (j - mean)/std
			strength_signature.append(val)
	
	x= np.array(strength_signature,dtype=np.float32)
	strength_signature_final = x.T
	return strength_signature_final





readings = fieldtrain.data
labels = fieldtrain.target

testdata = fieldtesting.data
testlabel = fieldtesting.target

#calculate normalized value of each X reading



# comment this if you dont want normalized values of the strength reading
#readings[:,3] = strength_signature_final


#call KNN classifier
n_neighbors= 6


clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance',algorithm='ball_tree', p=1)
clf.fit(readings,labels)


result =[]
result_prob = []

for data in testdata:

	#print(data)
	Z = clf.predict(data)
	Z1 = clf.predict_proba(data)
	result.append(Z)
	result_prob.append(Z1)

#Z = clf.predict([[53.9,52.1457,13.6396,59.882]])
#Z1 = clf.predict_proba([[53.9,52.1457,13.6396,59.882]])
print(result)

accuracy = clf.score(testdata,testlabel)

print(accuracy)
#print(result)

#print(result_prob)

#strength_mean = np.mean(strength_reading)

#("rem",X_features)

#FFT signature of each reading

