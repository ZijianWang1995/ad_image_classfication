import pandas as pd
import numpy as np
import pandas_profiling
import matplotlib
import matplotlib.pyplot as plt 
from sklearn import tree
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.metrics import precision_recall_fscore_support,classification_report, confusion_matrix, recall_score, precision_score, accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

matplotlib.use('tkagg') # only MacPro needs this line to plot images

## step 1: Data preprocess
data = pd.read_csv("data.csv", header=None) #read the data

data.replace('ad.', 1, inplace=True)
data.replace('nonad.', 0, inplace=True)

## step 2: create train and test sets
df_label = pd.DataFrame(data, columns=[1558]) #extra the last column which is the label
df_data = pd.DataFrame.drop(data, columns = [1558]) #delete the last label column

#30% data are splitted into test data set, they are shuffled
X_train, X_test, y_train, y_test = train_test_split(df_data, df_label, test_size = 0.3, random_state = 40)


## Step 3 build and search best parameters for KNN
k_range = range(1,31)
weights_options=['uniform','distance']
param = {'n_neighbors':k_range, 'weights':weights_options}
cv = StratifiedShuffleSplit(n_splits=10, random_state=15)

# estimator = knn, param_grid = param, n_jobs = -1 to instruct scikit learn to use all available processors. 
grid = GridSearchCV(KNeighborsClassifier(), param,cv=cv,verbose = False, n_jobs=-1)
grid.fit(X_train,y_train)

knn = grid.best_estimator_
y_pred = knn.predict(X_test)

## step 4: performance measurement
knn_accuracy = accuracy_score(y_test, y_pred) * 100
knn_precision = precision_score(y_test, y_pred) * 100
knn_recall = recall_score(y_test, y_pred) * 100
knn_f1 = f1_score(y_test, y_pred) * 100
print("knn accuracy: %0.2f" % knn_accuracy)
print("knn precision: %0.2f" %knn_precision)
print("knn recall: %0.2f" %knn_recall)
print("knn f1: %0.2f" %knn_f1)


# knn accuracy: 94.49
# knn precision: 87.76
# knn recall: 76.11
# knn f1: 81.52