import pandas as pd
import numpy as np
import pandas_profiling
import graphviz
import matplotlib
import matplotlib.pyplot as plt 
from sklearn import tree
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.metrics import precision_recall_fscore_support,classification_report, confusion_matrix, recall_score, precision_score, accuracy_score, classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier

matplotlib.use('tkagg') # only MacPro needs this line to plot images

## step 1: Data preprocess
data = pd.read_csv("data.csv", header=None) #read the data
data.sample(5)  #see what the data look like

# replace labels. AD = 1; nonAD = 0
data.replace('ad.', 1, inplace=True)
data.replace('nonad.', 0, inplace=True)

# check missing values
def missing_percentage(df):
    """This function takes a DataFrame(df) as input and returns two columns:
        * total missing values
        * total missing values percentage"""
    total = df.isnull().sum().sort_values(ascending = False)
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)
    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])

missing_percentage(data) # apply the function to the data


## step 2: create train and test sets
df_label = pd.DataFrame(data, columns=[1558]) #extra the last column which is the label
df_data = pd.DataFrame.drop(data, columns = [1558]) #delete the last label column

#30% data are splitted into test data set, they are shuffled
X_train, X_test, y_train, y_test = train_test_split(df_data, df_label, test_size = 0.3, random_state = 40)
