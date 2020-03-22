import pandas as pd
import numpy as np
import pandas_profiling
from sklearn import tree
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_iris
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report


data = pd.read_csv("data.csv", header=None) #read the data
data.sample(5)  #see what the data look like

#check missing values
def missing_percentage(df):
    """This function takes a DataFrame(df) as input and returns two columns:
        * total missing values
        * total missing values percentage"""
    total = df.isnull().sum().sort_values(ascending = False)
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)
    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])

# apply the function to the data
missing_percentage(data)

