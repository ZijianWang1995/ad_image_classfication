import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_iris
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report


"Step 1: Data proprecessing"
df_data = pd.read_csv('data.csv', header=None) #read data

df_label = pd.DataFrame(df_data, columns=[1558]) #extra the last column which is the label

df_data = pd.DataFrame.drop(df_data, columns = [1558]) #delete the last label column

#create train and test data sets, 30% data are splitted into test data set
X_train, X_test, y_train, y_test = train_test_split(df_data, df_label, test_size = 0.3, random_state = 40)


"Step 2: Decision tree model"
clf = tree.DecisionTreeClassifier(criterion="entropy") #create decision tree
clf = clf.fit(X_train, y_train) #fit the model

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
#print(precision_recall_fscore_support(y_test, y_pred, average=None))


#print(y_predict)
#print(cross_val_score(clf,X_test, y_test, cv = 10))



# Kernel Density Plot
import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15,8),)
ax=sns.kdeplot(data.loc[(data[1558] == 'ad.'),  0] , color='gray',shade=True,label='ad', bw=5)
ax=sns.kdeplot(data.loc[(data[1558] == 'nonad.'), 0] , color='g',shade=True, label='nonad', bw=5)
plt.title('Fare Distribution Survived vs Non Survived', fontsize = 25, pad = 40)
plt.ylabel("Frequency of Passenger Survived", fontsize = 15, labelpad = 20)
plt.xlabel("Fare", fontsize = 15, labelpad = 20);

a = data.loc[(data[1558] == 'ad.'),  0]