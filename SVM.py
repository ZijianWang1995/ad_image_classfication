import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, validation_curve
from sklearn.metrics import precision_recall_fscore_support,classification_report, confusion_matrix, recall_score, precision_score, accuracy_score, classification_report, f1_score
from sklearn.svm import SVC

# matplotlib.use('tkagg') # only MacPro needs this line to plot images


## step 1: Data preprocess
data = pd.read_csv("data.csv", header=None) #read the data
data.sample(5)  #see what the data look like

# replace labels. AD = 1; nonAD = 0
data.replace('ad.', 1, inplace=True)
data.replace('nonad.', 0, inplace=True)


## step 2: create train and test sets
df_label = pd.DataFrame(data, columns=[1558]) #extra the last column which is the label
df_data = pd.DataFrame.drop(data, columns = [1558]) #delete the last label column

#30% data are splitted into test data set, they are shuffled
X_train, X_test, y_train, y_test = train_test_split(df_data, df_label, test_size = 0.3, random_state = 40)

## step 2.1: feature scaling
st_scale = StandardScaler()
# transforming "train_x"
X_train = st_scale.fit_transform(X_train)
# transforming "test_x"
X_test = st_scale.transform(X_test)



## step 3: build SVM
# step 3.1 choose best C
Cs = [0.1, 1, 2,  4, 6, 8, 10, 12, 14, 16, 18] ## penalty parameter C for the error term. 

train_score, test_score = validation_curve(SVC(kernel='rbf', probability=True), 
        X_train, y_train, param_name = 'C', param_range= Cs , cv=10, scoring = 'accuracy')

train_score =  np.mean(train_score,axis=1)
test_score = np.mean(test_score,axis=1)

# plot out the accuracy trend with C
plt.plot(Cs,train_score, marker='.', drawstyle="steps-post", label = 'training')
plt.plot(Cs,test_score, marker='.', drawstyle="steps-post", label = 'testing')
plt.legend(loc='best')
plt.xlabel('C')
plt.ylabel('accuracy')
plt.savefig('svm_c.png', dpi=300)
plt.show()
# train_score = [0.84070233, 0.97644502, 0.97974255, 0.98640532, 0.98828976,
#        0.98869375, 0.99071273, 0.99104916, 0.99104916, 0.99111646,
#        0.99125105]
# test_score=[0.8394902 , 0.95940788, 0.96122241, 0.96425275, 0.96365034,
#        0.96424541, 0.96304059, 0.96243818, 0.96123336, 0.96063465,
#        0.96063465]
# C = 4 was choosen


# step 3.2 choose best gamma
gammas  = [0.0001, 0.001, 0.01, 0.1, 1]

train_score2, test_score2 = validation_curve(SVC(kernel='rbf', probability=True, C=4), 
        X_train, y_train, param_name = 'gamma', param_range= gammas , cv=10, scoring = 'accuracy')

train_score2 =  np.mean(train_score2,axis=1)
test_score2 = np.mean(test_score2,axis=1)

# plot out the accuracy trend with gamma
plt.plot(gammas,train_score2, marker='.', drawstyle="steps-post", label = 'training')
plt.plot(gammas,test_score2, marker='.', drawstyle="steps-post", label = 'testing')
plt.legend(loc='best')
plt.xlabel('gamma')
plt.ylabel('accuracy')
plt.savefig('svm_gamma.png', dpi=300)
plt.show()
# train_score2=[0.97267606, 0.9883571 , 0.99246231, 0.99818295, 0.99878865]
# test_score2 =[0.96180282, 0.95580076, 0.87647526, 0.86858534, 0.86675607]
# gamma = 0.0001

## step 4: build model 
svm = SVC(kernel='rbf', probability=True, C=4 , gamma=0.0001)

svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

## step 5: performance measurement
# c_m = confusion_matrix(y_test, y_pred)
# print(c_m)
svm_accuracy = accuracy_score(y_test, y_pred) *100
svm_precision = precision_score(y_test, y_pred) *100
svm_recall = recall_score(y_test, y_pred)*100
svm_f1 = f1_score(y_test, y_pred)*100
print("SVM accuracy: %0.2f" %svm_accuracy)
print("SVM precision: %0.2f" %svm_precision)
print("SVM recall: %0.2f" %svm_recall)
print("SVM f1: %0.2f" %svm_f1)

# SVM accuracy: 97.03
# SVM precision: 97.92
# SVM recall: 83.19
# SVM f1: 89.95
