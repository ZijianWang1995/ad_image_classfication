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


## Step 3: build model 1: decision tree 
# set parameters
max_depth = range(1,30)
criterion = ["entropy"] #using entropy to decide nodes
param = {'max_depth':max_depth, 'criterion':criterion}

# search the best parameters
grid = GridSearchCV(DecisionTreeClassifier(), param_grid= param, verbose=False, return_train_score=True)
grid.fit(X_train, y_train) #fit the model

# print parameters and scores
# print(grid.best_params_)
# print(grid.best_score_)
# print(grid.best_estimator_)

# the best tree
dt = grid.best_estimator_

# plot the tree
dot_data = tree.export_graphviz(decision_tree=dt, out_file=None)
graph = graphviz.Source(dot_data)
#graph
graph.format='png'
graph.render('dt_raw', view=True)

# predict test dataset
y_pred = dt.predict(X_test)

## step 4: performance measurement
# c_m = confusion_matrix(y_test, y_pred)
# print(c_m)
dc_accuracy = accuracy_score(y_test, y_pred)
dc_precision = precision_score(y_test, y_pred)
dc_recall = recall_score(y_test, y_pred)
dc_f1 = f1_score(y_test, y_pred)
print("decision tree accuracy: %0.2f" %dc_accuracy)
print("decision tree precision: %0.2f" %dc_precision)
print("decision tree recall: %0.2f" %dc_recall)
print("decision tree f1: %0.2f" %dc_f1)


## step 5: prune the tree
path = dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# copy another tree for pruning
dc_temp = grid.best_estimator_

# clfs = []
train_scores = []
test_scores = []
for i in ccp_alphas:
    dc_temp.set_params(ccp_alpha = i)
    dc_temp.fit(X_train, y_train)
    train_scores.append(dc_temp.score(X_train, y_train))
    test_scores.append(dc_temp.score(X_test, y_test))

 
# delete the last one 
ccp_alphas = ccp_alphas[:-1]
train_scores = train_scores[ :-1]
test_scores = test_scores[ :-1]


# plot out
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.plot(ccp_alphas, train_scores, marker='.', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='.', label="test",
        drawstyle="steps-post")
ax.legend()
#present the max point 
plt.plot(ccp_alphas[test_scores.index(max(test_scores))],max(test_scores) , color='red', marker='o')
plt.show()
fig.savefig("overfit.png",dpi=300)


## step 6: build a new tree with the prune parameter
# find out the the best ccp_alpha
p = test_scores.index(max(test_scores))
ccp_alpha_final = ccp_alphas[p]

dt_final = grid.best_estimator_
dt_final.set_params(ccp_alpha=ccp_alpha_final)
dt_final.fit(X_train, y_train) #fit the model
y_pred2=dt_final.predict(X_test) #predict the model

## step 7: measure the performance of the pruned decision tree
# c_m2 = confusion_matrix(y_test, y_pred2)
# print(c_m2)
dc_accuracy2 = accuracy_score(y_test, y_pred2)
dc_precision2 = precision_score(y_test, y_pred2)
dc_recall2 = recall_score(y_test, y_pred2)
dc_f12 = f1_score(y_test, y_pred2)
print("pruned decision tree accuracy: %0.2f" %dc_accuracy2)
print("pruned decision tree precision: %0.2f" %dc_precision2)
print("pruned decision tree recall: %0.2f" %dc_recall2)
print("pruned decision tree f1: %0.2f" %dc_f12)

dot_data = tree.export_graphviz(decision_tree=dt_final, out_file=None)
graph = graphviz.Source(dot_data)
#graph
graph.format='png'
graph.render('dt_prune', view=True)

## step 8: plot out the performace of two trees
labels = ['accuracy', 'precision', 'recall', 'f1']
dc_meansure_raw = [round(dc_accuracy * 100,2), round(dc_precision * 100,2), round(dc_recall * 100,2), round(dc_f1 * 100,2)]
dc_meansure_prune =[round(dc_accuracy2 * 100,2), round(dc_precision2 * 100,2), round(dc_recall2 * 100,2), round(dc_f12 * 100,2)]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, dc_meansure_raw, width, label='Before')
rects2 = ax.bar(x + width/2, dc_meansure_prune, width, label='After pruning')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('%')
#ax.set_title('Measure Criterion')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()
fig.savefig("performance2.png",dpi=300)