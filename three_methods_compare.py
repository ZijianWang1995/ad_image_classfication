import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

knn_accuracy= 94.49
knn_precision= 87.76
knn_recall= 76.11
knn_f1= 81.52

SVM_accuracy= 97.03
SVM_precision= 97.92
SVM_recall= 83.19
SVM_f1= 89.95

dt_accuracy= 96.61
dt_precision= 91.59
dt_recall= 86.73
dt_f1= 89.09

labels = ['accuracy', 'precision', 'recall', 'F1']

knn = [knn_accuracy, knn_precision, knn_recall, knn_f1]
SVM = [SVM_accuracy, SVM_precision, SVM_recall, SVM_f1]
dt = [dt_accuracy, dt_precision, dt_recall, dt_f1]

x = np.arange(len(labels))  # the label locations
width = 0.27  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width*2/3, knn, width, label='KNN')
rects2 = ax.bar(x + width/3, SVM,  width, label='SVM')
rects3 = ax.bar(x + width/3*4, dt, width, label='DT')

plt.ylim((75,100))
ax.set_ylabel('%')
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
autolabel(rects3)

fig.tight_layout()

plt.show()
fig.savefig("performance_feature.png",dpi=300)