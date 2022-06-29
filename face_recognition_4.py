import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
import face_detection_2 as data
dframe=data.df
x=dframe.drop(["Target"],axis="columns")
y=dframe[["Target"]]
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scl=StandardScaler()
x_scl=scl.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x_scl,y,train_size=0.8,random_state=0)
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
model_linear=SVC(kernel="linear")
model_linear.fit(x_train,y_train)
print("Accuracy is :",(model_linear.score(x_test,y_test))*100,"%")
score=cross_val_score(model_linear,x,y,cv=6)
print("cross validation score is :",score.mean())
"""from joblib import dump
dump(model_linear,"model.pkl")
dictionary={"messi ":0,"nirmala sitharaman ":1,"ratan tata ":2,"virat kohli  ":3}
import json
with open('data_map.json', 'w') as fp:
    json.dump(dictionary, fp)"""
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,model_linear.predict(x_test))
import seaborn as sns
sns.heatmap(cm,annot=True)
plt.show()