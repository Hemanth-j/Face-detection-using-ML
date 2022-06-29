#building efficinet model
import warnings
warnings.filterwarnings("ignore")
import face_detection_2 as data
dframe=data.df
import matplotlib.pyplot as plt
x=dframe.drop(["Target"],axis="columns")
#print(x)
y=dframe[["Target"]]
#print(y)
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scl=StandardScaler()
x_scl=scl.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x_scl,y,train_size=0.8,random_state=0)
"""print(dframe)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)"""
"""
model1.fit(x_train,y_train)
print("Accuracy is : ",(model1.score(x_test,y_test))*100,"%")
from sklearn.metrics import confusion_matrix
y_pred=model1.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)
plt.xlabel("Y Predicted")
plt.ylabel("Y Truth")
plt.show()"""
from sklearn.model_selection import cross_val_score
def testing(model):
    model.fit(x_train,y_train)
    print("Accuracy of ",model,"is :",(model.score(x_test,y_test))*100,"%")
    score=cross_val_score(model,x,y,cv=6)
    print("cross validation score of ",model ,":",score.mean(),"\n")
from sklearn.linear_model  import LogisticRegression
model=LogisticRegression()
testing(model)
from sklearn.linear_model import SGDClassifier
model=SGDClassifier(random_state=0)
testing(model)
from sklearn.svm import SVC
model_linear=SVC(kernel="linear",probability=True)
model_rbf=SVC(kernel="rbf")
model_poly=SVC(kernel="poly")
testing(model_linear)
testing(model_rbf)
testing(model_poly)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
testing(model)
model=RandomForestClassifier()
testing(model)
from sklearn.ensemble import AdaBoostClassifier
model=AdaBoostClassifier(n_estimators=100)
testing(model)
model_ada_svm_linear=AdaBoostClassifier(n_estimators=100,random_state=0,base_estimator=SVC(kernel="linear",probability=True))
testing(model_ada_svm_linear)
from sklearn.ensemble import GradientBoostingClassifier
model=GradientBoostingClassifier(random_state=0)
testing(model)
"""from xgboost import XGBClassifier
model=XGBClassifier()
testing(model)"""
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
testing(model)
from catboost import CatBoostClassifier
model=CatBoostClassifier(iterations=100,learning_rate=0.3)
testing(model)
from lightgbm import LGBMClassifier
model=LGBMClassifier()
testing(model)