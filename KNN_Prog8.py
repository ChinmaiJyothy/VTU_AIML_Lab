import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
data=load_iris()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['Class']=data.target_names[data.target]
df.head()
x=df.iloc[:,:-1].values
y=df.Class.values
print(x[:5])
print(y[:5])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier
knn_classifier=KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(x_train,y_train)
predictions=knn_classifier.predict(x_test)
print(predictions)
from sklearn.metrics import accuracy_score,confusion_matrix
print('Training accuracy score is : ',accuracy_score(y_train,knn_classifier.predict(x_train)))
print('Testing accuracy score is : ',accuracy_score(y_test,knn_classifier.predict(x_test)))
print('Training Confusion is : ',confusion_matrix(y_train,knn_classifier.predict(x_train)))
print('Testing Confusion is : ',confusion_matrix(y_test,knn_classifier.predict(x_test)))

#Output
#[[5.1 3.5 1.4 0.2]
# [4.9 3.  1.4 0.2]
# [4.7 3.2 1.3 0.2]
# [4.6 3.1 1.5 0.2]
# [5.  3.6 1.4 0.2]]
#['setosa' 'setosa' 'setosa' 'setosa' 'setosa']
#['setosa' 'virginica' 'setosa' 'virginica' 'virginica' 'versicolor'
# 'setosa' 'virginica' 'setosa' 'virginica' 'versicolor' 'versicolor'
# 'virginica' 'setosa' 'versicolor' 'virginica' 'setosa' 'virginica'
# 'versicolor' 'setosa' 'virginica' 'setosa' 'virginica' 'versicolor'
# 'virginica' 'setosa' 'setosa' 'virginica' 'versicolor' 'virginica']
#Training accuracy score is :  0.9833333333333333
#Testing accuracy score is :  0.9
#Training Confusion is :  [[40  0  0]
# [ 0 39  1]
# [ 0  1 39]]
#Testing Confusion is :  [[10  0  0]
# [ 0  7  3]
# [ 0  0 10]]
