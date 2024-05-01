# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. start.
2. Import chardet.
3. Read the dataset.
4. Import SVC from sklearn.
5. Fit the data in the model and run the algorithm.
6.stop.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: JEEVAGOWTHAM S
RegisterNumber: 212222230053
*/
```
```
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
### 1. Result output
![image](https://github.com/JeevaGowtham-S/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118042624/4ac8eeae-41c9-4fa1-a6f5-0d03b077ebd6)


### 2. data.head() 
![image](https://github.com/JeevaGowtham-S/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118042624/8a8379da-4dd3-4925-9c0b-15b56b20563c)


### 3. data.info()
![image](https://github.com/JeevaGowtham-S/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118042624/93123349-1e4c-4099-9fb7-bab744e1e54f)


### 4. data.isnull().sum()
![image](https://github.com/JeevaGowtham-S/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118042624/dd4ebbb7-eaa3-4530-8510-0daac48d9cb6)


### 5. Y_prediction value
![image](https://github.com/JeevaGowtham-S/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118042624/b776bac4-82f9-4e91-9461-d1c08a4d617b)


### 6. Accuracy value
![image](https://github.com/JeevaGowtham-S/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118042624/5788ee80-61c2-4f80-983c-71248b5e1c91)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
