# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Kiruthiga.B
RegisterNumber:212224040160  
*/

import pandas as pd
data = pd.read_csv("Employee.csv")
data
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

# Data:

<img width="1360" height="443" alt="image" src="https://github.com/user-attachments/assets/dd63fd28-6c6a-48f0-8808-27a6124d451f" />

# Data.head():

<img width="1313" height="232" alt="image" src="https://github.com/user-attachments/assets/db8d93a4-ff0b-4e79-88b4-aadc8eaab7eb" />

# Data.info():

<img width="547" height="378" alt="image" src="https://github.com/user-attachments/assets/42865bf4-0b1b-49ee-974f-2fb0238bfd2e" />

# Data.isnull().sum():

<img width="436" height="255" alt="image" src="https://github.com/user-attachments/assets/3428de9b-56f4-4383-9496-937ef913ecbd" />

# accuracy:

<img width="332" height="57" alt="image" src="https://github.com/user-attachments/assets/67023176-20fd-4b31-aba1-9d96c8fb9464" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
