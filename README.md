# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas for data manipulation, scikit-learn for machine learning operations, and any other necessary libraries.
2. Use pandas to read the CSV file containing your dataset (Employee.csv) and store it in a DataFrame.
3. Encode categorical variables if necessary, such as using Label Encoding for the "salary" column.
4. Define feature variables (x) and the target variable (y) based on relevant attributes for predicting the target variable.
5. Split the dataset into training and testing sets using train_test_split().
6. Fit the model to the training data using the fit() method.
7. Predict the target variable for the test set using the predict() method.
8. Evaluate the model's performance using appropriate metrics, such as accuracy score.
9. Pass the input data to the predict() method of the trained model to obtain predictions.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SHARMITHA V
RegisterNumber:  212223110048
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
*/
```

## Output:
![decision tree classifier model](sam.png)
![image](https://github.com/sharmitha3/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145974496/1a04501e-0dc7-48c8-a039-813c6fa3d57f)
![image](https://github.com/sharmitha3/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145974496/051bd4df-f227-498f-abde-5ed15c53c42b)
![image](https://github.com/sharmitha3/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145974496/e3502c22-e584-4280-8cb5-86a92d4ce147)
![image](https://github.com/sharmitha3/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145974496/a0134c78-dc93-4a3e-9597-e40a808eac00)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
