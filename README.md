# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Vanisha Ramesh
RegisterNumber: 212222040174 
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/student_scores.csv')
#displaying the content in datafile
df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
y_test

#graph plot for training data
plt.scatter(x_train,y_train,color="darkseagreen")
plt.plot(x_train,regressor.predict(x_train),color="plum")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(x_test,y_test,color="darkblue")
plt.plot(x_test,regressor.predict(x_test),color="plum")
plt.title("Hours vs Scores (Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
1. df.head()

![Screenshot 2023-09-11 181606](https://github.com/Vanisha0609/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119104009/a4c0d11a-036b-43c6-aeb2-5fd4593e0fb2)


2. df.tail()
   
![Screenshot 2023-09-11 181732](https://github.com/Vanisha0609/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119104009/6147004d-de59-446e-8e80-35a8ad942475)

3. Array value of X
   
![Screenshot 2023-09-11 181832](https://github.com/Vanisha0609/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119104009/5f639995-8e0f-4ab7-a627-5b3831d86dd0)

4. Array value of Y
   
![Screenshot 2023-09-11 181910](https://github.com/Vanisha0609/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119104009/6609818a-6e7c-4972-b5fe-fc2ff796871b)

5. Values of Y prediction
    
![Screenshot 2023-09-11 182012](https://github.com/Vanisha0609/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119104009/c5ef8ec7-c6f0-488c-a38b-9fb79a036079)

6. Array values of Y test
    
![Screenshot 2023-09-11 182144](https://github.com/Vanisha0609/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119104009/fa2feba3-ed59-48ad-867e-09cd949a2de2)

7. Training Set Graph
    
![tset](https://github.com/Vanisha0609/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119104009/3fb136b9-7951-47af-89e0-71189af307cd)

8. Test Set Graph
    
![Screenshot 2023-09-11 182306](https://github.com/Vanisha0609/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119104009/d3092d86-1297-41da-bd58-84808785404e)

9. Values of MSE, MAE and RMSE
    
![Screenshot 2023-09-11 182318](https://github.com/Vanisha0609/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119104009/41f3f722-d0be-4613-8b3f-416bf7c81637)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
