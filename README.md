# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:


Developed by: ASTLE JOE A S 

RegisterNumber: 212224240019 
```
Program to implement the simple linear regression model for predicting the marks scored.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv(r"C:\Users\astle\Downloads\DATASET-20250226\student_scores.csv")

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

plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color="purple")
plt.plot(x_test,regressor.predict(x_test),color="yellow")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE; = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE =',mae)

rmse=np.sqrt(mse)

print("RMSE = ",rmse)

```

## Output:
![image](https://github.com/user-attachments/assets/38217b6c-c50b-4cff-b6fd-6f4ca19ac592)

![image](https://github.com/user-attachments/assets/5b263a24-f11d-4322-9e22-dc63e8187821)

![image](https://github.com/user-attachments/assets/6903282f-5b5b-49d2-a163-c1f7f5f0f03d)

![image](https://github.com/user-attachments/assets/79999b6e-a053-404f-ae57-bfeb0d3e200c)

![image](https://github.com/user-attachments/assets/1b25e5b3-183f-43cf-b4f8-f85fde26231c)

![image](https://github.com/user-attachments/assets/91104fff-04e9-4d1b-a93a-9124706a2efb)

![image](https://github.com/user-attachments/assets/052321b9-5893-4d59-bcea-98a8e469b915)

![image](https://github.com/user-attachments/assets/f91f8b5f-6a87-45e0-ad1f-b7a4391a9b10)

![image](https://github.com/user-attachments/assets/198b4de0-426b-4372-84aa-a0c7e11d1970)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
