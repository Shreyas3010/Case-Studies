import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math

data = pd.read_csv("../input/ENB2012_data.csv")

#data.columns=['Relative Compactness','Surface Area','Wall Area','Roof Area','Overall Height','Orientation','Glazing Area','Glazing Area Distribution','Heating Load','Cooling Load'
#]


X=data.iloc[:,0:8]
y1=data.iloc[:,8:9]
y2=data.iloc[:,9:10]

X1_train, X1_test, y1_train, y1_test = train_test_split(X,y1,test_size = 0.3,train_size=0.7, random_state = 0)
X2_train, X2_test, y2_train, y2_test = train_test_split(X,y2,test_size = 0.3,train_size=0.7, random_state = 0)

#y1 output

regr_1 = RandomForestRegressor(n_estimators=100,oob_score=True,max_depth=5)
regr_1.fit(X1_train, y1_train.values.ravel())

#predict

y1_test_pred=regr_1.predict(X1_test)
y1_train_pred=regr_1.predict(X1_train)

error_y1_test=((abs(y1_test['Y1'].values-y1_test_pred))*100)/y1_test['Y1'].values
error_y1_train=((abs(y1_train['Y1'].values-y1_train_pred))*100)/y1_train['Y1'].values

error_y1_test_mean=np.mean(error_y1_test)
error_y1_train_mean=np.mean(error_y1_train)



plt.figure(figsize=(25, 20))  
plt.scatter(range(len(y1_test)),error_y1_test)  
plt.ylabel('Error in Y1 test | actual-predicted |/actual (%) ')
#plt.xlabel()
plt.title('Error in Y1 test (Random Forest Regression)')
plt.suptitle('Error (mean) : '+str(error_y1_test_mean)+' OOB Score : '+str(regr_1.oob_score_))
plt.savefig('../graph/Energy_Efficiency_RandomForestRegression_y1_test_error.png')
plt.show()


plt.figure(figsize=(25, 20))  
plt.scatter(range(len(y1_train)),error_y1_train)  
plt.ylabel('Error in Y1 train | actual-predicted |/actual (%) ')
#plt.xlabel()
plt.title('Error in Y1 train (Random Forest Regression)')
plt.suptitle('Error (mean) : '+str(error_y1_train_mean)+' OOB Score : '+str(regr_1.oob_score_))
plt.savefig('../graph/Energy_Efficiency_RandomForestRegression_y1_train_error.png')
plt.show()

#y2 output

regr_2 = RandomForestRegressor(n_estimators=100,oob_score=True,max_depth=5)
regr_2.fit(X2_train, y2_train.values.ravel())

#predict

y2_test_pred=regr_2.predict(X2_test)
y2_train_pred=regr_2.predict(X2_train)

error_y2_test=((abs(y2_test['Y2'].values-y2_test_pred))*100)/y2_test['Y2'].values
error_y2_train=((abs(y2_train['Y2'].values-y2_train_pred))*100)/y2_train['Y2'].values

error_y2_test_mean=np.mean(error_y2_test)
error_y2_train_mean=np.mean(error_y2_train)

plt.figure(figsize=(25, 20))  
plt.scatter(range(len(y2_test)),error_y2_test)  
plt.ylabel('Error in Y2 test | actual-predicted |/actual (%) ')
#plt.xlabel()
plt.title('Error in Y2 test (Random Forest Regression)')
plt.suptitle('Error (mean) : '+str(error_y2_test_mean)+' OOB Score : '+str(regr_2.oob_score_))
plt.savefig('../graph/Energy_Efficiency_RandomForestRegression_y2_test_error.png')
plt.show()

plt.figure(figsize=(25, 20))  
plt.scatter(range(len(y2_train)),error_y2_train)  
plt.ylabel('Error in Y2 train | actual-predicted |/actual (%) ')
#plt.xlabel()
plt.title('Error in Y2 train (Decision Tree Regression)')
plt.suptitle('Error (mean) : '+str(error_y2_train_mean)+' OOB Score : '+str(regr_2.oob_score_))
plt.savefig('../graph/Energy_Efficiency_RandomForestRegression_y2_train_error.png')
plt.show()
