import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.model_selection import train_test_split


#sample dataset
X=np.array([[1000,2],[1500,3],[2000,4],[2500,5]]) #featur area,bedroom
Y=np.array([20000,30000,40000,50000])#prices

#split data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

#create ridge model
model =Ridge(alpha=1.0)#aplha= regularization strength 

#train model
model.fit(X_train,Y_train)

#predict
prediction=model.predict(X_test)

#evaluate
MSE=mean_squared_error(Y_test,prediction)
print("prediction:",prediction)
print("mean squared errror:",MSE)