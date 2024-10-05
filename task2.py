import numpy as np
from sklearn.datasets import load_iris,fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import accuracy_score,mean_squared_error
#loading dataset
#iris dataset for classification
iris=load_iris()
X_iris=iris.data
y_iris=(iris.target!=0).astype(int)
#iris dataset contain 0,1,2 targets inwhich 0 is setosa,1 is versicolor ,2 is viginica
#if target=0, then it is setosa otherwise it will be versicolor or virginica
#.astype(Int) will convert this into integer constant.
#The California Housing dataset is a regression dataset where the task is
#  to predict the median house value in a neighborhood based on features like the
# average income, population, number of households, etc.
#so the housing dataset for regression
california_housing=fetch_california_housing()
X_boston=california_housing.data
y_boston=california_housing.target
#standardising the features for both the datasets
scaler=StandardScaler()
X_iris=scaler.fit_transform(X_iris)
X_boston=scaler.fit_transform(X_boston)
#split into the training and the testing datasets
X_trainset_iris,X_testing_iris,y_trainset_iris,y_testset_iris= train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)
X_trainset_boston, X_testset_boston, y_trainset_boston, y_testset_boston = train_test_split(X_boston, y_boston, test_size=0.2, random_state=42)

#selecting logistic regression from scratch for classification sector
class Logisticregressionscratch:
    def __init__(self, learning_rate=0.01, n_iters=1000):#created a constructor for the class
        self.learning_rate=learning_rate #it is basically thr alpha, inwhich it will be updated in each iteration
        self.n_iters=n_iters #number of iteration inwhich algorithm will run
        self.weights=None #it will store the coefficient of the regression model
        self.bias=None #this will store the bias or the intercept of the regression model

    def sigmoid(self,z):
        return 1/(1+np.exp(-z)) #The sigmoid function maps any real-valued number into the range [0, 1]
                                # sigma(z)=1/(1+e^-z)
                                #This is used to convert the linear model's output into probabilities, which are then used to make binary classifications.
    
    def fit(self,X,y):#: This method trains the logistic regression model by adjusting the weights and bias using gradient descent.
        n_samples,n_features=X.shape#n samples is the number of data points
        #n_features is the number of features
        self.weights=np.zeros(n_features)#Initializes the weights to zeros. Each feature gets its own weight
        self.bias=0 #initialising bias as zero

        for _ in range(self.n_iters):
            linear_model =np.dot(X,self.weights)+self.bias # this computes the linear combination of the x and the weight + bias, it is the the output
            #of the simple linear regression model
            y_predicted=self.sigmoid(linear_model)# The linear model output is passed through the sigmoid function to get the predicted probabilities (y_predicted), which will be values between 0 and 1.
            dw=(1/n_samples)*np.dot(X.T,(y_predicted-y))#The gradient of the loss function with respect to the weights.
            db=(1/n_samples)*np.sum(y_predicted-y) #The gradient of the loss function with respect to the bias.
            self.weights -= self.learning_rate*dw #updating the weight.
            self.bias -= self.learning_rate*db #updating the bias

    def predict(self,X): #This method is used to make predictions on new data after the model has been trained.
        linear_model=np.dot(X,self.weights)+self.bias # Compute the linear combination of the input features and the weights, plus the bias, just like in the training process.
        y_predicted=self.sigmoid(linear_model) # Pass the linear model output through the sigmoid function
        y_predicted_cls=[1 if i>0.5 else 0 for i in y_predicted] #This converts the predicted probabilities into binary class labels (1 if the probability is greater than 0.5, otherwise 0).
        return np.array(y_predicted_cls) #Returns the predicted class labels as a NumPy array.
    
#So the functions says that , the logical regression is Y=Xw+b. the w and b are weight and bias , 
#which we calculate from the iteration so when we found the w and b then it is applied into y
#if the value of y is greater than 0.5 then it is classified into 1, otherwise 0

#same kind of things happen in  linear regression also
class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias #This computes the predicted values by applying the learned weights and bias to the input features.
                                                 #Predict the target values by applying the linear model (without sigmoid).


log_reg_scratch = Logisticregressionscratch(learning_rate=0.1, n_iters=1000)
log_reg_scratch.fit(X_trainset_iris, y_trainset_iris)
y_pred_scratch = log_reg_scratch.predict(X_testing_iris)
lin_reg_scratch = LinearRegressionScratch(learning_rate=0.01, n_iters=1000)
lin_reg_scratch.fit(X_trainset_boston, y_trainset_boston)
y_pred_scratch_boston = lin_reg_scratch.predict(X_testset_boston)

log_reg_sklearn = LogisticRegression()
log_reg_sklearn.fit(X_trainset_iris, y_trainset_iris)
y_pred_sklearn = log_reg_sklearn.predict(X_testing_iris)
lin_reg_sklearn = LinearRegression()
lin_reg_sklearn.fit(X_trainset_boston, y_trainset_boston)
y_pred_sklearn_boston = lin_reg_sklearn.predict(X_testset_boston)

accuracy_scratch = np.sum(y_pred_scratch == y_testset_iris) / len(y_testset_iris)
accuracy_sklearn = accuracy_score(y_testset_iris, y_pred_sklearn)
mse_scratch = np.mean((y_pred_scratch_boston - y_testset_boston) ** 2)
mse_sklearn = mean_squared_error(y_testset_boston, y_pred_sklearn_boston)
print(f"Accuracy of Logistic Regression from Scratch: {accuracy_scratch:.4f}")
print(f"Accuracy of Logistic Regression with Scikit-learn: {accuracy_sklearn:.4f}")
print(f"MSE of Linear Regression from Scratch: {mse_scratch:.4f}")
print(f"MSE of Linear Regression with Scikit-learn: {mse_sklearn:.4f}")

# in the scratch function we use gradient descent with a learning rate alpha, depending upon that only
#the soultion for w and b will be getting.If these are not well-tuned, the model might not converge to an optimal solution or may converge slowly.
#Scikit-learn’s implementation uses more advanced optimization techniques such as Stochastic Gradient Descent (SGD), LBFGS, or Coordinate Descent

#The scratch implementation likely uses basic optimization methods like vanilla gradient descent.
#  While this works, it may be less efficient than more advanced optimization techniques like Stochastic Gradient Descent (SGD),
# BFGS, or Adam, which are used in library-based models (such as Scikit-learn).
#Implementing mathematical operations like the sigmoid function or gradient calculations can lead to numerical issues like overflow or underflow
#it may lead to inaccurate result.
