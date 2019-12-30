""" Fits an SVR RBF model to the top 6 features chosen by the multivariate linear regression model. 

Performs a grid search over multiple kernels and coefficient values. 
Computes R2 score and RMSE.

Reads from /data/postsWithReciRoot.csv
"""

import pandas as pd
import numpy as np
import operator
import random 
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn import datasets, linear_model
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from statistics import mean
from pyearth import Earth
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
posts = pd.read_csv("../data/postsWithReciRoot.csv")
image_path = "figures/svr_{:d}.png"
image_num = 1

def plotResiduals(y_test, predicted):
    global image_num
    residuals = y_test - predicted
    plt.subplot(1, 2, 1)
    plt.axhline(y=0, color='k', linestyle='-.')
    plt.scatter(predicted, residuals, color='b')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.subplot(1, 2, 2)
    plt.hist(residuals, normed=True, bins=40)
    plt.show()
    plt.savefig(image_path.format(image_num), bbox_inches='tight')
    image_num += 1

# Fits an SVR RBF model with the parameters chosen by the grid search 
def svrRbf(p, xLabels, yLabel):
    global image_num
    # Randomly shuffle rows
    p = p.sample(frac=1).reset_index(drop=True)
    # Split train and test
    twentyPercent = -1*round(p.shape[0]*0.2)
    n = len(xLabels)
    xCol = p[xLabels].values.reshape(-1,n)
    X_train = xCol[:twentyPercent]
    X_test = xCol[twentyPercent:]
    y_train = p[yLabel][:twentyPercent].values.reshape(-1,1)
    y_test = p[yLabel][twentyPercent:].values.reshape(-1,1)
    # Fit linear regression model
    model = SVR(kernel='rbf', C=0.1, gamma=0.1, epsilon=0.0001)
    model.fit(X_train, y_train)
    # Make predictions
    predicted = model.predict(X_test)
    r2 = r2_score(y_test, predicted)
    mse = mean_squared_error(y_test, predicted)
    predicted = predicted.reshape(-1,1)
    # Plot residuals
    plotResiduals(y_test, predicted)
    return r2, mse

# Performs a grid search to find best coefficients for this data
def svrParameterTuning(p, xLabels, yLabel):
    # Randomly shuffle rows
    p = p.sample(frac=1).reset_index(drop=True)
    # Use 1% of the data
    onePercent = -1*round(p.shape[0]*0.2)
    p = p[:onePercent]
    # Reshape data
    n = len(xLabels)
    X = p[xLabels].values.reshape(-1,n)
    # Scale features
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    Y = p[yLabel].values.reshape(-1,1)
    # Range of parameters to search within
    C_range = [0.1, 1, 100, 1000]
    epsilon_range = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    gamma_range = [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
    # Grid search
    svr = SVR()
    parameters = {'kernel': ('linear', 'linear'), 'C':C_range,'gamma': gamma_range,'epsilon': epsilon_range}
    grid = GridSearchCV(svr, parameters, cv=2, verbose=14)
    grid.fit(X, Y)
    print("The best classifier is: ", grid.best_estimator_)

features = ["log_love_per_sec","log_sad_per_sec","reciroot_reactions_per_sec", "log_wow_per_sec", "reciroot_mins_to_first_comment", "log_like_per_sec"]
#svrParameterTuning(posts, features, "reciroot_mins_to_100_comment")

r2, mse = svrRbf(posts, features, "reciroot_mins_to_100_comment")
print("R2 = ", r2)
print("MSE = ", mse)