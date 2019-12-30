""" Fits a univariate linear regression model between mins_to_100_comment and each other transformed feature. 

For each model:
Plots the residuals and line of fit
Computes the R2 score and RMSE

Reads from /data/postsWithReciRoot.csv
"""

import pandas as pd
import numpy as np
import operator
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from yellowbrick.regressor import ResidualsPlot
posts = pd.read_csv("../data/postsWithReciRoot.csv")
image_path = "figures/univariate_{:d}.png"
image_num = 1

def uniRegression(p, xLabel, yLabel):
    global image_num
    # Randomly shuffle rows
    p = p.sample(frac=1).reset_index(drop=True)
    # Split train and test
    twentyPercent = -1*round(p.shape[0]*0.2)
    xCol = p[xLabel].values.reshape(-1,1)
    X_train = xCol[:twentyPercent]
    X_test = xCol[twentyPercent:]
    y_train = p[yLabel][:twentyPercent].values.reshape(-1,1)
    y_test = p[yLabel][twentyPercent:].values.reshape(-1,1)
    # Fit linear regression model
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)
    # Make predictions
    predicted = lr.predict(X_test)
    r2 = r2_score(y_test, predicted)
    mse = mean_squared_error(y_test, predicted)
    # Plot expected vs. predicted
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, predicted, color='blue', linewidth=2)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()
    plt.savefig(image_path.format(image_num), bbox_inches='tight')
    image_num += 1
    print("R2 = ",r2)
    print("MSE = ",mse)
    visualizer = ResidualsPlot(lr)
    # Plot residuals
    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    visualizer.show()                 # Finalize and render the figure
    
features = ["log_angry_per_sec","log_sad_per_sec","log_love_per_sec","log_wow_per_sec","log_haha_per_sec","log_like_per_sec","reciroot_mins_to_first_comment","reciroot_shares_per_sec","reciroot_reactions_per_sec","vader_sentiment"]
for f in features:
    uniRegression(posts, f, "reciroot_mins_to_100_comment")

