""" Fits a multivariate linear regression model to all features, to predict mins_to_100_comment  

Fits a multivariate linear regression model multiple times - the number of times is defined by NUMITER.
Each iteration, 
* One feature is added to the model at a time and the change in R2 and RMSE is noted for each feature. 
* The order in which features are added changes randomly.
* P-values and coefficients are displayed. 
Over all iterations, 
Keeps track of the addition of which six features leads to the largest increase in the R2 score.

Reads from /data/postsWithReciRoot.csv
"""

import pandas as pd
import numpy as np
import operator
import random 
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from yellowbrick.regressor import ResidualsPlot
from statistics import mean
from math import log
posts = pd.read_csv("../data/postsWithReciRoot.csv")
image_path = "figures/multivariate_{:d}.png"
image_num = 1

""" Number of iterations of linear regression to run. """
NUMITER = 1

""" Create labels for each feature """
features = ["log_angry_per_sec",
            "log_sad_per_sec",
            "log_love_per_sec",
            "log_wow_per_sec",
            "log_haha_per_sec",
            "log_like_per_sec",
            "reciroot_mins_to_first_comment",
            "reciroot_shares_per_sec",
            "reciroot_reactions_per_sec",
            "vader_sentiment",
            "vader_pos",
            "vader_neg",
            "vader_neu"]
featureLabels = ["Angry reactions per second", 
                "Sad reactions per second", 
                "Love reactions per second", 
                "Wow reactions per second", 
                "Haha reactions per second",
                "Likes per second",
                "Minutes until first comment", 
                "Shares per second",
                "Reactions per second",
                "Post Sentiment",
                "Post Positivity",
                "Post Negativity",
                "Post Neutrality"]
featureToLabel = dict(zip(features, featureLabels))

""" Plots residuals for a given pair of arrays of predicted and actual values. """
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
    plt.savefig("../paper_images/multireg_residuals.png")
    plt.show()
    plt.savefig(image_path.format(image_num), bbox_inches='tight')
    image_num += 1

""" Fits a multivariate linear regression model for all predictors in xLabels.
    The feature yLabel is the outcome.
    Prints P-values and plots coefficients.
    Returns the R2 score and RMSE for the model. """
def multiRegression(p, xLabels, yLabel):
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
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)
    # Make predictions
    predicted = lr.predict(X_test)
    r2 = r2_score(y_test, predicted)
    mse = mean_squared_error(y_test, predicted)
    if len(xLabels) == 13:
        # P-values and coefficients
        X2_train = sm.add_constant(X_train)
        est = sm.OLS(y_train, X2_train)
        est2 = est.fit()
        print(est2.summary())
        print("p-values")
        print(est2.pvalues)
        l = xLabels
        l.insert(0, "Const")
        params = [abs(number) for number in est2.params]
        sorted_coeffs = [list(t) for t in sorted(zip(est2.params, l), reverse=True)]
        print("Coefficients")
        print(sorted_coeffs)
        sorted_abs_coeffs = [list(t) for t in sorted(zip(params, l), reverse=True)]
        print("Absolute values of coefficients")
        print(sorted_abs_coeffs)
        coeff = []
        feature = []
        for j in range(1,len(sorted_abs_coeffs)):
            coeff.append(abs(sorted_abs_coeffs[j][0]))
            feature.append(featureToLabel[sorted_abs_coeffs[j][1]])
        x = list(range(0,len(feature)))
        # Plot coefficients
        plt.clf()
        plt.figure(figsize=(10,15))
        plt.xticks(x,feature, rotation='vertical')
        plt.bar(x, coeff, align='center', alpha=0.5)
        plt.yscale('log')
        plt.xlabel('Features')
        plt.ylabel('Log of absolute value of coefficient')
        plt.tight_layout()
        plt.show()
        plt.savefig(image_path.format(image_num), bbox_inches='tight')
        image_num += 1
        coeff = []
        feature = []
        for j in range(0,len(sorted_coeffs)-1):
            coeff.append(sorted_coeffs[j][0])
            feature.append(featureToLabel[sorted_coeffs[j][1]])
        plt.clf()
        plt.figure(figsize=(10,15))
        plt.xticks(x,feature, rotation='vertical')
        plt.bar(x, coeff, align='center', alpha=0.5)
        plt.yscale('log')
        plt.xlabel('Features')
        plt.ylabel('Log of coefficient')
        plt.tight_layout()
        plt.show()
        plt.savefig(image_path.format(image_num), bbox_inches='tight')
        image_num += 1
    return r2, mse

""" Shuffles a list of predictors and calls multiRegression adding one predictor at a time.
    Returns a list of R2 scores and RMSEs. """
def stepwiseMultiRegression():
    global image_num
    features = ["log_angry_per_sec","log_sad_per_sec","log_love_per_sec","log_wow_per_sec","log_haha_per_sec","log_like_per_sec","reciroot_mins_to_first_comment","reciroot_shares_per_sec","reciroot_reactions_per_sec","vader_sentiment","vader_pos","vader_neg","vader_neu"]
    random.shuffle(features)
    r2_stepwise = []
    mse_stepwise = []
    for i in range(len(features)):
        r,m = multiRegression(posts, features[0:i+1], "reciroot_mins_to_100_comment")
        r2_stepwise.append(r)
        mse_stepwise.append(m)
    return r2_stepwise, mse_stepwise, features

""" Performs stepwiseMultiRegression multiple times and keeps track of top 6 features for each iteration.
    Plots the change in R2 score and RMSE for each iteration.
    Returns a dictionary that 
    maps each feature to the number of iterations in which it was part of the top 6 features"""
def runStepwiseFor(numIter):
    global image_num
    # Dictionary that stores feature:count (in the top 6 features, across all stepwise iterations)
    numOcc = {}
    featureNames = []
    for j in range(numIter):
        r2, mse, f= stepwiseMultiRegression()
        for g in range(len(f)):
            featureNames.append(featureToLabel[f[g]])
        x = list(range(0,len(f)))
        plt.clf()
        plt.xticks(x,featureNames, rotation='vertical')
        plt.plot(x, r2)
        plt.xlabel('Features')
        plt.ylabel('R2')
        plt.tight_layout()
        plt.show()
        plt.savefig(image_path.format(image_num), bbox_inches='tight')
        image_num += 1
        plt.clf()
        plt.xticks(x, featureNames, rotation='vertical')
        plt.plot(x, mse)
        plt.xlabel('Features')
        plt.ylabel('RMSE')
        plt.tight_layout()
        plt.show()
        plt.savefig(image_path.format(image_num), bbox_inches='tight')
        image_num += 1
        # Calculate % increase in r2 and decrease in rmse that each feature leads to 
        r2_delta = []
        rmse_delta = []
        for i in range(1,len(r2)):
            val = (r2[i]-r2[i-1])
            r2_delta.append(val)
            val = (mse[i]-mse[i-1])
            rmse_delta.append(val)
        delta_df = pd.DataFrame(list(zip(featureNames[1:],r2_delta, rmse_delta)), 
                       columns =['Feature added', 'Change in R2', 'Change in RMSE']) 
        print(delta_df.sort_values(by=['Change in R2'], ascending=False))
        x = list(range(0,len(f)-1))
        plt.clf()
        plt.xticks(x,featureNames[1:], rotation='vertical')
        plt.bar(x, r2_delta, align='center', alpha=0.5)
        plt.xlabel('Features')
        plt.ylabel('Change in R2')
        plt.tight_layout()
        plt.show()
        plt.savefig(image_path.format(image_num), bbox_inches='tight')
        image_num += 1
        plt.clf()
        plt.xticks(x, featureNames[1:], rotation='vertical')
        plt.bar(x, rmse_delta, align='center', alpha=0.5)
        plt.xlabel('Features')
        plt.ylabel('Change in RMSE')
        plt.tight_layout()
        plt.show()
        plt.savefig(image_path.format(image_num), bbox_inches='tight')
        image_num += 1
        for e in delta_df.sort_values(by=['Change in R2'], ascending=False)[0:6]["Feature added"]:
            if e in numOcc:
                numOcc[e] += 1
            else:
                numOcc[e] = 1
    return numOcc

numOcc = runStepwiseFor(NUMITER)
# Sort by the number of times each feature occured in the top 6 list in each itertion of stepwise linear regression
sorted_numOcc = sorted(numOcc.items(), key=operator.itemgetter(1), reverse = True)
print(sorted_numOcc)
print("\nTop 6 features and frequency of occurance: ")
for s in sorted_numOcc:
    print(s)