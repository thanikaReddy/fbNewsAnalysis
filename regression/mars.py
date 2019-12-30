""" Fits a MARS model to the top 6 features chosen by the multivariate linear regression model. 

Computes R2 score and RMSE.
Plots feature importances by RSS and number of subsets. 

Reads from /data/postsWithReciRoot.csv
"""

import pandas as pd
import numpy as np
import operator
import random 
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from statistics import mean
from pyearth import Earth
posts = pd.read_csv("../data/postsWithReciRoot.csv")
image_path = "figures/mars_{:d}.png"
image_num = 1

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

def mars(p, xLabels, yLabel):
    global image_num
    criteria = ('rss', 'gcv', 'nb_subsets')
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
    # Fit MARS model
    model = Earth(feature_importance_type=criteria)
    model.fit(X_train, y_train)
    # Make predictions
    predicted = model.predict(X_test)
    r2 = r2_score(y_test, predicted)
    mse = mean_squared_error(y_test, predicted)
    predicted = predicted.reshape(-1,1)
    # Plot residuals
    plotResiduals(y_test, predicted)
    # Print summary
    print(model.trace())
    print(model.summary())
    # Plot feature importances 
    importances = model.feature_importances_
    for crit in criteria:
        x = list(range(0,len(xLabels)))
        sorted_rss = [list(t) for t in sorted(zip(importances[crit], xLabels), reverse=True)]
        coeff = []
        feature = []
        for j in range(0,len(sorted_rss)):
            coeff.append(abs(sorted_rss[j][0]))
            feature.append(featureToLabel[sorted_rss[j][1]])
        plt.clf()
        plt.xticks(x,feature, rotation='vertical')
        plt.bar(x, coeff, align='center', alpha=0.5)
        plt.xlabel('Features')
        label = "Importance ("+crit+")"
        plt.ylabel(label)
        plt.tight_layout()
        label = "mars_imp_"+crit
        plt.show()
        plt.savefig(image_path.format(image_num), bbox_inches='tight')
        image_num += 1
    return r2, mse


features = ["log_love_per_sec","log_sad_per_sec","reciroot_reactions_per_sec", "log_wow_per_sec", "reciroot_mins_to_first_comment", "log_like_per_sec"]
r_arr = []
m_arr = []
x = range(100)
for i in range(1):
    r,m = mars(posts, features, "reciroot_mins_to_100_comment")
    r_arr.append(r)
    m_arr.append(m)

print("Mean R2 = ", mean(r_arr))
print("Mean RMSE = ", mean(m_arr))