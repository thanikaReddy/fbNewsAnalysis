""" Explores the relationships between the sentiment of a post and all other features.

Plots line plots binned by sentiment (width 0.1)
Reads from `/data/postsWithReciRoot.csv`
"""

import pandas as pd
import numpy as np
import operator
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
posts = pd.read_csv("../data/postsWithReciRoot.csv")
posts = posts[posts["mins_to_100_comment"]>0]
image_path = "figures/relationships_sentiment_{:d}.png"
image_num = 1
features = ["angry_per_sec",
            "sad_per_sec",
            "love_per_sec",
            "wow_per_sec",
            "haha_per_sec",
            "like_per_sec",
            "mins_to_first_comment",
            "shares_per_sec",
            "reciroot_reactions_per_sec",
            "vader_sentiment",
            "vader_pos",
            "vader_neg",
            "vader_neu",
            "mins_to_100_comment",
            "total_reactions",
            "shares",
            "reactions_per_sec"]
featureLabels = ["Angry", 
                "Sad", 
                "Love", 
                "Wow", 
                "Haha",
                "Likes",
                "Minutes until first comment", 
                "Number of shares per second",
                "Reactions per second",
                "Post Sentiment",
                "Post Positivity",
                "Post Negativity",
                "Post Neutrality",
                "Minutes until $100^{th}$ comment",
                "Total number of reactions",
                "Number of shares",
                "Total reactions per second"]
featureToLabel = dict(zip(features, featureLabels))

def plotBinnedBySentiment(yLabels):
    global image_num
    x = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]
    for yLabel in yLabels:
        print(yLabel)
        y = []
        for i in x:
            p = posts[((posts["vader_sentiment"]>=(i-1)) & (posts["vader_sentiment"]<(i+1)))]
            y.append(p[yLabel].mean())
        plt.plot(x, y, label=featureToLabel[yLabel])
        plt.show()
        plt.savefig(image_path.format(image_num), bbox_inches='tight')
        image_num += 1
        #plt.plot(x, y)
    plt.xlabel("VADER sentiment of a post")
    plt.legend(loc='best')

def plotBinnedBySentimentBox(yLabels):
    global image_num
    x = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]
    data = []
    for yLabel in yLabels:
        print(yLabel)
        y = []
        for i in x:
            p = posts[((posts["vader_sentiment"]>=(i-1)) & (posts["vader_sentiment"]<(i+1)))]
            data.append(list(p[yLabel]))
        #plt.plot(x, y, label=featureToLabel[yLabel])
    plt.boxplot(data, showfliers=False)
    plt.xticks([1,2,3,4,5,6,7,8,9,10], x)
    plt.xlabel("VADER sentiment of a post")
    plt.ylabel(featureToLabel[yLabels[0]])
    #plt.legend(loc='best')
    plt.show()
    plt.savefig(image_path.format(image_num), bbox_inches='tight')
    image_num += 1

plotBinnedBySentiment(yLabels = ["angry_per_sec", "sad_per_sec", "love_per_sec", "wow_per_sec", "haha_per_sec"])
plotBinnedBySentiment(yLabels = ["total_reactions"])
plotBinnedBySentiment(yLabels = ["shares"])
plotBinnedBySentiment(yLabels = ["reactions_per_sec"])
plotBinnedBySentiment(yLabels = ["shares_per_sec"])
plotBinnedBySentimentBox(yLabels = ["mins_to_first_comment"])
plotBinnedBySentiment(yLabels = ["mins_to_100_comment"])
plotBinnedBySentimentBox(yLabels = ["mins_to_100_comment"])