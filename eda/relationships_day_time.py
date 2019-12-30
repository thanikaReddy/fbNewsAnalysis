""" Explores the relationships between the day/time of a post creation and all other features.

Plots line plots binned by one-hot encoded day/time features
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
image_path = "figures/relationships_day_time_{:d}.png"
image_num = 1

def plotBinnedByDay(yLabels):
    global image_num
    x = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for yLabel in yLabels:
        print(yLabel)
        y = []
        for day in x:
            p = posts[(posts["day_name"]==day)]
            y.append(p[yLabel].mean())
        plt.plot(x, y, label=yLabel)
    plt.xticks(x,x, rotation='vertical')
    plt.xlabel("day created")
    plt.ylabel("feature")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    plt.savefig(image_path.format(image_num), bbox_inches='tight')
    image_num += 1

def plotBinnedByTime(yLabels):
    global image_num
    x = ["pre_work", "work", "post_work", "sleep"]
    for yLabel in yLabels:
        print(yLabel)
        y = []
        for time in x:
            p = posts[(posts["time_segment"]==time)]
            y.append(p[yLabel].mean())
        plt.plot(x, y, label=yLabel)
    plt.xticks(x,x, rotation='vertical')
    plt.xlabel("time created")
    plt.ylabel("feature")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    plt.savefig(image_path.format(image_num), bbox_inches='tight')
    image_num += 1

plotBinnedByDay(["total_reactions"])
plotBinnedByDay(["reactions_per_sec"])
plotBinnedByDay(["shares"])
plotBinnedByDay(["shares_per_sec"])
plotBinnedByDay(["vader_sentiment"])
plotBinnedByDay(["vader_pos", "vader_neg"])
plotBinnedByDay(["mins_to_100_comment"])
plotBinnedByDay(["mins_to_first_comment"])

x = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
y1 = []
y2 = []
for day in x:
    p = posts[(posts["day_name"]==day)]
    tot = p.shape[0]
    frac_pos = p[p["vader_sentiment"]>=0.7].shape[0] / tot
    frac_neg = p[p["vader_sentiment"]<=-0.7].shape[0] / tot
    y1.append(frac_pos)
    y2.append(frac_neg)
plt.plot(x, y1, label="Fraction of posts with sentiment >= 0.7")
plt.plot(x, y2, label="Fraction of posts with sentiment <= -0.7")
plt.xticks(x,x, rotation='vertical')
plt.xlabel("day created")
plt.ylabel("feature")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()
plt.savefig(image_path.format(image_num), bbox_inches='tight')
image_num += 1


plotBinnedByTime(["total_reactions"])
plotBinnedByTime(["reactions_per_sec"])
plotBinnedByTime(["shares"])
plotBinnedByTime(["shares_per_sec"])
plotBinnedByTime(["vader_sentiment"])
plotBinnedByTime(["vader_pos", "vader_neg"])
plotBinnedByTime(["mins_to_100_comment"])
plotBinnedByTime(["mins_to_first_comment"])