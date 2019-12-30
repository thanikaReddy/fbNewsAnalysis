""" Explores the relationships between each (transformed) feature and the mins_to_100_comment feature.

Plots either scatter plots, or line plots showing the mean over a certain category.
Reads from `/data/postsWithReciRoot.csv`
"""
import pandas as pd
import numpy as np
import operator
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
posts = pd.read_csv("../data/postsWithReciRoot.csv")
image_path = "figures/relationships_outcome_{:d}.png"
image_num = 1

# VADER sentiment vs. mins_to_100_comment
fig = px.scatter(posts, x="vader_sentiment", y="reciroot_mins_to_100_comment")
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1

# Number of each reaction per second vs. minutes to 100 comments
reaction = ["log_angry_per_sec", "log_sad_per_sec", "log_love_per_sec", "log_haha_per_sec", "log_wow_per_sec", "log_like_per_sec"]
for r in reaction:
    fig = px.scatter(posts, x=r, y="reciroot_mins_to_100_comment")
    fig.show()
    fig.write_image(image_path.format(image_num))
    image_num += 1

# Minutes to first comment and minutes to 100 comments
fig = px.scatter(posts, x="reciroot_mins_to_first_comment", y="reciroot_mins_to_100_comment")
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1

# Shares per sec and minutes to 100 comments
fig = px.scatter(posts, x="reciroot_shares_per_sec", y="reciroot_mins_to_100_comment")
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1

# Reactions per sec and minutes to 100 comments
fig = px.scatter(posts, x="reciroot_reactions_per_sec", y="reciroot_mins_to_100_comment")
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1

# Day of week, time of day and mintues to 100 comments
days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
y = []
for d in days:
    y.append(posts[posts[d]==1]["reciroot_mins_to_100_comment"].mean())
fig = go.Figure(data=go.Scatter(x=days, y=y))
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1
y = []
for d in days:
    y.append(posts[posts[d]==1]["mins_to_100_comment"].mean())
fig = go.Figure(data=go.Scatter(x=days, y=y))
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1
segment = ["pre_work", "work", "post_work", "sleep"]
y = []
for s in segment:
    y.append(posts[posts[s]==1]["reciroot_mins_to_100_comment"].mean())
fig = go.Figure(data=go.Scatter(x=segment, y=y))
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1
y = []
for s in segment:
    y.append(posts[posts[s]==1]["mins_to_100_comment"].mean())
fig = go.Figure(data=go.Scatter(x=segment, y=y))
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1