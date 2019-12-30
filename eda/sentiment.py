""" Plots distributions of each sentiment field.

Plots the distribution of each of the sentiment fields 
Displays information to help analyze the relationship between the fraction of 
reactions of each type received by a post, and its sentiment as computed by VADER

Reads from /data/postsWithFracReactions.csv
"""

import pandas as pd
import numpy as np
import operator
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
posts = pd.read_csv("../data/postsWithFracReactions.csv")
image_path = "figures/sentiment_{:d}.png"
image_num = 1

# Distribution of sentiment
fig = px.histogram(posts, x="vader_sentiment")
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1
# Log-transform sentiment 
posts["log_vader_sentiment"] = np.log10(posts["vader_sentiment"]+2)
fig = go.Figure()
fig.add_trace(go.Histogram(x=posts['log_vader_sentiment'],
                           xbins=dict(
                               start=-1,
                               end=1,
                               size=0.01
                           )))
fig = px.histogram(posts, x="vader_pos")
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1
fig = px.histogram(posts, x="vader_neu")
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1
fig = px.histogram(posts, x="vader_neg")
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1
# Plot fraction of each type of reactions for posts with negative sentiment
postsNegativeTextBlob = posts[posts['vader_sentiment'] <= -0.5]
postsFracReactions = postsNegativeTextBlob[['frac_wow', 'frac_sad', 'frac_love', 'frac_haha', 'frac_angry', 'frac_like']].copy()
z = postsFracReactions.T.to_numpy()
x = np.arange(z.shape[1])+1
y = ['Wow', 'Sad' ,'Love', 'Haha', 'Angry' ,'Like']
fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale='Viridis',
        colorbar=dict(
            title="Fraction of reactions")))
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1
# Mean fraction of each reaction type across all posts with negative sentiment
print("Post sentiment <= -0.5")
m = (postsNegativeTextBlob["frac_angry"] + postsNegativeTextBlob["frac_sad"]).mean()
print("Mean angry+sad fraction = ", m)
m = (postsNegativeTextBlob["frac_love"]).mean()
print("Mean love fraction = ", m)
m = (postsNegativeTextBlob["frac_haha"]).mean()
print("Mean haha fraction = ", m)
# Plot fraction of each type of reactions for posts with positive sentiment
postsPositiveTextBlob = posts[posts['vader_sentiment'] >= 0.5]
postsFracReactions = postsPositiveTextBlob[['frac_wow', 'frac_sad', 'frac_love', 'frac_haha', 'frac_angry', 'frac_like']].copy()
z = postsFracReactions.T.to_numpy()
x = np.arange(z.shape[1])+1
y = ['Wow', 'Sad' ,'Love', 'Haha', 'Angry' ,'Like']
fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale='Viridis',
        colorbar=dict(
            title="Fraction of reactions")))
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1
# Mean fraction of each reaction type across all posts with positive sentiment
print("Post sentiment <= -0.5")
m = (postsPositiveTextBlob["frac_angry"] + postsPositiveTextBlob["frac_sad"]).mean()
print("Mean angry+sad fraction = ", m)
m = (postsPositiveTextBlob["frac_love"]).mean()
print("Mean love fraction = ", m)
m = (postsPositiveTextBlob["frac_haha"]).mean()
print("Mean haha fraction = ", m)