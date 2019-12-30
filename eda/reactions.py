""" Plots distributions of the fraction and log-transformed number of reactions (of each type) obtained per second.  

Explores the distribution of the fraction of each type of reaction on each post, 
i.e. it looks at the distribution of six features: 
frac_angry, frac_wow, frac_sad, frac_haha, frac_love and frac_like.
Explores the distribution of the number of reactions (of each type) obtained per second, 
and the distributions of their log transformed values (resulting in a normal distribution).

Reads from /data/postsProcessed.csv
Writes log-transformed features to /data/postsWithFracReactions.csv
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
posts = pd.read_csv("../data/postsProcessed.csv")

# Proportion of each reaction across the entire dataset
labels = ['Angry', 'Sad', 'Wow', 'Haha', 'Love', 'Like']
values = [posts['react_angry'].sum(), posts['react_sad'].sum(), posts['react_wow'].sum(), posts['react_haha'].sum(), posts['react_love'].sum(), posts['react_like'].sum()]
fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
fig.show()

# Calculate fraction of each type of reaction
posts['frac_like'] = posts['react_like']/posts['total_reactions']
posts['frac_angry'] = posts['react_angry']/posts['total_reactions']
posts['frac_haha'] = posts['react_haha']/posts['total_reactions']
posts['frac_love'] = posts['react_love']/posts['total_reactions']
posts['frac_sad'] = posts['react_sad']/posts['total_reactions']
posts['frac_wow'] = posts['react_wow']/posts['total_reactions']
posts['post_number'] = np.arange(len(posts))
posts['post_number'] = posts['post_number'] + 1

image_path = "figures/reactions_{:d}.png"
image_num = 1

# Distribution of all reaction fractions
frac_features = ['frac_wow', 'frac_sad', 'frac_love', 'frac_haha', 'frac_angry', 'frac_like']
for f in frac_features:
    fig = px.histogram(posts, x=f)
    fig.show()
    fig.write_image(image_path.format(image_num))
    image_num += 1
 
# Fraction of each type of reaction on each post, across all posts
postsFracReactions = posts[frac_features].copy()
z = postsFracReactions.T.to_numpy()
x = posts['post_number']
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

# Calculate number of reactions (of each type) per second and log-transform
per_sec_features = ['wow_per_sec', 'sad_per_sec', 'love_per_sec', 'haha_per_sec', 'angry_per_sec', 'like_per_sec']
log_per_sec_features = ['log_wow_per_sec', 'log_sad_per_sec', 'log_love_per_sec', 'log_haha_per_sec', 'log_angry_per_sec', 'log_like_per_sec']
num_features = ['react_wow', 'react_sad', 'react_love', 'react_haha', 'react_angry', 'react_like']
for i in range(len(per_sec_features)):
    posts[per_sec_features[i]] = (posts[num_features[i]]+1)/(posts["hrs_since_creation"]*60*60)
    posts[log_per_sec_features[i]] = np.log10(posts[per_sec_features[i]])
    fig = px.histogram(posts, x=log_per_sec_features[i])
    fig.show()
    fig.write_image(image_path.format(image_num))
    image_num += 1

# Write log-tranformed features to CSV
posts.to_csv ('../data/postsWithFracReactions.csv', index = None, header=True) 