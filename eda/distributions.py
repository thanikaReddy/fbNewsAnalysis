""" Plots the distribution of all other features and applies a reciprocal-root transform.

Plots the distribution of all other features: 
mins_to_first_comment
day of the week a post was created
time of the day it was created
number of shares per second
number of reactions per second
Applies a reciprocal root transform to make a normal distribution out of:
mins_to_first_comment, mins_to_100_comment, shares_per_sec and reactions_per_sec
Reads from `/data/postsWithFracReactions.csv`
Writes reciprocal-root transformed features to `/data/postsWithReciRoot.csv`
"""

import pandas as pd
import numpy as np
import operator
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
posts = pd.read_csv("../data/postsWithFracReactions.csv")
image_path = "figures/distributions_{:d}.png"
image_num = 1

""" ------------------------------------------- """
""" Minutes until a post gets its first comment """
""" ------------------------------------------- """
# Remove posts with negative mins to first comment - there are around 100 such posts
postsOrig = posts
print(postsOrig.shape)
posts = posts[posts["mins_to_first_comment"]>=0]
print(posts.shape)
fig = px.histogram(posts, x="mins_to_first_comment")
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1
# Of 18019 posts, check how many get 100 comments in less than 300 mins
posts[posts["mins_to_first_comment"]<300].shape
fig = px.histogram(posts[posts["mins_to_first_comment"]<300], x="mins_to_first_comment")
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1
# Log scale x-axis because it it long tailed
posts["log_mins_to_first_comment"] = np.log10(posts["mins_to_first_comment"])
fig = px.histogram(posts, x="log_mins_to_first_comment")
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1
# Reciprocal root transform x-axis because it it long tailed
minVal = posts[posts["mins_to_first_comment"]>0]["mins_to_first_comment"].min()
posts["reciroot_mins_to_first_comment"] = -1/((posts["mins_to_first_comment"]+minVal)**(1/6))
fig = px.histogram(posts, x="reciroot_mins_to_first_comment")
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1
print("Percentage of posts that get first comment in less than 10 mins: ")
print(posts[posts["mins_to_first_comment"]<10].shape[0]/posts.shape[0])
fig = px.histogram(posts[posts["mins_to_first_comment"]<10], x="mins_to_first_comment")
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1
print("Percentage of posts that get first comment in less than 3 mins: ")
print(posts[posts["mins_to_first_comment"]<3].shape[0]/posts.shape[0])

""" ------------------------------------------- """
""" Minutes until a post gets its 100th comment """
""" ------------------------------------------- """
# Remove posts with negative mins to 100th comment - there are around 100 such posts
postsOrig1 = posts
print(postsOrig1.shape)
posts = posts[(posts["mins_to_100_comment"]>0) & (posts["mins_to_100_comment"]<100000)]
print(posts.shape)
fig = px.histogram(posts, x="mins_to_100_comment", labels={'mins_to_100_comment':'Minutes until 100<sup>th</sup> comment'})
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1
print("Percentage of posts that get 100 comments in less than 8000 mins(~5.5 days): ")
print(posts[posts["mins_to_100_comment"]<4000].shape[0]/posts.shape[0])
fig = px.histogram(posts[posts["mins_to_100_comment"]<4000], x="mins_to_100_comment")
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1
print("Percentage of posts that get 100 comments in less than 4000 mins (~2.7 days): ")
print(posts[posts["mins_to_100_comment"]<4000].shape[0]/posts.shape[0])
# Reciprocal root transform x-axis because it it long tailed
posts["reciroot_mins_to_100_comment"] = -1/(posts["mins_to_100_comment"]**(1/1000))
fig = px.histogram(posts, x="reciroot_mins_to_100_comment", nbins=30, labels={'reciroot_mins_to_100_comment':'Reciprocal root of minutes until 100<sup>th</sup> comment'})
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1

""" ------------------------------"""
""" Day and time of post creation """
""" ----------------------------- """
# Fill in days
posts['day_name'] = "null"
posts.loc[posts.monday == 1, 'day_name'] = "monday"
posts.loc[posts.tuesday == 1, 'day_name'] = "tuesday"
posts.loc[posts.wednesday == 1, 'day_name'] = "wednesday"
posts.loc[posts.thursday == 1, 'day_name'] = "thursday"
posts.loc[posts.friday == 1, 'day_name'] = "friday"
posts.loc[posts.saturday == 1, 'day_name'] = "saturday"
posts.loc[posts.sunday == 1, 'day_name'] = "sunday"
# Fill in time of day
posts['time_segment'] = "null"
posts.loc[posts.pre_work == 1, 'time_segment'] = "pre_work"
posts.loc[posts.work == 1, 'time_segment'] = "work"
posts.loc[posts.post_work == 1, 'time_segment'] = "post_work"
posts.loc[posts.sleep == 1, 'time_segment'] = "sleep"
# Initialize subplots
fig = make_subplots(
    rows=1, cols=2
)
# Add trace
fig.add_trace(go.Histogram(x=posts["day_name"]), row=1, col=1)
fig.add_trace(go.Histogram(x=posts["time_segment"]), row=1, col=2)
# Update axes labels
fig.update_xaxes(title_text="day_name", row=1, col=1)
fig.update_xaxes(title_text="time_segment", row=1, col=2)
fig.update_yaxes(title_text="count", row=1, col=1)
fig.update_yaxes(title_text="count", row=1, col=2)
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1

""" -----------------"""
""" Number of shares """
""" -----------------"""
fig = px.histogram(posts, x="shares")
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1
print("Percentage of posts that get <= 400 shares")
print(posts[posts["shares"]<=400].shape[0]/posts.shape[0])
fig = px.histogram(posts[posts["shares"]<=400], x="shares")
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1
fig = px.histogram(posts[posts["shares"]>400], x="shares")
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1
fig = px.histogram(posts[posts["shares"]<=400], x="shares_per_sec")
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1
# Per second
print("Percentage of posts that get <= 0.002 shares_per_sec")
print(posts[posts["shares_per_sec"]<=0.002].shape[0]/posts.shape[0])
fig = px.histogram(posts[posts["shares_per_sec"]<=0.002], x="shares_per_sec")
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1
# Reciprocal root transform
minVal = posts[posts["shares_per_sec"]>0]["shares_per_sec"].min()
posts["reciroot_shares_per_sec"] = -1/((posts["shares_per_sec"]+minVal)**(1/200))
fig = px.histogram(posts, x="reciroot_shares_per_sec")
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1

""" -----------------------------------------"""
""" Total number of reactions (of all types) """
""" -----------------------------------------"""
fig = px.histogram(posts, x="total_reactions")
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1
print("Percentage of posts that get <= 2000 reactions")
print(posts[posts["total_reactions"]<=2000].shape[0]/posts.shape[0])
fig = px.histogram(posts[posts["total_reactions"]<=2000], x="total_reactions")
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1
fig = px.histogram(posts[posts["total_reactions"]<=2000], x="reactions_per_sec")
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1
print("Percentage of posts that get <= 0.01 reactions_per_sec")
print(posts[posts["reactions_per_sec"]<=0.01].shape[0]/posts.shape[0])
fig = px.histogram(posts[posts["reactions_per_sec"]<=0.01], x="reactions_per_sec")
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1
minVal = posts[posts["reactions_per_sec"]>0]["reactions_per_sec"].min()
posts["reciroot_reactions_per_sec"] = -1/((posts["reactions_per_sec"]+minVal)**(1/200))
fig = px.histogram(posts, x="reciroot_reactions_per_sec")
fig.show()
fig.write_image(image_path.format(image_num))
image_num += 1

# Write transformed features to file
posts.to_csv ('../data/postsWithReciRoot.csv', index = None, header=True) 