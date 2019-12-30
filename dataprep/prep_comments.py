"""Derives additional features for each comment in /facebook-news/fb_news_comments_1000K_hashed.csv 

Joins /facebook-news/fb_news_comments_1000K_hashed.csv and /facebook-news/fb_news_posts_20K.csv
Derives sentiment and time since post creation, for each comment. 
Writes result to ../data/commentsProcessed.csv
"""

import pandas as pd
from datetime import datetime 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sid_obj = SentimentIntensityAnalyzer() 
posts = pd.read_csv("facebook-news/fb_news_posts_20K.csv")
comments = pd.read_csv("facebook-news/fb_news_comments_1000K_hashed.csv")
# Join data from posts table
commentsWithPost = posts.merge(comments, on='post_id')
commentsWithPost = commentsWithPost[['post_id', 'created_time_x', 'created_time_y', 'message_y']]
for index,row in commentsWithPost.iterrows():
    tPost = datetime.strptime(row['created_time_x'].split("+")[0], '%Y-%m-%dT%H:%M:%S')
    tComment = datetime.strptime(row['created_time_y'].split("+")[0],'%Y-%m-%dT%H:%M:%S')
    commentsWithPost.loc[index,'hrs_after_post'] = ((tComment-tPost).total_seconds())/ 3600
# Rename columns
commentsWithPost = commentsWithPost.rename(columns={"created_time_x": "post_creation_time", "created_time_y": "comment_creation_time", "message_y": "comment_text"})
# Add time feature
for index,row in commentsWithPost.iterrows():
    commentsWithPost.loc[index,'mins_after_post'] = row["hrs_after_post"]*60
# Add sentiment features
for index,row in commentsWithPost.iterrows():
    if(isinstance(row["comment_text"],str)):
        data = sid_obj.polarity_scores(row["comment_text"])
        commentsWithPost.loc[index,'vader_sentiment'] = data["compound"]
        commentsWithPost.loc[index,'vader_pos'] = data["pos"]
        commentsWithPost.loc[index,'vader_neg'] = data["neg"]
        commentsWithPost.loc[index,'vader_neu'] = data["neu"]
    else:
        commentsWithPost.loc[index,'vader_sentiment'] = 0
        commentsWithPost.loc[index,'vader_pos'] = 0
        commentsWithPost.loc[index,'vader_neg'] = 0
        commentsWithPost.loc[index,'vader_neu'] = 0
# Write to file
comments.to_csv ('../data/commentsProcessed.csv', index = None, header=True)