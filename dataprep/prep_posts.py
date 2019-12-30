"""Derives additional features for each post in /facebook-news/fb_news_posts_20K.csv 

Joins data with ../data/commentsProcessed.csv and uses VADER to derive the following features:

Day of the week the post was created (7 boolean fields and one int field - 0 to 6 for Monday to Sunday)
Average number of shares per second since post creation
Sentiment (-1 to 1)
Subjectivity (0 to 1)
Total number of reactions
Average number of reactions per second since post creation
Time of the day the post was created (4 boolean fields corresponding to the bins: 1 am to 7 am, 7 am to 10 am, 10 am to 5 pm, 5 pm to 1 am).
Minutes between post creation and first comment
Minutes between post creation and 100th comment

Writes them to ../data/postsProcessed.csv
"""

import pandas as pd
from datetime import datetime
import json
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sid_obj = SentimentIntensityAnalyzer() 
posts = pd.read_csv("facebook-news/fb_news_posts_20K.csv")
# Determine the day of the week the post was created and the average number of shares per second
for index,row in posts.iterrows():
    cTime = datetime.strptime(row['created_time'].split("+")[0], '%Y-%m-%dT%H:%M:%S')
    sTime = datetime.strptime(row['scrape_time'].split(".")[0], '%Y-%m-%d %H:%M:%S')
    diffSecs = (sTime-cTime).total_seconds() + 12*60*60
    posts.loc[index,'hrs_since_creation'] = diffSecs/3600
    posts.loc[index,'shares_per_sec'] = row['shares']/diffSecs
    day = cTime.weekday()
    posts.loc[index,'weekday'] = day
    posts.loc[index,'monday'] = 0
    posts.loc[index,'tuesday'] = 0
    posts.loc[index,'wednesday'] = 0
    posts.loc[index,'thursday'] = 0
    posts.loc[index,'friday'] = 0
    posts.loc[index,'saturday'] = 0
    posts.loc[index,'sunday'] = 0
    if day == 0:
        posts.loc[index,'monday'] = 1
    elif day == 1:
        posts.loc[index,'tuesday'] = 1
    elif day == 2:
        posts.loc[index,'wednesday'] = 1
    elif day == 3:
        posts.loc[index,'thursday'] = 1
    elif day == 4:
        posts.loc[index,'friday'] = 1
    elif day == 5:
        posts.loc[index,'saturday'] = 1
    elif day == 6:
        posts.loc[index,'sunday'] = 1
# Determine the total number of reactions and average number of reactions per second
for index,row in posts.iterrows():
    total = row["react_angry"]+row["react_haha"]+row["react_like"]+row["react_love"]+row["react_sad"]+row["react_wow"]
    posts.loc[index,'total_reactions'] = total
    posts.loc[index,'reactions_per_sec'] =  total/(row["hrs_since_creation"]*60*60)
# Determine the time of the day a post was created
t1 = datetime.strptime("01:00:00","%H:%M:%S").time()
t2 = datetime.strptime("07:00:00","%H:%M:%S").time()
t3 = datetime.strptime("10:00:00","%H:%M:%S").time()
t4 = datetime.strptime("17:00:00","%H:%M:%S").time()
for index,row in posts.iterrows():
    cTime = datetime.strptime(row['created_time'].split("+")[0], '%Y-%m-%dT%H:%M:%S').time()
    posts.loc[index,'pre_work'] = 0
    posts.loc[index,'work'] = 0
    posts.loc[index,'post_work'] = 0
    posts.loc[index,'sleep'] = 0
    if cTime >= t1 and cTime < t2:
        posts.loc[index,'sleep'] = 1
    elif cTime >= t2 and cTime < t3:
        posts.loc[index,'pre_work'] = 1
    elif cTime >= t3 and cTime < t4:
        posts.loc[index,'work'] = 1
    else:
        posts.loc[index,'post_work'] = 1
# Join with data from the prepared comments table
comments = pd.read_csv("../data/commentsProcessed.csv")
commentsGrouped = comments.groupby(['post_id'])
for index,row in posts.iterrows():
    if row["post_id"] in commentsGrouped.groups:
        commentGroup = commentsGrouped.get_group(row["post_id"])
        posts.loc[index,'mins_to_first_comment'] = commentGroup['mins_after_post'].min()
        posts.loc[index,'mins_to_100_comment'] = commentGroup['mins_after_post'].max()
# Remove empty posts and posts with no comments
posts = posts.dropna(subset=['mins_to_first_comment', 'message'])
# Determine post sentiment
for index,row in posts.iterrows():
    if(isinstance(row["message"],str)):
        data = sid_obj.polarity_scores(row["message"])
        posts.loc[index,'vader_sentiment'] = data["compound"]
        posts.loc[index,'vader_pos'] = data["pos"]
        posts.loc[index,'vader_neg'] = data["neg"]
        posts.loc[index,'vader_neu'] = data["neu"]
# Write to file
posts.to_csv ('../data/postsProcessed.csv', index = None, header=True) 