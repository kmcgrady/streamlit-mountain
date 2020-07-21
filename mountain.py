import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timezone, date, timedelta
import os
import re
import tweepy
import csv
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.neural_network import MLPClassifier
from dotenv import load_dotenv

from components.tweet import embed_tweet

load_dotenv()

'''
# Predicting Mount Rainier's Visibility

Always described as the “Rainy” City. Seattle gets its bad rap for its always dreary weather. While there are certainly a lot of cloudy days, Seattle is known for its beautiful landscapes. They certainly call it the Emerald City for some reason! Seattle is surrounded by two mountain ranges: the Olympic Mountains and the Cascade Range. The tallest mountain in Washington is Mount Rainier, which sits on the Cascade Range. Its majestic beauty shines like a beacon in the mountain range. Figure 1 shows the mountain up close.

Seattle is a little distance away, but on good days, we can see the mountain. Figure 2 is a picture of the mountain from our Space Needle. On very cloudy days, however, the mountain will not be visible. While we can get some accuracy in determining the likelihood that we will see the mountain given the weather forecast, my goal is to predict the chances of the mountain being visible on a given day.

## The Dataset

To first determine the state of seeing Mount Rainier, I relied on a twitter account, [@IsMtRainierOut](https://twitter.com/IsMtRainierOut). This account takes pictures from the Space Needle and assesses if the mountain is visible or not. The observations are made daily, sometimes multiple times a day. The “cloudiness” of the sky certainly is a strong factor in determining whether Seattle will be able to see the mountain, but there is more nuance in the way the clouds need to be formed. We can see the mountain if the clouds are higher than the mountain, and we can see the mountain if the clouds are lower than the mountain and the Space Needle. In this case, I took climate data from the (National Oceanic and Atmospheric Administration) NOAA, which is calculated at Boeing Field, a few miles south of the Space Needle and in front of the mountain.

Try it out, select a date, and we will show the tweet for that day.
'''


FIRST_TWEET = date(2014, 4, 16)
MT_RAINIER_CHECK_USER = "IsMtRainierOut"
mountain_day_check = st.date_input(
    label="Date to check whether Mt Rainier is out",
    value=datetime.today(),
    min_value=FIRST_TWEET,
    max_value=datetime.today())

# Twitter API credentials
consumer_key = os.getenv("TWITTER_CONSUMER_KEY")
consumer_secret = os.getenv("TWITTER_CONSUMER_SECRET")
access_key = os.getenv("TWITTER_ACCESS_KEY")
access_secret = os.getenv("TWITTER_ACCESS_SECRET")

@st.cache(allow_output_mutation=True)
def get_all_tweets(to_date):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    all_tweets = []

    # make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name = MT_RAINIER_CHECK_USER, count = 200)

    # save most recent tweets
    all_tweets.extend(new_tweets)

    # save the id of the oldest tweet less one
    oldest = all_tweets[-1].id - 1

    # keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print("getting tweets before %s" % (oldest))

        # all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(
            screen_name = MT_RAINIER_CHECK_USER,
            count = 200,
            max_id=oldest)

        # save most recent tweets
        all_tweets.extend(new_tweets)

        # update the id of the oldest tweet less one
        oldest = all_tweets[-1].id - 1

        print("...%s tweets downloaded so far" % (len(all_tweets)))
    
    return all_tweets

def date_to_str(d):
    return date.strftime(d, "%Y/%m/%d")

tweets = get_all_tweets(date_to_str(datetime.today()))

def tweet_date_equal(t):
    t_date = date_to_str(t.created_at.date())
    m_date = date_to_str(mountain_day_check)

    return t_date == m_date

tweets_that_day = [t for t in tweets if tweet_date_equal(t)]

if len(tweets_that_day) == 0:
    st.info('There were no tweets that day')
else:
    st.write('There were {} tweet{} found'.format(len(tweets_that_day), 's' if len(tweets_that_day) > 1 else ''))
    for tweet in tweets_that_day:
        embed_tweet(tweet.id)

def nth(lst, n):
  return list(map(lambda a: a[n], lst))

PROBLEM = 'rainier'

def to_minute(dt):
  regex = '^(\d\d\d\d)-(\d\d)-(\d\d) (\d\d):(\d\d)$'
  m = re.search(regex, dt)
  time = {}
  
  headers = ['year', 'month', 'day', 'hour', 'minute']
  for j, header in enumerate(headers):
    time[header] = m.group(j + 1)
  
  d = datetime(int(time['year']), int(time['month']), int(time['day']), hour=int(time['hour']), minute=int(time['minute']), tzinfo=timezone.utc)
  d31 = datetime(int(time['year']), 12, 31, hour=11, minute=59, second=59, tzinfo=timezone.utc)
  january1st = datetime(int(time['year']), 1, 1, tzinfo=timezone.utc)
  timesince = d - january1st
  total = d31 - january1st
  
  return 100 * float(timesince.total_seconds()) / total.total_seconds()

def obtain_data(filename):
  features = []
  classifications = []

  feature_names = []

  with open(filename, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for i, row in enumerate(reader):
      feature = [row[5]] + [row[8]] + row[10:18] + [row[20]]
      if i == 0:
        feature_names = feature
        continue
    
      feature[0] = to_minute(feature[0])
      feature = list(map(float, feature))
      features.append(feature)
    
      if row[91] == 'True':
        classifications.append(1)
      else:
        classifications.append(0)
  
  return feature_names, features, classifications
  

names, features, classifications = obtain_data('datasets/climate_tweet_data_train.csv')
test_names, test_features, test_classifications = obtain_data('datasets/climate_tweet_data_test.csv')

def get_classifier(j):
    return MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(j), random_state=1, max_iter=2000)

@st.cache(suppress_st_warning=True)
def tree_learn(j):
    clf = get_classifier(j)
    return cross_val_score(clf, features, classifications, cv=5)
    

BEST_CHOICE = 'Use Best'
SELECT_CHOICE = 'Select Number'

criterion = st.selectbox(
    'Select Neural Network Classification',
    (BEST_CHOICE, SELECT_CHOICE)
)


if criterion == SELECT_CHOICE:
    hidden_layers = st.slider(
        'Select number of neurons in middle layer',
        2, len(names), step=1
    )
else:
    depth_scores = []
    neurons = range(2, len(names) + 1)
    for j in neurons:
        scores = tree_learn(j)
        depth_scores.append((1, (j), scores.mean()))
    
    num_layers, layer, mean = sorted(depth_scores, key=lambda a: a[-1])[-1]

    st.write("Best is to use {} neurons".format(str(layer)))

    hidden_layers = layer

clf = get_classifier(hidden_layers)

train_sizes, train_scores, valid_scores = learning_curve(clf, features, classifications, train_sizes=np.linspace(0.1, 1.0, 5), cv=5)
st.line_chart(pd.DataFrame(np.append([np.mean(train_scores, axis=1)], [np.mean(valid_scores, axis=1)], axis=0).transpose(), index=train_sizes, columns=['Training Score', 'Testing Score']))

clf.fit(features, classifications)

predictions = clf.predict(test_features)
errors = np.where((np.array(test_classifications) - predictions) != 0)[0]

st.write('### Results')

st.write('**Number of Predictions:** ' + str(len(predictions)))
st.write('**Number of Errors:** ' + str(len(errors)))
st.write('**Accuracy:** ' + str(1. - (float(len(errors)) / len(test_classifications))))
