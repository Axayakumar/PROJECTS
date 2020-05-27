import tweepy
import numpy as np
import configurations
import pandas as pd
import time
import sys
from textblob import TextBlob
import re
import string
import collections
from collections import Counter



def generate_chart(tweetsDataframe):

  
   sentiment_pie = []
   retweet_table = []
    
  
   sentiment_pie.append(["Sentiment","Tweets"])
   sentiment_count = tweetsDataframe["sentiments_group"].value_counts()
   sentiment_count= sentiment_count.to_dict()
   for key,value in sentiment_count.items():
      temp = [key,value]
      sentiment_pie.append(temp)



   retweet_table.append(["Tweets"])
   df = tweetsDataframe[['translate']].drop_duplicates()[:6]
   for key in df['translate']:
      temp = [key]
      retweet_table.append(temp)
  

   new_list = []
   for item in tweetsDataframe['translate']:
   	new_item = [item]
   	new_list.append(new_item)

   text = ""    
   text_tweets = new_list
   length = len(text_tweets)

   for i in range(0, length):
    text = text_tweets[i][0] + " " + text

   lower_case = text.lower()
   cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
   tokenized_words = cleaned_text.split()

   stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
               "will", "just", "don", "should", "now"]

   final_words = [word for word in tokenized_words if word not in stop_words]
   emotion_list = []
   with open('emotions.txt', 'r') as file:
    	for line in file:
    		clear_line = line.replace('\n', '').replace(',', '').replace("'", '').strip()
    		word, emotion = clear_line.split(':')
    		if word in final_words:
    			emotion_list.append(emotion)

   w = Counter(emotion_list)
   c = dict(w)

   emotion_keys = []
   emotion_values = []

   for k,y in c.items():
    	emotion_keys.append(k)
    	emotion_values.append(y)

   emotion_table = []
   emotion_table.append(["Emotion","Count"])

   for key,value in zip(emotion_keys,emotion_values):
   	 temp = [key,value]
   	 emotion_table.append(temp)

   return (emotion_table[:4],sentiment_pie,retweet_table)

def AccessTwitter(search_string):
    
    key = configurations.consumer_key
    secret = configurations.consumer_secret
    access_token = configurations.access_token
    access_secret = configurations.access_secret
    
    auth = tweepy.OAuthHandler(consumer_key=key,consumer_secret=secret)
    auth.set_access_token(access_token, access_secret)
    
    api = tweepy.API(auth)
    tweet_list = []
    search_string = search_string + ' -filter:retweets -#tgif'
    for tweet in api.search(q=search_string,count=100,tweet_mode="extended",lang="en"):
        tweet_list.append(tweet)
        
    (word_freqs_js,max_freq,all_sentence,tweet_Data) = filter_twitter_data(tweet_list)
    
    (emotion_plot,sentiment_pie,retweet_table) = generate_chart(tweet_Data)
    
    

    return (emotion_plot,word_freqs_js,max_freq, sentiment_pie,retweet_table)


def filter_twitter_data(tweets):
    all_sentence = []
    id_list = [tweet.id for tweet in tweets]
    tweet_Data = pd.DataFrame(id_list,columns=['id'])
    tweet_Data["retweet_count"]= [tweet.retweet_count for tweet in tweets]
   


    Sentiments_list = []
    Sentiments_group = []
    Subjectivity_list = []
    Subjectivity_group = []
    tweet_text_list = []
    tweet_source = []
    tweet_translation= []
    tweet_location_list = []
    
    for tweet in tweets:
        raw_tweet_text = tweet.full_text
        message = TextBlob(tweet.full_text)
        location = tweet.author.location
        source = tweet.source
        tweet_source.append(source)
        message = str(message)
        new_message = ""

        for letter in range(0,len(message)):
            current_read =message[letter]
            if ord(current_read) > 126:
                continue
            else:
                new_message =new_message+current_read

        message = new_message
        tweet_translation.append(message)
        #message = fix_pattern(message)
        message = TextBlob(message)
        
        sentiment = message.sentiment.polarity
        if (sentiment > 0):
            Sentiments_group.append('positive')
        elif (sentiment < 0):
            Sentiments_group.append('negative')
        else:
            Sentiments_group.append('neutral')
            
        subjectivity = message.sentiment.subjectivity
        if (subjectivity > 0.4):
            Subjectivity_group.append('subjective')
        else:
            Subjectivity_group.append('objective')
            
        Sentiments_list.append(sentiment)
        Subjectivity_list.append(subjectivity)
        tweet_text_list.append(raw_tweet_text)
        tweet_location_list.append(location)
        
        
    tweet_Data["sentiments"] = Sentiments_list
    tweet_Data["sentiments_group"] = Sentiments_group
    tweet_Data["subjectivity"]= Subjectivity_list
    tweet_Data["subjectivity_group"] = Subjectivity_group
    tweet_Data["location"] = tweet_location_list
    tweet_Data["text"] = tweet_text_list
    tweet_Data["translate"] = tweet_translation

  
    for a in tweet_Data['translate']:
        all_sentence.append(a)
    str1 = " "
    b = str1.join(all_sentence)
    b = ' '.join(re.sub("(#[A-Za-z0-9]+)|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",b).split())
    all_sentence = b.split()
    
    text = b

    stripped_text = []
    stripped_text = [word for word in text.split() if word.isalpha() and word.lower() not in open("stopwords", "r").read() and len(word) >= 2]
    word_freqs = Counter(stripped_text)
    word_freqs_cloud = dict(word_freqs)
    add = sorted(word_freqs_cloud.items(), key=lambda x: x[1], reverse=True)
    word_freqs = dict(add[:60])

    word_freqs_js = []
    
    for key,value in word_freqs.items():
        temp = {"text": key, "size": value}
        word_freqs_js.append(temp) 

    max_freq = max(word_freqs.values())
   


    return (word_freqs_js,max_freq,all_sentence,tweet_Data)
