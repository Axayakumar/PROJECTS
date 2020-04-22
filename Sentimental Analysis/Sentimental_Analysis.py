#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import the libraries
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Twitter API credentials
consumer_key = "DdgFTEL6osrDhXeBNrTZT0QXu"
consumer_secret = "96muCKHsUX4rjxXBlGou5kDy9V5vWafRxhZTSBXUJrfftlbLcO"
access_key = "2904703940-n0biqVFiStlhr6K0ZHj1QACcUcXWYu9W4hU92lF"
access_secret = "qIm3Uwx0Bvk5hOlAk2SIScFklVrKwaG3MBiXeXZd0YP7C"

auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)
#Finding user of twitter accpunt credentials
print(api.me().name)


# In[3]:


serachterm = input('enter the Keyword/hashtag to search about: ')
noofcounts = int(input('enter how many tweets to analyse: '))


# In[4]:


#extract tweets from twitter user
#also we can use below query to get tweets, but as it was only extecting 100 tweets i used other query, 
#if i use this i have to use 'i.full_text' & 'tweet.full_text' instead of 'i.text' and 'tweet.text' 
#posts = api.search(q=serachterm,count=noofcounts,lang="en",tweet_mode='extended')

#also we can use below query to get tweets as per https://www.youtube.com/watch?v=eFdPGpny_hY
posts = tweepy.Cursor(api.search, q=serachterm, lang="en").items(noofcounts)
#also we can use below query to get tweets based on twitter username
#new_tweets = api.user_timeline(screen_name = screen_name,count=200)

'''#print the last 5 posts tweeted on COVID
print("show the last 5 posts: \n")
for i in posts[0:5]:
    print(i.full_text + '\n')'''   #--->it only works with posts = api.search


# In[5]:


#Create a dataframe DF with a column called tweets
#df_backup = pd.DataFrame([tweet.text for tweet in posts], columns=['Tweets'])
#Created backup copy because to avoid time consuming DF creation for more numbrt of tweets


# In[145]:


df = df_backup.copy()
df.head()


# In[ ]:


# Open/create a file to append data to
#csvFile = open(searchTerm+'_result.csv', 'a')
# Use csv writer
#csvWriter = csv.writer(csvFile)


# In[146]:


df.count()


# # Data Cleaning

# Dropping duplicates rows Let us see and delete duplicate rows if any. There are two types of duplicates-duplicates with same values for all columns(this duplication happens when same tweets are collected again by tweet-collector) and duplicates with the same text for tweets(This occurs when two or more users post the same tweet.)

# In[147]:


print(len(df.index))
serlis=df.duplicated().tolist()
print(serlis.count(True))
serlis=df.duplicated(['Tweets']).tolist()
print(serlis.count(True))


# rows which are duplicated in the column-Tweets and row-duplicates are a subset of Tweets-duplicates. So we drop all the duplicate rows.

# In[148]:


#Dropping duplicate rows in column Tweets
df=df.drop_duplicates(['Tweets'])


# In[149]:


#checking if duplicated are removed
print(len(df.index))
serlis=df.duplicated().tolist()
print(serlis.count(True))
serlis=df.duplicated(['Tweets']).tolist()
print(serlis.count(True))


# In[150]:


df.count()


# In[151]:


#remove punctuations
import string
string.punctuation
def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

df['Tweets'] = df['Tweets'].apply(lambda x: remove_punct(x))
df


# In[152]:


#Removing other things
#Ctreate function to clean text
def CleanText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) #Remove @ Mentiones
    text = re.sub("RT @[\W]*:",'', text) #Remove RT starting with
    text = re.sub("@[\W]*:",'', text) #Remove RT
    text = re.sub(r'^[^a]*', '', text) #To remove everything before a certain character
    text = re.sub(r'#', '', text)  # removing # tags
    text = re.sub(r'RT[\s]+','', text)  # removing Reetweets with white spaces
    text = re.sub(r'https?:\/\/\S+', '', text)  # removing hyper links #https?---> may or manot have s after http
    text = re.sub(r'https', '', text)  # removing https tags
    return text
df['Tweets'] = df['Tweets'].apply(CleanText)
df


# In[ ]:


#To save tweets in CSV
#df.to_csv('Tweets.CSV')


# In[153]:


df['Tweets'][50]


# In[304]:


'''def tokenization(text):
    text = re.split('\W+', text)
    return text

df['Tweet_tokenized'] = df['Tweets'].apply(lambda x: tokenization(x.lower()))
df.head()'''


# 1) Tokenization: the process of segmenting text into words, clauses or sentences (here we will separate out words and remove punctuation).
# 
# 2) Stemming: reducing related words to a common stem.
# 
# 3) Removal of stop words: removal of commonly used words unlikely to be useful for learning.

# # Tokanization
# We will now apply the word_tokenize to all records, making a new column in our imdb DataFrame. Each entry will be a list of words. Here we will also strip out non alphanumeric words/characters (such as numbers and punctuation) using .isalpha (you could use .isalnum if you wanted to keep in numbers as well).

# In[154]:


import nltk
def identify_tokens(row):
    Tweets = row['Tweets']
    tokens = nltk.word_tokenize(Tweets)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words

df['Tweets'] = df.apply(identify_tokens, axis=1)
df


# # Stemming
# Stemming reduces related words to a common stem. It is an optional process step, and it it is useful to test accuracy with and without stemming. Let’s look at an example.

# In[ ]:


'''#Exapmple for Stemming
from nltk.stem import PorterStemmer
stemming = PorterStemmer()

my_list = ['frightening', 'frightened', 'frightens']

# Using a Python list comprehension method to apply to all words in my_list

print ([stemming.stem(word) for word in my_list])


Out:
['frighten', 'frighten', 'frighten']'''


# In[155]:


from nltk.stem import PorterStemmer
stemming = PorterStemmer()
def stem_list(row):
    Tweets = row['Tweets']
    stemmed_list = [stemming.stem(word) for word in Tweets]
    return (stemmed_list)

df['Tweets'] = df.apply(stem_list, axis=1)
df


# # Removing stop words
# ‘Stop words’ are commonly used words that are unlikely to have any benefit in natural language processing. These includes words such as ‘a’, ‘the’, ‘is’.
# 
# As before we will define a function and apply it to our DataFrame.
# 
# We create a set of words that we will call ‘stops’ (using a set helps to speed up removing stop words).

# In[223]:


'''# Import stopwords with nltk.
from nltk.corpus import stopwords
stop = stopwords.words('english')

# Exclude stopwords with Python's list comprehension and pandas.DataFrame.apply.
df['Tweets'] = df['Tweets'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df'''


# In[156]:


from nltk.corpus import stopwords
stops = set(stopwords.words("english"))                  

def remove_stops(row):
    my_list = row['Tweets']
    meaningful_words = [w for w in my_list if not w in stops]
    return (meaningful_words)

df['Tweets'] = df.apply(remove_stops, axis=1)
df


# # Rejoin words
# Now we will rejoin our meaningful stemmed words into a single string.

# In[157]:


def rejoin_words(row):
    my_list = row['Tweets']
    joined_words = ( " ".join(my_list))
    return joined_words

df['Tweets'] = df.apply(rejoin_words, axis=1)
df


# In[186]:


'''#Example how to textblob works
from textblob import TextBlob
feedback1 = 'the food at reataurant was awesome'
feedback2 = 'the food at reataurant was vey good'
blob1 = TextBlob(feedback1)
blob2 = TextBlob(feedback2)
print(blob1.sentiment)
print(blob2.sentiment)'''


# In[158]:


from textblob import TextBlob

#Create function to get subjectivity --- Tells how subjective the text is
def getsubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

#Create function to get polarity --- tells how postive or negetive text is
def getpolarity(text):
    return TextBlob(text).sentiment.polarity

#Crte new two columns
df['Subjectivity'] = df['Tweets'].apply(getsubjectivity)
df['Polarity'] = df['Tweets'].apply(getpolarity)
df.head(10)


# In[159]:


#Plot the world cloud
allwords = ' '.join( [twts for twts in df['Tweets']] )
wordCloud = WordCloud(width = 500, height = 300, random_state = 21, max_font_size = 119).generate(allwords)

plt.imshow(wordCloud, interpolation = "bilinear")
plt.axis('off')
plt.show()


# In[160]:


#Create a function to compute negetive, postive & neutral analysis
def getanalysis(score):
    if score < 0:
        return 'Negetive'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'
    
df['Analysis'] = df['Polarity'].apply(getanalysis)

#Show the dataframe
df


# In[161]:


#Print all positive tweets
j=1
sortedDF = df.sort_values(by=['Polarity'])
for i in range(0, sortedDF.shape[0]):   #sortedDF.shape[0] --- Number of rows dataframe has
    if(sortedDF['Analysis'][i] == 'Positive'):
        print(str(j) + ') '+sortedDF['Tweets'][i])
        print()  #newline
        j = j+1


# In[162]:


#Print all positNegetiveive tweets
j=1
sortedDF = df.sort_values(by=['Polarity'])
for i in range(0, sortedDF.shape[0]):   #sortedDF.shape[0] --- Number of rows dataframe has
    if(sortedDF['Analysis'][i] == 'Negetive'):
        print(str(j) + ') '+sortedDF['Tweets'][i])
        print()  #newline
        j = j+1


# In[24]:


#Print all Neutral tweets
j=1
sortedDF = df.sort_values(by=['Polarity'])
for i in range(0, sortedDF.shape[0]):   #sortedDF.shape[0] --- Number of rows dataframe has
    if(sortedDF['Analysis'][i] == 'Neutral'):
        print(str(j) + ') '+sortedDF['Tweets'][i])
        print()  #newline
        j = j+1


# In[163]:


df


# In[164]:


#Plot the Polarity & Subjectivity
plt.figure(figsize=(8,6))
for i in range (0, df.shape[0]):
    plt.scatter(df['Polarity'][i], df['Subjectivity'][i], color='Blue')
    
plt.title('Sentimental Analysis')
plt.xlabel('polarity')
plt.ylabel('Subjectivity')
plt.show()          


# In[165]:


#Percentage of positive tweets
ptweets = df[df.Analysis == 'Positive']
ptweets = ptweets['Tweets']

print(round((ptweets.shape[0] / df.shape[0])*100, 1))


# In[166]:


#Percentage of Negetive tweets
ntweets = df[df.Analysis == 'Negetive']
ntweets = ntweets['Tweets']

print(round((ntweets.shape[0] / df.shape[0])*100, 1))


# In[167]:


#Percentage of Neutral tweets
neutrtweets = df[df.Analysis == 'Neutral']
neutrtweets = neutrtweets['Tweets']

print(round((neutrtweets.shape[0] / df.shape[0])*100, 1))


# In[168]:


print('total positive tweets are = ', ptweets.count())
print('total Negetive negetive are = ', ntweets.count())
print('total Neutra Neutral are = ',neutrtweets.count())
print('total Tweets are = ',df.shape[0])


# In[169]:


#Show the value counts

df['Analysis'].value_counts()

#plot and visualize the data
plt.title('Sentimental Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
df['Analysis'].value_counts().plot(kind='bar')
plt.show()


# In[170]:


#Plotting Py chart
#Create function to calculate percentage positive, negetive & Neytral values
def percentage(part, whole):
    return 100 * float(part)/float(whole)

''' Check functio working fine
abc = percentage(10, 100)
print(abc)'''

# ceating variables to store total positive, negetive & Neutral tweets & total tweets
poscount =  ptweets.count()
negcount = ntweets.count()
neutrcount = neutrtweets.count()
totcount = df.shape[0]

#Finding percentage positive, negetive & Neutral tweets
perposttweets = percentage(poscount, totcount)
pernegtweets = percentage(negcount, totcount)
perneutweets = percentage(neutrcount, totcount)
#Rounding off percentage values
perposttweets = round(perposttweets, 1)
pernegtweets = round(pernegtweets, 1)
perneutweets = round(perneutweets, 1)
print(perposttweets)
print(pernegtweets)
print(perneutweets)
labels = ['Positive ['+str(perposttweets)+'%]', 'Neutral ['+str(perneutweets)+'%]', 'Negetive ['+str(pernegtweets)+'%]']
sizes = [perposttweets, perneutweets, perneutweets]
colors = ['yellowgreen', 'gold', 'red']
patches, texts = plt.pie(sizes, colors=colors, startangle=90)
plt.legend(patches, labels, loc='best')
plt.title('how people are re reaching on '+serachterm+' by alalysing '+str(noofcounts)+' Tweets')
plt.axis('equal')
plt.tight_layout()
plt.show()


# In[171]:


df.tail()


# In[103]:


#Another exapmle to draw PiePlot
plot_size = plt.rcParams["figure.figsize"] 
print(plot_size[0]) 
print(plot_size[1])

plot_size[0] = 8
plot_size[1] = 6
plt.rcParams["figure.figsize"] = plot_size 
plt.title('how people are re reaching on '+serachterm+' by alalysing '+str(noofcounts)+' Tweets')
plt.axis('equal')

df.Analysis.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["red", "yellow", "green"])


# In[1]:


from plotly import graph_objs as go
fig = go.Figure(go.Funnelarea(
    text =df.Analysis,
    values = df.Tweets,
    title = {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"}
    ))
fig.show()


# In[ ]:





# In[95]:


'''#plot and visualize the data
plt.title('Sentimental Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
df['Analysis'].value_counts().plot(kind='box')
plt.show()'''


# In[174]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#OneHotEncoder --> To handle categorical values we use Onehot encoding refer -https://www.youtube.com/watch?v=xlFk1r6_a0g&list=PLlH6o4fAIji6FEsjFeo7gRgiwhPUkJ4ap&index=5
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[176]:


df_model = df.copy()
df_model


# In[179]:


labelencoder = LabelEncoder()
df_model['Analysis_model'] = labelencoder.fit_transform(df_model['Analysis'])
df_model


# In[ ]:





# In[198]:


df_model.head()


# In[233]:


onehotencoder = OneHotEncoder(categories = [4])
onehotencoder


# In[238]:


df_model['Analysis_Onehot'] = onehotencoder.fit_transform(df_model.Analysis_model.values.toarray()
df_model


# In[234]:


onehotencoder.fit_transform


# In[235]:


#df_model = df_model.drop(columns = 'Analysis_model_4')
df_model.count()


# In[ ]:


df_model.drop[5]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




