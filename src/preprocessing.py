import pandas as pd
import re
import emoji
from nltk.tokenize import TweetTokenizer
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
import sys
from unidecode import unidecode

# Tokenizer object
tknzr = TweetTokenizer()

# Lemmatizer object
lemmatizer = WordNetLemmatizer()

# Create stopwords set
notstopwords = set(('not', 'no'))
stopwords = set(stopwords.words('english')) - notstopwords


def data_preprocessing(path_tweets):
	tweets = pd.read_csv(path_tweets, encoding='utf-8',sep=',')
	tweets['text'] = tweets['text'].apply(lambda x: standardization(x))
	tweets['sentiment'] = tweets['airline_sentiment'].apply(lambda x:0 if x=='negative' else (1 if x=='neutral' else 2))
	return tweets['text'], tweets['sentiment']


# Launch preprocessing into an entire file.csv
def data_preprocessing(path_tweets, corpora):
	# create a dataframe
	data = pd.read_csv(path_tweets, encoding='utf-8', sep='\t', names=['id','class','text'])
	# change values 0->negative, 1->neutral, 2->positive
	if corpora=='train':
		data['class'] = data['class'].apply(lambda x: 0 if x=='negative' else (1 if x=='neutral' else 2 ))
	# apply standartization
	data['text'] = data['text'].apply(lambda x: standardization(x))
	return data['text'], data['class']


def data_preprocessing_test(path_tweets):
	data = pd.read_csv(path_tweets, encoding='utf-8',sep='\t')
	data['text'] = data['Tweet'].apply(lambda x: standardization(x))
	return data['text']


def standardization(tweet):
	tweet = re.sub(r"\\u2019", "'", tweet)
	tweet = re.sub(r"\\u002c", "'", tweet)
	tweet=' '.join(emoji.str2emoji(unidecode(tweet).lower().split()))
	tweet = re.sub(r"(http|https)?:\/\/[a-zA-Z0-9\.-]+\.[a-zA-Z]{2,4}(/\S*)?", " ", tweet)
	tweet = re.sub(r"\'ve", " have", tweet)
	tweet = re.sub(r" can\'t", " cannot", tweet)
	tweet = re.sub(r"n\'t", " not", tweet)
	tweet = re.sub(r"\'re", " are", tweet)
	tweet = re.sub(r"\'d", " would", tweet)
	tweet = re.sub(r"\'ll", " will", tweet)
	tweet = re.sub(r"\'s", "", tweet)
	tweet = re.sub(r"\'n", "", tweet)
	tweet = re.sub(r"\'m", " am", tweet)
	tweet = re.sub(r"@\w+", r' ',tweet)
	tweet = re.sub(r"#\w+", r' ',tweet)
	tweet = re.sub(r" [0-9]+ "," ",tweet)
	tweet = [lemmatizer.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v']  else lemmatizer.lemmatize(i) for i,j in pos_tag(tknzr.tokenize(tweet))]
	tweet = [ i for i in tweet if (i not in stopwords) and (i not in punctuation ) ]
	tweet = ' '.join(tweet)
	return tweet

# Quick tests
"""
print(standardization("http://google.com @avalard Have a good trips. See you tomorrow at the Jurys Inn? :( :‑) :‑) o_0"))
print(standardization("I just have to remember to go online tomorrow and watch Grey\u2019s Anatomy \u002c Scandal \u002c &\u2019 Vampire Diaries :‑) ."))
"""

"""
file_path = "data/task_A/data_train_3.csv"
t, c = data_preprocessing(file_path, 'train')

MAX_SEQUENCE_LENGTH = 0
for i in range(len(t)):
	print (i, "\t", t[i])
	if len(t[i]) > MAX_SEQUENCE_LENGTH:
		MAX_SEQUENCE_LENGTH = len(t[i])
print(MAX_SEQUENCE_LENGTH)
"""
