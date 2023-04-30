import pandas as pd
import matplotlib.pyplot as plt
import re
#import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding

amazondata = []
with open("amazonreviewssmall.txt") as f: 
    for line in f: 
        line = line.replace("__label__2 ", "positive\t")
        line = line.replace("__label__1 ", "negative\t")
        amazondata.append(line.split("\t"))
        # re.sub(r"__label_2 ", "positive\t", line)
        # re.sub(r"__label__1", "negative\t", line)
        #break
        
#print(amazondata)
amazon_review_df = pd.DataFrame(amazondata)
amazon_review_df.columns = ["sentiment", 'text']
#df.head
amazon_review_df = amazon_review_df.sample(13864, random_state=1)

amazon_review_df.head()
amazon_review_df['sentiment'].value_counts()
amazon_sentiment_label = amazon_review_df.sentiment.factorize()

amazon_review = amazon_review_df.text.values
tokenizer = Tokenizer(num_words=50000)
tokenizer.fit_on_texts(amazon_review)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(amazon_review)
amzn_padded_sequence = pad_sequences(encoded_docs, maxlen=200)

#process movie data
moviedata = []
with open("movies.data") as f: 
    for line in f: 
        moviedata.append(line.split("\t"))
        # re.sub(r"__label_2 ", "positive\t", line)
        # re.sub(r"__label__1", "negative\t", line)
        #break
        
#print(amazondata)
movie_review_df = pd.DataFrame(moviedata)
movie_review_df.columns = ["sentiment", 'text']
#df.head
movie_review_df = movie_review_df.sample(13864, random_state=1)

movie_review_df.head()
movie_review_df['sentiment'].value_counts()
movie_sentiment_label = movie_review_df.sentiment.factorize()
movie_review = movie_review_df.text.values
tokenizer = Tokenizer(num_words=50000)
tokenizer.fit_on_texts(movie_review)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(movie_review)
mov_padded_sequence = pad_sequences(encoded_docs, maxlen=200)

#process twitter data
df = pd.read_csv("Tweets.csv")
df.head()
df.columns
tweet_df = df[['text','airline_sentiment']]
print(tweet_df.shape)
tweet_df.head(5)
tweet_df = tweet_df[tweet_df['airline_sentiment'] != 'neutral']
print(tweet_df.shape)
tweet_df.head(5)
tweet_df["airline_sentiment"].value_counts()
sentiment_label = tweet_df.airline_sentiment.factorize()
sentiment_label

tweet = tweet_df.text.values
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(tweet)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(tweet)
padded_sequence = pad_sequences(encoded_docs, maxlen=200)

#process reddit data
df = pd.read_csv("Reddit_Data.csv")
df.head()
df.columns
reddit_df = df[['clean_comment','category']]
print(reddit_df.shape)
reddit_df.head(5)
reddit_df = reddit_df[reddit_df['category'] != 0]
print(reddit_df.shape)
reddit_df.head(5)
reddit_df["category"].value_counts()
reddit_sentiment_label = reddit_df.category.factorize()
reddit_sentiment_label

reddit_df["category"].replace(to_replace=1, value="positive", inplace=True)
reddit_df["category"].replace(to_replace=-1, value="negative", inplace=True)


reddit = reddit_df.clean_comment.values
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(reddit)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(reddit)
reddit_padded_sequence = pad_sequences(encoded_docs, maxlen=200)


embedding_vector_length = 32
model_twitter = Sequential() 
model_twitter.add(Embedding(vocab_size, embedding_vector_length, input_length=200) )
model_twitter.add(SpatialDropout1D(0.25))
model_twitter.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model_twitter.add(Dropout(0.2))
model_twitter.add(Dense(1, activation='sigmoid')) 
model_twitter.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])  
print(model_twitter.summary()) 

model_amazon = Sequential() 
model_amazon.add(Embedding(vocab_size, embedding_vector_length, input_length=200) )
model_amazon.add(SpatialDropout1D(0.25))
model_amazon.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model_amazon.add(Dropout(0.2))
model_amazon.add(Dense(1, activation='sigmoid')) 
model_amazon.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])  
print(model_amazon.summary()) 

model_movie = Sequential() 
model_movie.add(Embedding(vocab_size, embedding_vector_length, input_length=200) )
model_movie.add(SpatialDropout1D(0.25))
model_movie.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model_movie.add(Dropout(0.2))
model_movie.add(Dense(1, activation='sigmoid')) 
model_movie.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])  
print(model_movie.summary()) 

model_reddit = Sequential() 
model_reddit.add(Embedding(vocab_size, embedding_vector_length, input_length=200) )
model_reddit.add(SpatialDropout1D(0.25))
model_reddit.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model_reddit.add(Dropout(0.2))
model_reddit.add(Dense(1, activation='sigmoid')) 
model_reddit.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])  
print(model_reddit.summary()) 

twitter = model_twitter.fit(padded_sequence, sentiment_label[0],validation_split=0.2, epochs=5, batch_size=32) #twitter
amazon = model_amazon.fit(amzn_padded_sequence, amazon_sentiment_label[0],validation_split=0.2, epochs=5, batch_size=32) #amazon
movie = model_movie.fit(mov_padded_sequence,movie_sentiment_label[0],validation_split=0.2, epochs=5, batch_size=32) #movie
reddit = model_reddit.fit(reddit_padded_sequence,reddit_sentiment_label[0],validation_split=0.2, epochs=5, batch_size=32) #reddit


plt.plot(twitter.history['accuracy'], label='acc')
plt.plot(twitter.history['val_accuracy'], label='val_acc')
plt.title("Twitter Data Accuracy")
plt.legend()
plt.show()
plt.plot(twitter.history['loss'], label='loss')
plt.plot(twitter.history['val_loss'], label='val_loss')
plt.title("Twitter Data Loss")
plt.legend()
plt.show()


plt.plot(amazon.history['accuracy'], label='acc')
plt.plot(amazon.history['val_accuracy'], label='val_acc')
plt.title("Amazon Data Accuracy")
plt.legend()
plt.show()
plt.plot(amazon.history['loss'], label='loss')
plt.plot(amazon.history['val_loss'], label='val_loss')
plt.title("Amazon Data Loss")
plt.legend()
plt.show()

plt.plot(movie.history['accuracy'], label='acc')
plt.plot(movie.history['val_accuracy'], label='val_acc')
plt.title("Movie Data Accuracy")
plt.legend()
plt.show()
plt.plot(movie.history['loss'], label='loss')
plt.plot(movie.history['val_loss'], label='val_loss')
plt.title("Movie Data Loss")
plt.legend()
plt.show()


plt.plot(reddit.history['accuracy'], label='acc')
plt.plot(reddit.history['val_accuracy'], label='val_acc')
plt.title("Reddit Data Accuracy")
plt.legend()
plt.show()
plt.plot(reddit.history['loss'], label='loss')
plt.plot(reddit.history['val_loss'], label='val_loss')
plt.title("Reddit Data Loss")
plt.legend()
plt.show()


def predict_sentiment_amazon(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(model_amazon.predict(tw).round().item())
    print("Predicted label: ", amazon_sentiment_label[1][prediction])
    
def predict_sentiment_twitter(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(model_twitter.predict(tw).round().item())
    print("Predicted label: ", sentiment_label[1][prediction])

def predict_sentiment_movie(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(model_movie.predict(tw).round().item())
    print("Predicted label: ", movie_sentiment_label[1][prediction])

def predict_sentiment_reddit(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(model_reddit.predict(tw).round().item())
    print("Predicted label: ", reddit_sentiment_label[1][prediction])


test_sentence1 = "Still waiting on bags from flight 1613/2440 yesterday  First Class passenger not happy with your service."
predict_sentiment_twitter(test_sentence1)
predict_sentiment_reddit(test_sentence1)
predict_sentiment_movie(test_sentence1)
predict_sentiment_amazon(test_sentence1)