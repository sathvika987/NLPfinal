import pandas as pd
import matplotlib.pyplot as plt
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding

amazondata = []
with open("Amazon_Data.txt") as f: 
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
amazon_review_df = amazon_review_df.sample(frac=1, random_state=1)
# test data
test_amazon_review_df = amazon_review_df[10000:15000]
# train and validation data
amazon_review_df = amazon_review_df[:10000]

amazon_review_df.head()
amazon_review_df['sentiment'].value_counts()
amazon_sentiment_label = amazon_review_df.sentiment.factorize()


amazon_review = amazon_review_df.text.values
amzntokenizer = Tokenizer(num_words=5000)
amzntokenizer.fit_on_texts(amazon_review)
amznvocab_size = len(amzntokenizer.word_index) + 1
amznencoded_docs = amzntokenizer.texts_to_sequences(amazon_review)
amzn_padded_sequence = pad_sequences(amznencoded_docs, maxlen=200)

#process movie data
moviedata = []
with open("Rateitall_Data.txt") as f: 
    for line in f: 
        moviedata.append(line.split("\t"))
        # re.sub(r"__label_2 ", "positive\t", line)
        # re.sub(r"__label__1", "negative\t", line)
        #break
        
#print(amazondata)
movie_review_df = pd.DataFrame(moviedata)
movie_review_df.columns = ["sentiment", 'text']
#df.head
movie_review_df = movie_review_df.sample(frac=1, random_state=1)

test_movie_df = movie_review_df[10000:]
movie_df = movie_review_df[:10000]

movie_review_df.head()
movie_review_df['sentiment'].value_counts()
movie_sentiment_label = movie_review_df.sentiment.factorize()
movie_review = movie_review_df.text.values
movtokenizer = Tokenizer(num_words=5000)
movtokenizer.fit_on_texts(movie_review)
movvocab_size = len(movtokenizer.word_index) + 1
movencoded_docs = movtokenizer.texts_to_sequences(movie_review)
mov_padded_sequence = pad_sequences(movencoded_docs, maxlen=200)

#process twitter data
df = pd.read_csv("Tweet_Data.csv")
df.head()
df.columns
tweet_df = df[['text','airline_sentiment']]
print(tweet_df.shape)
tweet_df.head(5)
tweet_df = tweet_df[tweet_df['airline_sentiment'] != 'neutral']
print(tweet_df.shape)
tweet_df.head(5)
tweet_df.columns = ("text", "sentiment")

tweet_df = tweet_df.sample(frac=1, random_state=1)
test_tweet_df = tweet_df[10000:]
tweet_df = tweet_df[:10000]

tweet_df["sentiment"].value_counts()
twsentiment_label = tweet_df.sentiment.factorize()

tweet = tweet_df.text.values
twtokenizer = Tokenizer(num_words=5000)
twtokenizer.fit_on_texts(tweet)
twvocab_size = len(twtokenizer.word_index) + 1
twencoded_docs = twtokenizer.texts_to_sequences(tweet)
twpadded_sequence = pad_sequences(twencoded_docs, maxlen=200)

#process reddit data
df = pd.read_csv("Reddit_Data.csv")
df.head()
df.columns
reddit_df = df[['clean_comment','category']]
reddit_df.columns = ("text", "sentiment")

print(reddit_df.shape)
reddit_df.head(5)
reddit_df = reddit_df[reddit_df['sentiment'] != 0]
print(reddit_df.shape)
reddit_df["sentiment"].replace(to_replace=1, value="positive", inplace=True)
reddit_df["sentiment"].replace(to_replace=-1, value="negative", inplace=True)
reddit_df.head(5)
reddit_df["sentiment"].value_counts()
reddit_sentiment_label = reddit_df.sentiment.factorize()
reddit_sentiment_label
reddit_df = reddit_df.sample(frac=1, random_state=1)
test_reddit_df = reddit_df[10000:15000]
reddit_df = reddit_df[:10000]

reddit = reddit_df.text.values
rdttokenizer = Tokenizer(num_words=5000)
rdttokenizer.fit_on_texts(reddit)
rdtvocab_size = len(rdttokenizer.word_index) + 1
rdtencoded_docs = rdttokenizer.texts_to_sequences(reddit)
reddit_padded_sequence = pad_sequences(rdtencoded_docs, maxlen=200)


embedding_vector_length = 32
model_twitter = Sequential() 
model_twitter.add(Embedding(twvocab_size, embedding_vector_length, input_length=200) )
model_twitter.add(SpatialDropout1D(0.25))
model_twitter.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model_twitter.add(Dropout(0.2))
model_twitter.add(Dense(1, activation='sigmoid')) 
model_twitter.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])  
print(model_twitter.summary()) 

model_amazon = Sequential() 
model_amazon.add(Embedding(amznvocab_size, embedding_vector_length, input_length=200) )
model_amazon.add(SpatialDropout1D(0.25))
model_amazon.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model_amazon.add(Dropout(0.2))
model_amazon.add(Dense(1, activation='sigmoid')) 
model_amazon.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])  
print(model_amazon.summary()) 

model_movie = Sequential() 
model_movie.add(Embedding(movvocab_size, embedding_vector_length, input_length=200) )
model_movie.add(SpatialDropout1D(0.25))
model_movie.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model_movie.add(Dropout(0.2))
model_movie.add(Dense(1, activation='sigmoid')) 
model_movie.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])  
print(model_movie.summary()) 

model_reddit = Sequential() 
model_reddit.add(Embedding(rdtvocab_size, embedding_vector_length, input_length=200) )
model_reddit.add(SpatialDropout1D(0.25))
model_reddit.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model_reddit.add(Dropout(0.2))
model_reddit.add(Dense(1, activation='sigmoid')) 
model_reddit.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])  
print(model_reddit.summary()) 

with tf.device('/cpu:0'):
    twitter = model_twitter.fit(twpadded_sequence, twsentiment_label[0],validation_split=0.2, epochs=5, batch_size=32) #twitter
    amazon = model_amazon.fit(amzn_padded_sequence, amazon_sentiment_label[0],validation_split=0.2, epochs=5, batch_size=32) #amazon
    movie = model_movie.fit(mov_padded_sequence,movie_sentiment_label[0],validation_split=0.2, epochs=5, batch_size=32) #movie
    reddit = model_reddit.fit(reddit_padded_sequence,reddit_sentiment_label[0],validation_split=0.2, epochs=5, batch_size=32) #reddit


# plt.plot(twitter.history['accuracy'], label='acc')
# plt.plot(twitter.history['val_accuracy'], label='val_acc')
# plt.title("Twitter Data Accuracy")
# plt.legend()
# plt.xlabel("Epoch")
# plt.xlim(0,4)
# plt.show()
# plt.plot(twitter.history['loss'], label='loss')
# plt.plot(twitter.history['val_loss'], label='val_loss')
# plt.title("Twitter Data Loss")
# plt.legend()
# plt.xlabel("Epoch")
# plt.xlim(0,4)
# plt.show()


# plt.plot(amazon.history['accuracy'], label='acc')
# plt.plot(amazon.history['val_accuracy'], label='val_acc')
# plt.title("Amazon Data Accuracy")
# plt.legend()
# plt.xlabel("Epoch")
# plt.xlim(0,4)
# plt.show()
# plt.plot(amazon.history['loss'], label='loss')
# plt.plot(amazon.history['val_loss'], label='val_loss')
# plt.title("Amazon Data Loss")
# plt.legend()
# plt.xlabel("Epoch")
# plt.xlim(0,4)
# plt.show()

# plt.plot(movie.history['accuracy'], label='acc')
# plt.plot(movie.history['val_accuracy'], label='val_acc')
# plt.title("Movie Data Accuracy")
# plt.legend()
# plt.xlabel("Epoch")
# plt.xlim(0,4)
# plt.show()
# plt.plot(movie.history['loss'], label='loss')
# plt.plot(movie.history['val_loss'], label='val_loss')
# plt.title("Movie Data Loss")
# plt.legend()
# plt.xlabel("Epoch")
# plt.xlim(0,4)
# plt.show()


# plt.plot(reddit.history['accuracy'], label='acc')
# plt.plot(reddit.history['val_accuracy'], label='val_acc')
# plt.title("Reddit Data Accuracy")
# plt.legend()
# plt.xlabel("Epoch")
# plt.xlim(0,4)
# plt.show()
# plt.plot(reddit.history['loss'], label='loss')
# plt.plot(reddit.history['val_loss'], label='val_loss')
# plt.title("Reddit Data Loss")
# plt.legend()
# plt.xlabel("Epoch")
# plt.xlim(0,4)
# plt.show()


def predict_sentiment(text, model, sentimentlabel, tokenizer):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(model.predict(tw).round().item())
    print("Predicted label: ", sentimentlabel[1][prediction])
    return sentimentlabel[1][prediction]

# test_sentence1 = "Still waiting on bags from flight 1613/2440 yesterday  First Class passenger not happy with your service."
# predict_sentiment(test_sentence1, model_amazon, amazon_sentiment_label)
# predict_sentiment(test_sentence1, model_twitter, sentiment_label)
# predict_sentiment(test_sentence1, model_reddit, reddit_sentiment_label)
# predict_sentiment(test_sentence1, model_movie, movie_sentiment_label)

print("PREDICITNG SENTIMENTS")

tweet_sentiments = test_tweet_df['text'].apply(predict_sentiment, model=model_twitter, sentimentlabel=twsentiment_label, tokenizer=twtokenizer)
amazon_sentiments = test_amazon_review_df['text'].apply(predict_sentiment, model=model_amazon, sentimentlabel=amazon_sentiment_label, tokenizer=amzntokenizer)
movie_sentiments = test_movie_df['text'].apply(predict_sentiment, model=model_movie, sentimentlabel=movie_sentiment_label, tokenizer=movtokenizer)
reddit_sentiments = test_reddit_df['text'].apply(predict_sentiment, model=model_reddit, sentimentlabel=reddit_sentiment_label, tokenizer=rdttokenizer)

def findaccuracy(testset, generatedset): 
    diff = pd.DataFrame(testset["sentiment"].compare(generatedset, align_axis=0))
    return (testset.shape[0] - (diff.shape[0]/2)) /  testset.shape[0]
# correctposcount = 0
# for row in diff.rows(): 
#     if 

f = open("outfile.txt", "w+")

f.write("accuracy for models trained on their own dataset \n")
f.write("twitter: " + str(findaccuracy(test_tweet_df, tweet_sentiments)) + "\n")
f.write("amazon: " + str(findaccuracy(test_amazon_review_df, amazon_sentiments))+ "\n")
f.write("movie: " + str(findaccuracy(test_movie_df, movie_sentiments))+ "\n")
f.write("reddit: " + str(findaccuracy(test_reddit_df, reddit_sentiments))+ "\n")


tweetwreddit_sentiments = test_tweet_df['text'].apply(predict_sentiment, model=model_reddit, sentimentlabel=twsentiment_label, tokenizer=rdttokenizer)
tweetwmovie_sentiments = test_tweet_df['text'].apply(predict_sentiment, model=model_movie, sentimentlabel=twsentiment_label, tokenizer=movtokenizer)
tweetwamazon_sentiments = test_tweet_df['text'].apply(predict_sentiment, model=model_amazon, sentimentlabel=twsentiment_label, tokenizer=amzntokenizer)

f.write("\n accuracy for different models testing on twitter dataset \n")
f.write("reddit: "  + str(findaccuracy(test_tweet_df, tweetwreddit_sentiments)) + "\n")
f.write("movie: " + str(findaccuracy(test_tweet_df, tweetwmovie_sentiments)) + "\n")
f.write("amazon: "  + str(findaccuracy(test_tweet_df, tweetwamazon_sentiments)) + "\n")


amazonwreddit_sentiments = test_amazon_review_df['text'].apply(predict_sentiment, model=model_reddit, sentimentlabel=amazon_sentiment_label, tokenizer=rdttokenizer)
amazonwmovie_sentiments = test_amazon_review_df['text'].apply(predict_sentiment, model=model_movie, sentimentlabel=amazon_sentiment_label, tokenizer=movtokenizer)
amazonwtweet_sentiments = test_amazon_review_df['text'].apply(predict_sentiment, model=model_twitter, sentimentlabel=amazon_sentiment_label, tokenizer=twtokenizer)

f.write("\n accuracy for different models testing on amazon dataset \n")
f.write("reddit: "  + str(findaccuracy(test_amazon_review_df, amazonwreddit_sentiments)) + "\n")
f.write("movie: "  + str(findaccuracy(test_amazon_review_df, amazonwmovie_sentiments)) + "\n")
f.write("twitter: " + str(findaccuracy(test_amazon_review_df, amazonwtweet_sentiments)) + "\n")

moviewreddit_sentiments = test_movie_df['text'].apply(predict_sentiment, model=model_reddit, sentimentlabel=movie_sentiment_label, tokenizer=rdttokenizer)
moviewamazon_sentiments = test_movie_df['text'].apply(predict_sentiment, model=model_amazon, sentimentlabel=movie_sentiment_label, tokenizer=amzntokenizer)
moviewtweet_sentiments = test_movie_df['text'].apply(predict_sentiment, model=model_twitter, sentimentlabel=movie_sentiment_label, tokenizer=twtokenizer)

f.write("\n accuracy for different models on movie dataset \n")
f.write("reddit: "  + str(findaccuracy(test_movie_df, moviewreddit_sentiments)) + "\n")
f.write("amazon: " + str(findaccuracy(test_movie_df, moviewamazon_sentiments)) + "\n")
f.write("twitter: " + str(findaccuracy(test_movie_df, moviewtweet_sentiments)) + "\n")

redditwmovie_sentiments = test_reddit_df['text'].apply(predict_sentiment, model=model_movie, sentimentlabel=reddit_sentiment_label, tokenizer=movtokenizer)
redditwamazon_sentiments = test_reddit_df['text'].apply(predict_sentiment, model=model_amazon, sentimentlabel=reddit_sentiment_label, tokenizer=amzntokenizer)
redditwtweet_sentiments = test_reddit_df['text'].apply(predict_sentiment, model=model_twitter, sentimentlabel=reddit_sentiment_label, tokenizer=twtokenizer)

f.write("\n accuracy for different models testing on reddit dataset \n")
f.write("movie: " + str(findaccuracy(test_reddit_df, redditwmovie_sentiments)) + "\n")
f.write("amazon: " + str(findaccuracy(test_reddit_df, redditwamazon_sentiments)) + "\n")
f.write("twitter: " + str(findaccuracy(test_reddit_df, redditwtweet_sentiments)) + "\n")


f.close()
