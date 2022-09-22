import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import nltk
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from numpy import zeros
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import load_model
from keras.layers import Conv1D, Flatten, GlobalMaxPooling1D, Dropout, MaxPooling1D, Activation, Bidirectional, Conv2D, GlobalMaxPooling2D
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.corpus import wordnet as wn
from gensim.models import Word2Vec
from nltk.stem import PorterStemmer
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import svm
import pickle
from sklearn.externals import joblib
import re

def regex_(text):
    # 영어, 숫자, 특수만문자 제외 삭제.
    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+/(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    text = re.sub(pattern, '', text)
    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+' # URL제거
    text = re.sub(pattern, '', text)
    pattern = '(http|ftp|https):// (?:[-\w.]|(?:%[\da-fA-F]{2}))+' # URL제거
    text = re.sub(pattern, '', text)
    only_english = re.sub('[^ a-zA-Z]', '', text)
    only_english = only_english.lower()

    if bool(only_english and only_english.strip()) and len(only_english) >= 10:
        return only_english
    return False

def generate_pos_tweet(total_path):
    pic_str = 'pic.twitter.com/'
    file_list = os.listdir(total_path)
    text_list = []
    for i in range(len(file_list)):
        try:
            with open(total_path+file_list[i], 'r') as json_file:
                tweets = [json.loads(line) for line in json_file]
                count = 0

                if not tweets:
                    continue

                for tweet in tweets:
                    text = tweet['text']
                    tweet_l = text.split()
                    for t in tweet_l:
                        if pic_str in t:
                            len_text = len(t)
                    idx = text.find(pic_str)
                    if idx != -1:
                        text = text[:idx]+text[idx+len_text:]
                    
                    reg_text = regex_(text)
                    if reg_text:
                        count += 1
                        text_list.append(only_english)
        except:
            continue
    texts = pd.DataFrame(text_list, columns=['text'])
    return texts

def generate_neg_tweet(total_path):
    text_list = []
    with open(total_path+'general.json', 'r') as json_file:
        # 영어, 숫자, 특수만문자 제외 삭제.

        tweets = [json.loads(line) for line in json_file]
        for tweet in tweets:
            text = tweet['text']

            reg_text = regex_(text)

            if reg_text:
                text_list.append(reg_text)                
    texts = pd.DataFrame(text_list, columns=['text'])
    return texts

def generation_cve(total_path):
    description_lines = open(total_path, 'r')
    dscriptions = description_lines.readlines()

    text_list = []
    for i, d in enumerate(dscriptions):
        text = d
        reg_text = regex_(text)

        if reg_text:
            text_list.append(reg_text)

    texts = pd.DataFrame(text_list, columns=['text'])
    return texts

def compare_drop(top_data, last_data):
    top_n = len(top_data)
    last_n = len(last_data)
    range_ = top_n-last_n
    if range_ < 0:
        last_data = last_data.sample(n=(last_n+range_) , random_state=1)
    else:
        top_data = top_data.sample(n=(top_n-range_), random_state=1)
    return top_data, last_data

def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

def get_tokens(sentence):
    tknzr = TweetTokenizer()
    tokens = tknzr.tokenize(sentence)
    tokens = [token for token in tokens if (token not in stop_words and len(token) > 1)]
    tokens = [get_lemma(token) for token in tokens]
    return (tokens)


# CSI-positive data set
csi_positive = generate_pos_tweet(path+'/final_data/training_data/top_100/')
# CSI-negative data set
csi_negative = generate_neg_tweet(path+'/final_data/training_data/')

# token data
token_data = generation_cve(path+'/CVE/cve_list.txt')
stop_words = set(stopwords.words("english"))
token_list = (token_data['text'].apply(get_tokens))

word_counts = {}
for token in token_list:
     for w in token:
        word_counts[w] = word_counts.get(w, 0) + 1
word_counts = sorted(word_counts.items(), key=(lambda x: x[1]), reverse=True)
vocab = [w for i, w in enumerate(word_counts) if i < 5000]
vocab = [s for s, v in vocab]

t = Tokenizer()
t.fit_on_texts(vocab)
vocab_sizes = len(t.word_index)+1

csi_positive, csi_negative = compare_drop(csi_positive, csi_negative)

y_1 = pd.DataFrame([0] * len(csi_positive), columns=['label'])
y_2 = pd.DataFrame([1] * len(csi_negative), columns=['label'])

max_length = 100

X = pd.concat([csi_positive, csi_negative])
Y = pd.concat([y_1, y_2])

encoded_docs = t.texts_to_sequences(X['text'])

X = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
Y = Y.to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

embedding_vector_length = 100
hidden_dims = 250

model = Sequential()
model.add(Embedding(vocab_sizes, embedding_vector_length, input_length=max_length))
#model.add(Conv1D(128, 5, activation='relu'))
model.add(Bidirectional(LSTM(128)))
#model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=32)

scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

model.save('../model/lstm/model_1.h5')



