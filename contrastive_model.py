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
from keras.models import Model, Sequential
from keras.layers import Input, Dense, concatenate
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import load_model
from keras.layers import Conv1D, Flatten, GlobalMaxPooling1D, Dropout, MaxPooling1D, Activation, Bidirectional, Conv2D, GlobalMaxPooling2D
from keras.layers import MaxPool2D, MaxPool1D
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

def generation_csv(total_path):
    f = open(total_path, 'r', encoding='utf-8')
    rdr = csv.reader(f)
    descriptions = list(rdr)[1:]
    text_list = list()

    for description in descriptions:
        try:
            text = description[0]
        except:
            continue
        if text.startswith('RT'):
            text_l = text.split()
            text = ' '.join(text_l[2:])
        reg_text = regex_(text)

        if reg_text:
            text_list.append(only_english)
            
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

# CSI-poitive token data
token_data_1 = generation_cve(path+'/CVE/cve_list.txt')
# CSI-negative token data
token_data_2 = generation_csv(path+'/data/out.csv')

stop_words = set(stopwords.words("english"))

token_list_1 = (token_data_1['text'].apply(get_tokens))
token_list_2 = (token_data_2['text'].apply(get_tokens))

word_counts = {}
for token in token_list_1:
     for w in token:
        word_counts[w] = word_counts.get(w, 0) + 1
word_counts = sorted(word_counts.items(), key=(lambda x: x[1]), reverse=True)
pos_vocab = [w for i, w in enumerate(word_counts) if i < 5000]
pos_vocab = [s for s, v in pos_vocab]

word_counts = {}
for token in token_list_2:
     for w in token:
        word_counts[w] = word_counts.get(w, 0) + 1
word_counts = sorted(word_counts.items(), key=(lambda x: x[1]), reverse=True)
neg_vocab = [w for i, w in enumerate(word_counts) if i < 5000]
neg_vocab = [s for s, v in neg_vocab]

pos_t = Tokenizer()
pos_t.fit_on_texts(pos_vocab)
vocab_sizes = len(pos_t.word_index)+1

neg_t = Tokenizer()
neg_t.fit_on_texts(neg_vocab)
neg_vocab_sizes = len(neg_t.word_index)+1


csi_positive, csi_negative = compare_drop(csi_positive, csi_negative)

y_1 = pd.DataFrame([0] * len(csi_positive), columns=['label'])
y_2 = pd.DataFrame([1] * len(csi_negative), columns=['label'])

max_length = 100

X = pd.concat([csi_positive, csi_negative])
Y = pd.concat([y_1, y_2])

encoded_docs_pos = pos_t.texts_to_sequences(X['text'])
encoded_docs_neg = neg_t.texts_to_sequences(X['text'])

X_p = pad_sequences(encoded_docs_pos, maxlen=max_length, padding='post')
X_n = pad_sequences(encoded_docs_neg, maxlen=max_length, padding='post')
Y = Y.to_numpy()

embedding_vector_length = 100
hidden_dims = 250

# CSI positive Embedding model 
inputA = Input(shape=(100,))
x = Embedding(vocab_sizes, embedding_vector_length, input_length=max_length)(inputA)
#x = Conv1D(128, 5, activation='relu')(x)
#x = MaxPool1D(strides=1, padding='valid')(x)
x = Bidirectional(LSTM(128))(x)
x = Dropout(0.2)(x)
x = Activation('relu')(x)
x = Model(inputs=inputA, outputs=x)

# CSI negative Embedding model 
inputC = Input(shape=(100,))
k = Embedding(neg_vocab_sizes, embedding_vector_length, input_length=max_length)(inputC)
#k = Conv1D(128, 5, activation='relu')(k)
#k = MaxPool1D(strides=1, padding='valid')(k)
k = Bidirectional(LSTM(128))(k)
k = Dropout(0.2)(k)
k = Activation('relu')(k)
k = Model(inputs=inputC, outputs=k)

combined = concatenate([x.output, k.output], axis=1)
#z = Flatten()(combined)
z = Dense(8, activation='relu')(combined)
z = Dense(1, activation="sigmoid")(z)

model = Model(inputs=[x.input, k.input], outputs=z)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit([X_p, X_n], Y, epochs=10, batch_size=32)

model.save('../model/lstm/model_5.h5')