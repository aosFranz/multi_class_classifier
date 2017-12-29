import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation

from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
import numpy as np
import math

raw_data = pd.read_csv('/home/andreas/Data/Kaggle/Toxic_Comment/train.csv')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(raw_data['comment_text'])
tokenized = tokenizer.texts_to_sequences(raw_data['comment_text'])


def create_batches_of_size(list,size):
    batches = []
    i = 0
    while i < math.ceil(len(list) / size):
        batch_i = list[i * size:(i + 1) * size]
        batches.append(batch_i)
        i += 1
    return batches

batches = create_batches_of_size(tokenized,1000)
tfidf = [tokenizer.sequences_to_matrix(batch,mode='tfidf') for batch in batches]

print(len(tokenizer.__dict__))
print(len(tfidf[0][0]))

model = Sequential()
model.add(Dense(512,input_dim=len(tokenizer.__dict__)))


