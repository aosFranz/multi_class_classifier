import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation

from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
import numpy as np
import math

raw_data = pd.read_csv('Toxic_Comment/train.csv')

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
raw_labels = raw_data[list_classes].values

labels = []
for raw_label in raw_labels:
    if all(i == 0 for i in raw_label):
        full_label = np.append(raw_label,1)
        labels.append(full_label)
    else:
        full_label = np.append(raw_label,0)
        labels.append(full_label)
labels = np.array(labels)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(raw_data['comment_text'])
tokenized = tokenizer.texts_to_sequences(raw_data['comment_text'])


def create_batches_of_size(list,size):
    batches = []
    i = 0
    while i < math.ceil(len(list) / size):
        batch_i = np.array(list[i * size:(i + 1) * size])
        batches.append(batch_i)
        i += 1
    return batches

batches_x = create_batches_of_size(tokenized,1000)
batches_y = create_batches_of_size(labels,1000)
tfidf_batches = [tokenizer.sequences_to_matrix(batch,mode='tfidf') for batch in batches_x]


model = Sequential()
model.add(Dense(512,input_dim=tfidf_batches[0].shape[1]))
model.add(Activation('relu'))
model.add(Dense(7))
model.add(Activation('softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x=tfidfed,y=labels,batch_size=1000,epochs=10)

