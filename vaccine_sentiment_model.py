'''
vaccine sentiment analysis script
@ author : Darren Miller
@ Description of file: The script will uses a manually labelled dataset of vaccines, equivalent to nearly 1%
of the overall vaccine tweet dataset, to build a sentiment analysis model. This is to then be used in the
vaccine_sentiment_analysis.py script.

Current status: The script is complete and ready for use. Includes history and saved model for restoring.

Output created: Senti_model.tf -- the sentiment model for vaccines
'''
#%%
from os import truncate
from google.protobuf.text_format import Tokenizer
from nltk.corpus.reader.chasen import test
from nltk.util import pad_sequence
import numpy as np
from numpy.lib.function_base import rot90
from sklearn.metrics._plot.confusion_matrix import plot_confusion_matrix
from tensorflow.python.keras import activations
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import embeddings
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.wrappers import Bidirectional
from tensorflow.python.ops.gen_batch_ops import batch
seed = np.random.seed(42)

#%%
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist, bigrams
from collections import Counter, OrderedDict
import itertools
import copy
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
import pickle
AUTOTUNE = tf.data.experimental.AUTOTUNE
import matplotlib.pyplot as plt

# %%
''' reading in labelled tweets -- id and author_id have been altered going to and from excel but _id is fine'''
vaccine_tweets_labelled = pd.read_csv('vaccine_tweets_labelled.csv', index_col=0)

#%%
vaccine_tweets_labelled['Label'] = pd.Categorical(vaccine_tweets_labelled['Label'])

#%%
vaccine_tweets_labelled_list = vaccine_tweets_labelled['text'].to_list()

# %%
''' model details'''
vocab_size = 5000
embedding_size = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = 0.8

#%%
labelled_list_of_tweets = []
list_of_labels = vaccine_tweets_labelled['Label'].tolist()
stop_words = stopwords.words('english')

for tweet in vaccine_tweets_labelled_list:
  article = tweet
  for word in stop_words:
    token = ' ' + word + ' '
    article = article.replace(token, ' ')
    article = article.replace(' ',' ')
  labelled_list_of_tweets.append(article)

#%%
''' building the sentiment model - splitting data out '''
sentiment_train_x, sentiment_test_x, sentiment_train_y, sentiment_test_y = train_test_split(
    labelled_list_of_tweets, list_of_labels, test_size = 0.20,
    random_state = seed, stratify = list_of_labels)

#%%
over_sample = RandomOverSampler(random_state= seed)
sentiment_train_x_resampled, sentiment_train_y_resampled  = over_sample.fit_resample(np.array(sentiment_train_x).reshape(-1,1), sentiment_train_y)
#np.array(sentiment_train_x).reshape(-1,1), sentiment_train_y)

#%%
''' convert resampled_train_X to list'''
senti_train_x = []
for i in range(0, len(sentiment_train_x_resampled)):
  senti_train_x.append(sentiment_train_x_resampled[i][0])

#%%
tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(senti_train_x)
word_index = tokenizer.word_index
dict(list(word_index.items())[0:10])

#%%
train_sequences = tokenizer.texts_to_sequences(senti_train_x)

#%%
train_padded = pad_sequences(train_sequences, maxlen = max_length,
padding = padding_type, truncating = trunc_type)

#%%
test_sequences = tokenizer.texts_to_sequences(sentiment_test_x)
test_padded = pad_sequences(test_sequences, maxlen = max_length, 
padding = padding_type, truncating = trunc_type)

#%%
model = tf.keras.Sequential([
  keras.layers.Embedding(vocab_size+1, embedding_size, mask_zero=True),
  keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences = True)),
  keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
  keras.layers.Dropout(0.5),
  keras.layers.Dense(embedding_size, activation = 'relu'),
  keras.layers.Dropout(0.5),
  keras.layers.Dense(4, activation = 'softmax')
])

# %%
model.compile(loss = "sparse_categorical_crossentropy", optimizer = keras.optimizers.Adam(learning_rate = 0.000005),
              metrics = ["accuracy"])

#%%
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(sentiment_train_y),
                                                 sentiment_train_y)

class_weights = dict(enumerate(class_weights))

#%%
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 30, restore_best_weights=True)

# %%
history = model.fit(train_padded, np.array(sentiment_train_y_resampled), epochs=100,
                    validation_data=(test_padded, np.array(sentiment_test_y)),
                    validation_steps=100,
                    callbacks = [es])

# %%
test_loss, test_acc = model.evaluate(test_padded, np.array(sentiment_test_y))

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)


# %%
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'], 'g')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history.history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], 'g')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%
test_loss, test_acc = lstm_model.evaluate(test_padded, np.array(sentiment_test_y))

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)


# %%
