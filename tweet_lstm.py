# import dependencies
import pandas as pd
import keras
import numpy as np
import utils
import os
import pickle
from sklearn.model_selection import train_test_split

'''
Load the train data and split into train test sets
'''
df = pd.read_csv('./data/train.csv')
X = list(df['text'])
y = list(df['target'])
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#only OHE if > 2 classes or want prob of each class 'softmax' output activation
#Here just want 1 or 0
#y_train = pd.get_dummies(y_train).values
#y_test = pd.get_dummies(y_test).values

'''
tokenize the input X_train data
'''
tok = keras.preprocessing.text.Tokenizer(num_words=1000)
tok.fit_on_texts(X_train_raw)
#summarise top words
word_counts = pd.DataFrame(dict(tok.word_counts),index=['count']).transpose().sort_values(by='count',ascending=False)
num_words = len(word_counts)
tok_dict = dict(tok.index_word)
print(str(num_words)+' distinct words found')
print('top 10...')
print(word_counts.head(10))
print('bottom 10...')
print(word_counts.tail(10))
# summarize what was learned
#print(tok.word_counts)
#print(tok.document_count)
#print(tok.word_index)
#print(tok.word_docs)
# integer encode documents
X_train = tok.texts_to_sequences(X_train_raw)
max_sentence_len = max([max(a) for a in X_train if len(a) > 0])
#padd so all same length
X_train = keras.preprocessing.sequence.pad_sequences(X_train,padding='post')


'''
Load word embeddings
'''
word_embeddings_dict = utils.load_embeddings('./data/non_tracked/glove.6B.100d.txt')

'''
Create the embedding_matrix for the words in our vocabulary
'''
embeddings_words = list(word_embeddings_dict.keys())
wordvec_dim = word_embeddings_dict[embeddings_words[0]].shape[0]
embedding_matrix = np.zeros((num_words, wordvec_dim))
for i, word in tok_dict.items():
    # Look up the word embedding
    vector = word_embeddings_dict.get(word, None)
    # Record in matrix
    if vector is not None:
        embedding_matrix[i, :] = vector


'''
Setup the net
'''
model_lstm = keras.Sequential()
# initialise Ebedding layer num_words = len(idx_word) + 1 to deal with 0 padding
# input_length is the number of words ids per sample e.g 28
# NOT the sample size of the training data
# you do not need to supply that info
model_lstm.add(keras.layers.Embedding(input_dim=num_words,
                                      input_length=X_train.shape[1],
                                      output_dim=wordvec_dim,
                                      weights=[embedding_matrix],
                                      trainable=False,
                                      mask_zero=True))

# words which are not in the pretrained embeddings (with value 0) are ignored
model_lstm.add(keras.layers.Masking(mask_value=0.0))

# Recurrent layer
model_lstm.add(keras.layers.LSTM(200, return_sequences=False))
model_lstm.add(keras.layers.Dropout(0.4))
# model_lstm.add(keras.layers.LSTM(28, return_sequences=True))
# model_lstm.add(keras.layers.Dropout(0.2))
# model_lstm.add(keras.layers.LSTM(28, return_sequences=True))
# model_lstm.add(keras.layers.Dropout(0.2))
# model_lstm.add(keras.layers.LSTM(28, return_sequences=False))

# Output layer
model_lstm.add(keras.layers.Dense(1))
model_lstm.add(keras.layers.Activation('sigmoid'))

# Compile the model
model_lstm.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# model summary
model_lstm.summary()

##fit
model_lstm.fit(np.array(X_train), np.array(y_train), epochs=50, batch_size=128)



#predict on test data
X_test = tok.texts_to_sequences(X_test_raw)
X_test = keras.preprocessing.sequence.pad_sequences(X_test,padding='post',maxlen=max_sentence_len)
y_pred = model_lstm.predict(np.array(X_test))

#pickle model output
picklefile = './models/lstm.pickle'
os.system('rm ' + picklefile)
pickle_out = open(picklefile, "wb")
pickle.dump({'model':model_lstm,'X_train':X_train,'y_train':y_train,
             'X_test':X_test,'y_test':y_test,'y_pred':y_pred}, pickle_out)
pickle_out.close()



#optional run K-fold cross validation to asses model performance
#from benchmarking_models import run_cv
#y_pred = run_cv(X_train, y_train)

