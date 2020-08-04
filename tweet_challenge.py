# import dependencies
import pandas as pd
import keras
import numpy as np
import utils

'''
Load the train data
'''
df_train = pd.read_csv('./data/train.csv')
X_train = list(df_train['text'])
y_train = list(df_train['target'])
y_train = pd.get_dummies(y_train).values


'''
tokenize the input X_train data
'''
tok = keras.preprocessing.text.Tokenizer(num_words=1000)
tok.fit_on_texts(X_train)
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
X_train_seq = tok.texts_to_sequences(X_train)
max_sentence_len = max([max(a) for a in X_train_seq if len(a) > 0])
#padd so all same length
X_train_seq_pad = keras.preprocessing.sequence.pad_sequences(X_train_seq,
                                                                     padding='post')


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
nlabels = len(y_train[0])
model_lstm = keras.Sequential()
# initialise Ebedding layer num_words = len(idx_word) + 1 to deal with 0 padding
# input_length is the number of words ids per sample e.g 28
# NOT the sample size of the training data
# you do not need to supply that info
model_lstm.add(keras.layers.Embedding(input_dim=num_words,
                                      input_length=X_train_seq_pad.shape[1],
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
model_lstm.add(keras.layers.Dense(nlabels))
model_lstm.add(keras.layers.Activation('softmax'))

# Compile the model
model_lstm.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model summary
model_lstm.summary()

##fit
model_lstm.fit(X_train_seq_pad, y_train, epochs=5, batch_size=128)














#test_file='./data/test.csv'
