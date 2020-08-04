# import dependencies
import pandas as pd
import keras

'''
Load the train data
'''
df_train = pd.read_csv('./data/train.csv')
X_train = list(df_train['text'])
y_train = list(df_train['target'])


'''
tokenize the input X_train data
'''
tok = keras.preprocessing.text.Tokenizer(num_words=1000)
tok.fit_on_texts(X_train)
#summarise top words
word_counts = pd.DataFrame(dict(tok.word_counts),index=['count']).transpose().sort_values(by='count',ascending=False)
print(str(len(word_counts))+' distint words found')
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
max_sentence_len = max([max(a) for a in X_train_seq])
#padd so all same length
X_train_seq_pad = keras.preprocessing.sequence.pad_sequences(Xtrain_sequence,
                                                                     padding='post')


'''
Load word embeddings
'''













#test_file='./data/test.csv'
