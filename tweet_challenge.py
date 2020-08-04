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













#test_file='./data/test.csv'