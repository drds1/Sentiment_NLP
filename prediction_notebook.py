#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.models import model_from_json
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Conv1D, MaxPooling1D
from keras.layers.merge import concatenate
import prediction_utils as pu
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import pickle
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

class Nlp_Tweets:

    def __init__(self):
        self.output_submission = 'output_submission.csv'
        self.load_past_model = None
        self.train_df = None
        self.test_df = None
        self.glove_file = None
        self.epochs = 30
        self.validation_split = 0.2
        self.model = None
        self.batch_size = 64

    def load(self, train_file, test_file, glove_file):
        '''

        :param train_file:
        :param test_file:
        :param glove_file:
        :return:
        '''
        self.train_df = pd.read_csv(train_file)
        self.test_df = pd.read_csv(test_file)
        self.glove_file = glove_file

    def _clean(self,text):
        '''
        clean text
        :return:
        '''
        stext = pd.Series(text)
        cleaned_text = stext.apply(lambda x: pu.clean(str(x))).apply(lambda x: pu.preprocessor2(x))
        cleaned_text = cleaned_text.apply(lambda x: pu.preprocessor3(x)).apply(
            lambda x: pu.preprocessor4(x))
        cleaned_text = cleaned_text.apply(
            lambda x: ' '.join([word for word in x.split() if word not in (pu.stopwords)]))
        return cleaned_text


    def _predict(self,model, X_test_tx, X_test_ky, X_test_lc, batch_size = 32, verbose = 2):
        Y_pred = model.predict([X_test_tx, X_test_ky, X_test_lc], batch_size = batch_size, verbose = verbose)
        Y_pred = np.argmax(Y_pred, axis=1)
        return Y_pred

    def run(self):
        if self.train_df is None:
            raise Exception('must specify self.train_df (data frame of training data')
        if self.test_df is None:
            raise Exception('must specify self.test_df (dataframe of test data')
        if self.glove_file is None:
            raise Exception('must specify self.glove_file path link to word embeddings '
                            'file\n (see https://github.com/stanfordnlp/GloVe want glove6b100txt)')
        train_df = self.train_df
        test_df = self.test_df
        glove_file = self.glove_file

        # Randomization
        state = 1
        train_df = train_df.sample(frac=1, random_state=state)
        test_df = test_df.sample(frac=1, random_state=state)
        train_df.reset_index(inplace=True, drop=True)
        test_df.reset_index(inplace=True, drop=True)

        # ## ** 2: Data Preprocessing**
        train_df['text'] = train_df['text'].apply(lambda x: pu.remove_url(x))
        train_df['keyword_cleaned'] = train_df['keyword'].copy().apply(lambda x: pu.clean(str(x))).apply(
            lambda x: pu.preprocessor2(x))
        train_df['location_cleaned'] = train_df['location'].copy().apply(lambda x: pu.clean(str(x))).apply(
            lambda x: pu.preprocessor2(x))
        train_df['text_cleaned'] = train_df['text'].copy().apply(lambda x: pu.clean(x)).apply(lambda x: pu.preprocessor2(x))

        test_df['text'] = test_df['text'].apply(lambda x: pu.remove_url(x))
        test_df['keyword_cleaned'] = test_df['keyword'].copy().apply(lambda x: pu.clean(str(x))).apply(
            lambda x: pu.preprocessor2(x))
        test_df['location_cleaned'] = test_df['location'].copy().apply(lambda x: pu.clean(str(x))).apply(
            lambda x: pu.preprocessor2(x))
        test_df['text_cleaned'] = test_df['text'].copy().apply(lambda x: pu.clean(x)).apply(lambda x: pu.preprocessor2(x))

        # ## 3. Understanding of the training dataset - basic
        # Target count
        fig, ax = plt.subplots(figsize=(8, 5))
        pd.value_counts(train_df['target']).plot(kind="bar")
        ax.set_title('Target Count')
        ax.set_ylabel('Frequency')
        ax.grid(True)
        plt.savefig('target_count.pdf')

        # Top 20 Locations with most target occurance
        loc_pos = \
            train_df[
                (train_df.location_cleaned != 'nan') & (train_df.location_cleaned != ' ') & (train_df['target'] == 1)][
                'location_cleaned'].value_counts()
        loc_neg = \
            train_df[
                (train_df.location_cleaned != 'nan') & (train_df.location_cleaned != ' ') & (train_df['target'] == 0)][
                'location_cleaned'].value_counts()

        loc_pos_dict = loc_pos[:20].to_dict()
        loc_neg_dict = loc_neg[:20].to_dict()

        names0 = list(loc_neg_dict.keys())
        values0 = list(loc_neg_dict.values())
        names1 = list(loc_pos_dict.keys())
        values1 = list(loc_pos_dict.values())

        # Graph
        fig, (ax1, ax2) = plt.subplots(figsize=(20, 5), nrows=1, ncols=2)
        ax1.bar(range(len(loc_pos_dict)), values1, tick_label=names1)
        ax1.set_xticklabels(names1, rotation="vertical")
        ax1.set_ylim(0, 100)
        ax1.grid(True)
        ax1.set_title('Location with most Pos target')
        ax1.set_ylabel('Frequency')

        ax2.bar(range(len(loc_neg_dict)), values0, tick_label=names0)
        ax2.set_xticklabels(names0, rotation="vertical")
        ax2.set_ylim(0, 100)
        ax2.grid(True)
        ax2.set_title('Location with most Neg target')
        ax2.set_ylabel('Frequency')

        train_df['location_cleaned'] = train_df['location_cleaned'].copy().apply(lambda x: pu.preprocessor3(x)).apply(
            lambda x: pu.preprocessor4(x))
        train_df['text_cleaned'] = train_df['text_cleaned'].copy().apply(lambda x: pu.preprocessor3(x)).apply(
            lambda x: pu.preprocessor4(x))

        test_df['location_cleaned'] = test_df['location_cleaned'].copy().apply(lambda x: pu.preprocessor3(x)).apply(
            lambda x: pu.preprocessor4(x))
        test_df['text_cleaned'] = test_df['text_cleaned'].copy().apply(lambda x: pu.preprocessor3(x)).apply(
            lambda x: pu.preprocessor4(x))

        # ## 4. Understanding of the training dataset - word cloud visualization

        loc_pos_dict = loc_pos[:].to_dict()
        loc_neg_dict = loc_neg[:].to_dict()

        loc_list = list(loc_pos_dict.keys()) + list(loc_neg_dict.keys())
        unique_loc = []
        lower_loc = []
        for x in loc_list:
            if x not in unique_loc:
                unique_loc.append(x)

        for x in unique_loc:
            lower_loc.append(x.lower().replace(',', '').replace('.', ''))

        wc = pu.wc_base2(train_df[train_df.target == 0].text_cleaned)
        wc.plot_wc(title="Word Cloud of tweets with Negative target", stopwords=pu.stopwords)
        wc_s = set(wc.wordlist)
        wc2 = pu.wc_base2(train_df[train_df.target == 1].text_cleaned)
        wc2.plot_wc(title="Word Cloud of tweets with Positive target")
        wc2_s = set(wc2.wordlist)

        train_df['text_cleaned'] = train_df['text_cleaned'].apply(
            lambda x: ' '.join([word for word in x.split() if word not in (pu.stopwords)]))
        test_df['text_cleaned'] = test_df['text_cleaned'].apply(
            lambda x: ' '.join([word for word in x.split() if word not in (pu.stopwords)]))

        self.diag_train_df = train_df.copy()
        self.diag_test_df = test_df.copy()
        # ## 5.0 Data Augmentation -- EDA (Random  Swap - RS) and EDA (Random  Deletion - RD) - for training set

        train_df['index_no'] = train_df.index
        train_df['sent_w_index'] = train_df['text_cleaned'] + ' ' + train_df['index_no'].astype('str')

        train_df['da_text_cleaned'] = train_df['sent_w_index'].apply(lambda x: pu.random_swap(x))
        train_df['da_text_cleaned2'] = train_df['sent_w_index'].apply(lambda x: pu.random_del(x))

        train_df.drop(['index_no', 'sent_w_index'], axis=1, inplace=True)

        temp_df1 = list(
            zip(train_df.target, train_df.keyword_cleaned, train_df.location_cleaned, train_df.da_text_cleaned))
        temp_df2 = list(
            zip(train_df.target, train_df.keyword_cleaned, train_df.location_cleaned, train_df.da_text_cleaned2))

        x = pd.DataFrame(temp_df1, columns=['target', 'keyword_cleaned', 'location_cleaned', 'text_cleaned'])
        y = pd.DataFrame(temp_df2, columns=['target', 'keyword_cleaned', 'location_cleaned', 'text_cleaned'])

        train_df.drop(['da_text_cleaned', 'da_text_cleaned2'], axis=1, inplace=True)

        z = pd.concat([train_df, x, y], axis=0, join='outer', ignore_index=False, keys=None, sort=False)
        z = z[['id', 'keyword_cleaned', 'location_cleaned', 'text_cleaned', 'target']].copy()
        z.reset_index(inplace=True, drop=True)

        train_df = z
        test_df = test_df[['id', 'keyword_cleaned', 'location_cleaned', 'text_cleaned']].copy()

        # Randomization
        state = 1
        train_df = train_df.sample(frac=1, random_state=state)
        train_df.reset_index(inplace=True, drop=True)

        # ## 5.1 Build a Simple deep learning model
        top_word = 35000
        text_lengths = [len(x.split()) for x in (train_df.text_cleaned)]
        # text_lengths = [x for x in text_lengths if x < 50]
        plt.hist(text_lengths, bins=25)
        plt.title('Histogram of # of Words in Texts')

        tok = Tokenizer(num_words=top_word)
        tok.fit_on_texts((train_df['text_cleaned'] + train_df['keyword_cleaned'] + train_df['location_cleaned']))

        max_words = max(text_lengths) + 1
        max_words_ky = max([len(x.split()) for x in (train_df.keyword_cleaned)]) + 1
        max_words_lc = max([len(x.split()) for x in (train_df.location_cleaned)]) + 1
        print("top_word: ", str(top_word))
        print("max_words: ", str(max_words))
        print("max_words_ky: ", str(max_words_ky))
        print("max_words_lc: ", str(max_words_lc))

        # Training set
        val_value = 5000

        X_train_tx = tok.texts_to_sequences(train_df['text_cleaned'])
        X_train_ky = tok.texts_to_sequences(train_df['keyword_cleaned'])
        X_train_lc = tok.texts_to_sequences(train_df['location_cleaned'])

        X_test_tx = tok.texts_to_sequences(test_df['text_cleaned'])
        X_test_ky = tok.texts_to_sequences(test_df['keyword_cleaned'])
        X_test_lc = tok.texts_to_sequences(test_df['location_cleaned'])

        Y_train = train_df['target']

        print('Found %s unique tokens.' % len(tok.word_index))

        # One-hot category
        Y_train = to_categorical(Y_train)
        print("Y_train.shape: ", Y_train.shape)

        X_train_tx = sequence.pad_sequences(X_train_tx, maxlen=max_words)
        X_train_ky = sequence.pad_sequences(X_train_ky, maxlen=max_words_ky)
        X_train_lc = sequence.pad_sequences(X_train_lc, maxlen=max_words_lc)

        X_test_tx = sequence.pad_sequences(X_test_tx, maxlen=max_words)
        X_test_ky = sequence.pad_sequences(X_test_ky, maxlen=max_words_ky)
        X_test_lc = sequence.pad_sequences(X_test_lc, maxlen=max_words_lc)

        print("X_train_tx.shape: ", X_train_tx.shape)
        print("X_train_ky.shape: ", X_train_ky.shape)
        print("X_train_lc.shape: ", X_train_lc.shape)

        # download glove file from
        # https://github.com/stanfordnlp/GloVe
        # want glove6b100txt etc
        # make some python code to download this already if not present
        embeddings_dictionary = dict()
        glove_file = open(glove_file, encoding="utf8")

        for line in glove_file:
            records = line.split()
            word = records[0]
            vector_dimensions = asarray(records[1:], dtype='float32')
            embeddings_dictionary[word] = vector_dimensions

        glove_file.close()

        embedding_dim = 100
        embedding_matrix = zeros((top_word, embedding_dim))
        for word, index in tok.word_index.items():
            embedding_vector = embeddings_dictionary.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

        input1 = Input(shape=(max_words,))
        embedding_layer1 = Embedding(top_word, 100, weights=[embedding_matrix], input_length=max_words,
                                     trainable=False)(input1)
        dropout1 = Dropout(0.2)(embedding_layer1)
        lstm1_1 = LSTM(128, return_sequences=True)(dropout1)
        lstm1_2 = LSTM(128, return_sequences=True)(lstm1_1)
        lstm1_2a = LSTM(128, return_sequences=True)(lstm1_2)
        lstm1_3 = LSTM(128)(lstm1_2a)

        input2 = Input(shape=(max_words_ky,))
        embedding_layer2 = Embedding(top_word, 100, weights=[embedding_matrix], input_length=max_words_ky,
                                     trainable=False)(
            input2)
        dropout2 = Dropout(0.2)(embedding_layer2)
        lstm2_1 = LSTM(64, return_sequences=True)(dropout2)
        lstm2_2 = LSTM(64, return_sequences=True)(lstm2_1)
        lstm2_3 = LSTM(64)(lstm2_2)

        input3 = Input(shape=(max_words_lc,))
        embedding_layer3 = Embedding(top_word, 100, weights=[embedding_matrix], input_length=max_words_lc,
                                     trainable=False)(
            input3)
        dropout3 = Dropout(0.2)(embedding_layer3)
        lstm3_1 = LSTM(32, return_sequences=True)(dropout3)
        lstm3_2 = LSTM(32, return_sequences=True)(lstm3_1)
        lstm3_3 = LSTM(32)(lstm3_2)

        merge = concatenate([lstm1_3, lstm2_3, lstm3_3])

        dropout = Dropout(0.8)(merge)
        dense1 = Dense(256, activation='relu')(dropout)
        dense2 = Dense(128, activation='relu')(dense1)
        output = Dense(2, activation='softmax')(dense2)
        model = Model(inputs=[input1, input2, input3], outputs=output)
        model.summary()

        model.compile(loss="binary_crossentropy", optimizer="adam",
                      metrics=["accuracy"])

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)


        #save tokenizer and (needed to predict future entries)
        self.tok = tok
        self.max_words = max_words
        self.max_words_ky = max_words_ky
        self.max_words_lc = max_words_lc


        if self.load_past_model is None:
            history = model.fit([X_train_tx, X_train_ky, X_train_lc], Y_train,
                            validation_split=self.validation_split, epochs=self.epochs,
                            batch_size=self.batch_size,
                            verbose=2, callbacks=[es])
            pu.result_eva(history.history['loss'], history.history['val_loss'], history.history['accuracy'],
                          history.history['val_accuracy'])
        else:
            #https://machinelearningmastery.com/save-load-keras-deep-learning-models/
            model = pu.load_model(self.load_past_model)





        model.save('nlp_disaster.h5')

        model = Model()
        model = load_model('nlp_disaster.h5')
        model.compile(loss="binary_crossentropy", optimizer="adam",
                      metrics=["accuracy"])

        model.summary()
        Y_pred = self._predict(model, X_test_tx, X_test_ky, X_test_lc)
        pred_df = pd.DataFrame(Y_pred, columns=['target'])
        result = pd.concat([test_df, pred_df], axis=1, join='outer', ignore_index=False, keys=None, sort=False)
        result = result[['id', 'target']]
        result.tail()
        result.to_csv(self.output_submission, index=False)



        #save the trained model to use again
        self.model = model


class Nlp_General:
    def __init__(self):
        self.output_submission = 'output_submission.csv'
        self.train_df = None
        self.test_df = None
        self.glove_file = None
        self.epochs = 30
        self.validation_split = 0.2
        self.model = None
        self.batch_size = 64
    def load_data(self,train_text, train_target, test_text, glove_file):
        self.train_df = self._get_train(train_text,train_target)
        self.test_df = self._get_test(test_text)
        self.glove_file = glove_file


    def _get_train(self,text,target):
        n_train = len(text)
        return pd.DataFrame({'id': list(np.arange(n_train)),
                      'keyword': [np.nan] * n_train,
                      'location': [np.nan] * n_train,
                      'text': text,
                      'target': target})


    def _get_test(self,text):
        n_test = len(text)
        return pd.DataFrame({'id': list(np.arange(n_test)),
                      'keyword': [np.nan] * n_test,
                      'location': [np.nan] * n_test,
                      'text': text})


    def run(self):
        '''
        run the NLP_tweets class with the formatted data
        :return:
        '''
        nlpt = Nlp_Tweets()
        nlpt.epochs = self.epochs
        nlpt.output_submission = self.output_submission
        nlpt.train_df = self.train_df
        nlpt.test_df = self.test_df
        nlpt.glove_file = self.glove_file
        nlpt.epochs = self.epochs
        nlpt.validation_split = self.validation_split
        nlpt.batch_size = self.batch_size
        nlpt.run()
        self.nlpt = nlpt

    def predict(self,text,keywords = None, location = None):
        if keywords is None:
            kw = [np.nan]*len(text)
        else:
            kw = keywords
        if location is None:
            lc = [np.nan]*len(text)
        else:
            lc = location
        tok = self.nlpt.tok

        text_clean = pd.Series(self.nlpt._clean(text))
        text_tx = tok.texts_to_sequences(text_clean)
        text_tx = sequence.pad_sequences(text_tx, maxlen=self.nlpt.max_words)

        lc_clean = pd.Series(self.nlpt._clean(lc))
        lc_tx = tok.texts_to_sequences(lc_clean)
        lc_tx = sequence.pad_sequences(lc_tx, maxlen = self.nlpt.max_words_lc)

        kw_clean = pd.Series(self.nlpt._clean(kw))
        kw_tx = tok.texts_to_sequences(kw_clean)
        kw_tx = sequence.pad_sequences(kw_tx, maxlen = self.nlpt.max_words_ky)

        preds = self.nlpt._predict(self.nlpt.model, text_tx, kw_tx, lc_tx)

        return preds





if __name__ == '__main__':
    '''
    #test nlp tweets project
    x = Nlp_Tweets()
    x.epochs = 3
    x.validation_split = 0.2
    x.batch_size = 64
    x.load(train_file = './data/train.csv',
           test_file = './data/test.csv',
           glove_file = './data/glove.6B.100d.txt')
    x.run()
    #x.model.predict(new tweets)
    '''

    #test general project
    train_df_in = pd.read_csv('./data/train.csv')
    test_df_in = pd.read_csv('./data/test.csv')
    y = Nlp_General()
    y.epochs = 2
    y.validation_split = 0.2
    y.batch_size = 64
    train_text = list(train_df_in['text'])
    train_target = list(train_df_in['target'])
    test_text = list(test_df_in['text'])
    y.load_data(train_text= train_text, train_target= train_target,
                test_text= test_text,
                glove_file= './data/non_tracked/glove.6B.100d.txt')
    y.run()


    #save the trained nn
    # save simulation as pickle output
    picklefile = '/data/non_tracked/trained_nn.pickle'
    os.system('rm ' + picklefile)
    pickle_out = open(picklefile, "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


    #load previous simulation
    pickle_in = open(picklefile, "rb")
    y = pickle.load(pickle_in)

    y.predict(['disaster strikes plague fire'])
    y.predict(["isn't it a lovely day today"])

