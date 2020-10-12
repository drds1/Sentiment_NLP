import pandas as pd
from torch.utils.data import Dataset
import torch
import keras
import numpy as np
import utils
import os
import pickle
from sklearn.model_selection import train_test_split

#pytorch implementation of sentiment analysis challenge
#https://towardsdatascience.com/how-to-code-a-simple-neural-network-in-pytorch-for-absolute-beginners-8f5209c50fdd

class DisasterDataset(Dataset):

    def __init__(self, csvpath, mode='train'):
        self.mode = mode
        df = pd.read_csv(csvpath)
        le = LabelEncoder()
        if self.mode == 'train':
            '''
            Load the train data and split into train test sets
            '''
            X = list(df['text'])
            '''
            tokenize the input X_train data
            '''
            tok = keras.preprocessing.text.Tokenizer(num_words=1000)
            tok.fit_on_texts(X)
            if self.mode == 'train':
                self.tok = tok
                # integer encode documents
                X_train = tok.texts_to_sequences(X)
                #padd so all same length
                X_train = keras.preprocessing.sequence.pad_sequences(X_train,padding='post')
                self.maxpad = X_train.shape[1]
                self.inp = self.X_train.values
                self.oup = self.list(df['target'])
            else:
                #transform test data
                X_test = self.tok.texts_to_sequences(X)
                X_test = keras.preprocessing.sequence.pad_sequences(X_test,padding='post',maxlen=self.maxpad)
                self.inp = self.X_test.values

    def __len__(self):
        return len(self.inp)
    def __getitem__(self, idx):
        if self.mode == 'train':
            inpt  = torch.Tensor(self.inp[idx])
            oupt  = torch.Tensor(self.oup[idx])
            return { 'inp': inpt,
                     'oup': oupt}
        else:
            inpt = torch.Tensor(self.inp[idx])
            return { 'inp': inpt}


#template dataset class
#create dataset class
#class TitanicDataset(Dataset):
#    def __init__(self,csvpath, mode = 'train'):
#        self.mode = mode
#        df = pd.read_csv(csvpath)
#        le = LabelEncoder()
#        """
#        <------Some Data Preprocessing---------->
#        Removing Null Values, Outliers and Encoding the categorical labels etc
#        """
#        if self.mode == 'train':
#            df = df.dropna()
#            self.inp = df.iloc[:,1:].values
#            self.oup = df.iloc[:,0].values.reshape(len(df),1)
#        else:
#            self.inp = df.values
#
#    def __len__(self):
#        return len(self.inp)
#    def __getitem__(self, idx):
#        if self.mode == 'train':
#            inpt  = torch.Tensor(self.inp[idx])
#            oupt  = torch.Tensor(self.oup[idx])
#            return { 'inp': inpt,
#                     'oup': oupt}
#        else:
#            inpt = torch.Tensor(self.inp[idx])
#            return { 'inp': inpt}
#
#
if __name__ == '__main__':

    df = 1