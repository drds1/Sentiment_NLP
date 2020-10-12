import pandas as pd
from torch.utils.data import Dataset
import torch
import keras
import numpy as np
import utils
import os
import pickle
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm


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

                '''
                Load word embeddings
                '''
                word_counts = pd.DataFrame(dict(tok.word_counts), index=['count']).transpose().sort_values(by='count',                                                                                   ascending=False)
                num_words = len(word_counts)
                tok_dict = dict(tok.index_word)
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
                self.embedding_matrix = embedding_matrix


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


def swish(x):
    return x * F.sigmoid(x)

class Network(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(8, 16)
        self.b1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 8)
        self.b2 = nn.BatchNorm1d(8)
        self.fc3 = nn.Linear(8,4)
        self.b3 = nn.BatchNorm1d(4)
        self.fc4 = nn.Linear(4,1)

    def forward(self,x):

        x = swish(self.fc1(x))
        x = self.b1(x)
        x = swish(self.fc2(x))
        x = self.b2(x)
        x = swish(self.fc3(x))
        x = self.b3(x)
        x = F.sigmoid(self.fc4(x))

        return x


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
def train(model, x, y, optimizer, criterion):
    model.zero_grad()
    output = model(x)
    loss =criterion(output,y)
    loss.backward()
    optimizer.step()

    return loss, output

if __name__ == '__main__':

    ## Initialize the DataSet

    EPOCHS = 200
    BATCH_SIZE = 16
    data = DisasterDataset('./data/train.csv')
    data_train = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=False)

    # setup pytorch NN using script below.
    criterion = nn.MSELoss()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        correct = 0
        for bidx, batch in tqdm(enumerate(data_train)):
            x_train, y_train = batch['inp'], batch['oup']
            x_train = x_train.view(-1, 8)
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            loss, predictions = train(net, x_train, y_train, optm, criterion)
            for idx, i in enumerate(predictions):
                i = torch.round(i)
                if i == y_train[idx]:
                    correct += 1
            acc = (correct / len(data))
            epoch_loss += loss
        print('Epoch {} Accuracy : {}'.format(epoch + 1, acc * 100))
        print('Epoch {} Loss : {}'.format((epoch + 1), epoch_loss))