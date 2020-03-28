import pandas as pd
import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
import random
import spacy
from torchtext.vocab import GloVe

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


trainid = pd.read_csv('train.csv', sep=',')
testid = pd.read_csv('test.csv', sep=',')
text = pd.read_csv('text.csv', sep=',')


trainPD = pd.merge(text, trainid, left_on='id', right_on='id')
test = pd.merge(text, testid, left_on='id', right_on='id')


trainPD.columns = ['id', 'text', 'label']
trainPD.to_csv('trainPD.csv')


test.columns = ['id', 'text']
test.to_csv('testPD.csv')



#spacy.load("en_core_web_md")
nlp = spacy.load("en")


tokenizer = "spacy"
TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths=True, batch_first=True)



LABEL = data.LabelField()



train = data.TabularDataset(path='trainPD.csv', format='csv', fields=[('row', None), ('id', None), ('text', TEXT), ('label', LABEL)], skip_header=True)
# test = data.TabularDataset(path='test.csv', format='csv', fields=[('row', None), ('id', None), ('text', TEXT)], skip_header=True)


TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300), max_size=10000, min_freq=6)
LABEL.build_vocab(train,)


train_data, valid_data = train.split(split_ratio=0.8, random_state=random.seed(123))
train_iter, valid_iter = data.BucketIterator.splits((train_data, valid_data), batch_size=32, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)

print(TEXT.vocab.stoi)
print(test['text'][0])




"""
def load_dataset(test_sen=None):
'''
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.

    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.

    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
'''

    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True,
                      fix_length=200)
    LABEL = data.LabelField(tensor_type=torch.FloatTensor)
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors
    print("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print("Label Length: " + str(len(LABEL.vocab)))

    train_data, valid_data = train_data.split()  # Further splitting of training_data to create new training_data & validation_data
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=32,
                                                                   sort_key=lambda x: len(x.text), repeat=False,
                                                                   shuffle=True)

    '''Alternatively we can also use the default configurations'''
    # train_iter, test_iter = datasets.IMDB.iters(batch_size=32)

    vocab_size = len(TEXT.vocab)

    return TEXT, vocab_size, word_embeddings, train_iter, valid_iter, 

"""