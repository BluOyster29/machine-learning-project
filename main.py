import pandas as pd
from torchtext.data import Iterator, BucketIterator, TabularDataset, Field
from torch import nn
from ProjDataset import ProjDataset
from tqdm import tqdm_notebook
from torch.utils.data import DataLoader
import torch, pickle, random, os, spacy, argparse
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
import torch.optim as optim
from GRUNetwork import ConcatPoolingGRUAdaptive
from BatchGenerator import BatchGenerator

def get_args():

    parser = argparse.ArgumentParser(
        description="")

    parser.add_argument("-P", "--pretrained", dest='pretrained', type=str,
                        help="Use pretrained vectors (y/n)", default="n")
    parser.add_argument("-D", "--glove_dimension", dest='dim', type=int,
                        help="Choose dimension for pretrained embeddings", default=50)
    parser.add_argument("-G", "--use_gpu", dest='dim', type=str,
                        help="Use GPU for training (y/n)", default='y')

    args = parser.parse_args()

    return args

def get_fields():
    TEXT = Field(tokenize = 'spacy',
                  tokenizer_language = 'en',
                  lower = True,
                  include_lengths=True,
                  sequential=True,
                  use_vocab=True
                )
    LABEL = Field(sequential=False, use_vocab=False)

    return TEXT, LABEL

def get_dataset(TEXT, LABEL):

    lyric_datafield = [("lyrics", TEXT),
                     ("genre", LABEL)]


    train, test = TabularDataset.splits('data/dataframes/',
                                     train='training.csv',
                                     test ='testing.csv',
                                     format ='csv',
                                     fields = lyric_datafield,
                                     skip_header=True)

    return train, test

def get_vocab(trn, TEXT, pretrained_bool, dimensions):
    if pretrained==True:

        if dim == 50:
            TEXT.build_vocab(trn, max_size=100000, vectors="glove.6B/glove.6B.50d.txt")
        elif dim == 100:
            TEXT.build_vocab(trn, max_size=100000, vectors="glove.6B/glove.6B.100.txt")
        elif dim == 200:
            TEXT.build_vocab(trn, max_size=100000, vectors="glove.6B/glove.6B.200.txt")
        elif dim == 300:
            TEXT.build_vocab(trn, max_size=100000, vectors="glove.6B/glove.6B.300.txt")

        return TEXT

    else:
        TEXT.build_vocab(trn)
        return TEXT

def get_iterators(train_dataset, test_dataset):

        traindl, testdl = BucketIterator.splits(datasets=(train_dataset,test_dataset),
                     batch_sizes=(50,1),
                     sort_key=lambda x: len(x.lyrics),
                     device=None,
                     sort_within_batch=True,
                     repeat=False)

        return traindl, testdl

def train(model, nr_epochs, device, training_dl):

    nr_of_epochs = 20
    model.train()
    model = model.to(device)
    epoch_nr = 0
    EPOCH = list(range(nr_epochs))
    tenp = round(len(training_dl,) / 10)
    avg_loss = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch in tqdm_notebook(EPOCH):
        epoch_nr += 1
        epoch_loss = []
        count = 0
        for i in tqdm_notebook(training_dl):
            x = i[0][0]
            y = i[1].to(device)
            lengths = i[0][1]
            optimizer.zero_grad()
            try:
                output = model(x, lengths)
                loss = criterion(output, y.long())
                loss.backward()
                epoch_loss.append(loss.item())
                optimizer.step()
                avg_loss = sum(epoch_loss) / len(epoch_loss)
            except RuntimeError:
                continue
        print("Average loss at epoch %d: %.7f" % (epoch_nr, avg_loss))
    return trained_model
def evaluate(trained_model, testdl):
    correct = 0
    count = 0
    for i in tqdm_notebook(test_dl):

        x = i[0][0]
        y = i[1]
        lengths = i[0][1]
        try:
            predictions = pre_model(x, lengths)
            for prediction in zip(predictions,y):
                count+=1
                output, index = torch.max(prediction[0], 0)
                if index.item() == y[0].item():
                    correct += 1
        except:
            continue
    accuracy = (correct / count) * 100

    print('Model Accuracy: {}'.format(accuracy))

if __name__ == '__main__':

    args = get_args()
    pretrained = args.pretrained
    dim = args.dim
    x_field, y_field = get_fields()
    print('Generating Dataset')
    train_dataset, test_dataset = get_dataset(x_field, y_field)
    print('Generating Vocab')

    x_field = get_vocab(train_dataset, x_field, pretrained, dim)
    traindl, testdl = get_iterators(train_dataset, test_dataset)
    vocab_size = len(x_field.vocab)
    embedding_dim = dim
    n_out = 5
    EPOCHS = 10
    device = 'cpu'
    n_hidden = 128
    train_batch_it = BatchGenerator(traindl, 'lyrics', 'genre')
    test_batch_it = BatchGenerator(testdl, 'lyrics', 'genre')
    print('Generating Model')

    model = ConcatPoolingGRUAdaptive(vocab_size, embedding_dim,
                                    n_hidden, n_out, x_field.vocab.vectors,
                                    pretrained).to(device)

    print(model)
    print('Training')
    trained_model = train(model, EPOCHS, device, train_batch_it)
    print('Evaluating Model')
    evaluate(trained_model, test_batch_it)