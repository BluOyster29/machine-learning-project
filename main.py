import pandas as pd
from torchtext.data import Iterator, BucketIterator, TabularDataset, Field
from torch import nn
from ProjDataset import ProjDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch, pickle, random, os, spacy, argparse, dill
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
import torch.optim as optim
from GRUNetwork import Rnn_Gru
from BatchGenerator import BatchGenerator

def get_args():

    parser = argparse.ArgumentParser(
        description="")

    parser.add_argument("-P", "--pretrained", dest='pretrained', type=str,
                        help="Use pretrained vectors (y/n)", default="n")
    parser.add_argument("-B", "--batch_size", dest='batch_size', type=int,
                        help="size of batches", default="100")
    parser.add_argument("-E", "--epochs", dest='nr_epochs', type=int,
                        help="Number of epochs", default="10")
    parser.add_argument("-D", "--glove_dimension", dest='dim', type=int,
                        help="Choose dimension for pretrained embeddings", default=50)
    parser.add_argument("-G", "--use_gpu", dest='gpu', type=str,
                        help="Use GPU for training (y/n)", default='y')
    parser.add_argument("-M", "--model_name", dest='model_name', type=str,
                        help="name of model to save", default='unnamed_model')
    args = parser.parse_args()

    return args

def get_fields():
    
    TEXT = Field(lower = True,
                 include_lengths=True,
                 sequential=True,
                 use_vocab=True
                )
    
    LABEL = Field(sequential=False, use_vocab=False)

    return TEXT, LABEL

def get_dataset(TEXT, LABEL):

    lyric_datafield = [("lyrics", TEXT),
                     ("genre", LABEL)]


    train, test = TabularDataset.splits('data/',
                                     train='training.csv',
                                     test ='testing.csv',
                                     format ='csv',
                                     fields = lyric_datafield,
                                     skip_header=True)

    return train, test

def get_vocab(trn, TEXT, pretrained_bool, dimensions):
    
    print('Pretrained: {}'.format(pretrained), 'Dimensions: {}'.format(dimensions))
    if pretrained=='y':
        print('Pretrained: True')
        if dimensions == 50:
            print('Loading glove.6B.50d Vectors')
            TEXT.build_vocab(trn, max_size=100000, vectors="glove.6B.50d")
        elif dimensions == 100:
            print('Loading glove.6B.100d Vectors')
            TEXT.build_vocab(trn, max_size=100000, vectors="glove.6B.100d")
        elif dimensions == 200:
            print('Loading glove.6B.200d Vectors')
            TEXT.build_vocab(trn, max_size=100000, vectors="glove.6B.200d")
        elif dimensions == 300:
            print('Loading glove.6B.300d Vectors')
            TEXT.build_vocab(trn, max_size=100000, vectors="glove.6B.300d")

        return TEXT

    else:
        TEXT.build_vocab(trn)
        return TEXT

def get_iterators(train_dataset, test_dataset, batch_size):

        traindl, testdl = BucketIterator.splits(datasets=(train_dataset,test_dataset),
                     batch_sizes=(batch_size,1),
                     sort_key=lambda x: len(x.lyrics),
                     device=None,
                     sort_within_batch=True,
                     repeat=False)

        return traindl, testdl

def save_dataloaders(train_loader, test_loader, model_name):
    
    directory = 'dataloaders/'
    if os.path.exists(directory) == False:
        os.mkdir(directory)

    train = '{}_training_loader.pkl'.format(model_name)
    test = '{}_test_dataset.pkl'.format(model_name)

    for i in zip([train_loader, test_loader], [train,test]):
        with open(directory+i[1], 'wb') as file:
            dill.dump(i[0],file)

def train(model, nr_epochs, device, training_dl):
    CUDA_LAUNCH_BLOCKING=1
    
    model.train()
    model = model.to(device)
    epoch_nr = 0
    EPOCH = list(range(nr_epochs))
    avg_loss = 0
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch in tqdm(EPOCH):
        
        epoch_nr += 1
        epoch_loss = []
        count = 0
        
        for i in tqdm(training_dl):
           
            
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
    return model

def save_model(model, model_name):
    
    directory = 'trained_models/'
    if os.path.exists(directory) == False:
        os.mkdir(directory)
    torch.save(model.state_dict(), 'trained_models/{}.pt'.format(model_name))

def evaluate(trained_model, testdl):
    
    correct = 0
    count = 0
    for i in tqdm(testdl):

        x = i[0][0]
        y = i[1]
       
        lengths = i[0][1]
        
        try:
            predictions = trained_model(x, lengths)
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
    
    if args.gpu == 'y':
        device = 'cuda:00'
    else:
        device = 'cpu'
        
    pretrained = args.pretrained
    dim = args.dim
    x_field, y_field = get_fields()
    
    print('Generating Dataset')
    
    train_dataset, test_dataset = get_dataset(x_field, y_field)
    
    print('Generating Vocab')

    x_field = get_vocab(train_dataset, x_field, pretrained, dim)
    traindl, testdl = get_iterators(train_dataset, test_dataset, args.batch_size)
    vocab_size = len(x_field.vocab)
    embedding_dim = dim
    n_out = 5
    EPOCHS = args.nr_epochs
    n_hidden = 128
    train_batch_it = BatchGenerator(traindl, 'lyrics', 'genre')
    test_batch_it = BatchGenerator(testdl, 'lyrics', 'genre')

    print('Generating Model')

    model = Rnn_Gru(vocab_size, embedding_dim,
                                    n_hidden, n_out, x_field.vocab.vectors,
                                    pretrained, device).to(device)

    print(model)
    print('Training')
    
    trained_model = train(model, EPOCHS, device, train_batch_it)
    save_model(trained_model, args.model_name)

    print('Evaluating Model')
    evaluate(trained_model, test_batch_it)
