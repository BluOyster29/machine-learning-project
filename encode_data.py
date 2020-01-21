from ProjDataset import ProjDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch, pickle, random
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
import torch.optim as optim
from torch import nn
from GRUNetwork import RNN_GRU

def load_data(txt, labels):
    x = []
    with open(txt, 'r') as file:
        for i in file:
            x.append(i.split(' ')[:-1])
    y = []

    with open(labels, 'r') as file:
        for i in file:
            y.append(int(i))
    return x,y

def get_vocab(text_data):
    vocab = []
    for i in text_data:
        vocab += [x for x in i]
    return vocab

def encode(text_data,vocab):
    text_encoded = []
    for i in tqdm(text_data):
        encoded = []
        for word in i[:174]:
            if word in vocab:
                encoded.append(vocab[word])
            else:

                encoded.append(random.randint(0,len(vocab)))
        text_encoded.append(torch.LongTensor(encoded))

    return pad_sequence(text_encoded, batch_first=True, padding_value=0)

def output_dl(data, filename):
    with open('dataloaders/' + filename, 'wb') as file:
        pickle.dump(data, file)

def train(model, train_loader, vocab_size, device, nr_of_epochs, batch_size):

    print('Training')
    epoch_nr = 0
    EPOCH = list(range(10))
    tenp = round(len(train_dl,) / 10)
    for epoch in tqdm(EPOCH):
        epoch_nr += 1
        epoch_loss = []
        h = model.init_hidden(200)
        count = 0
        percent = 0
        for (x,y) in tqdm(train_dl):
            count +=1
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            h = h.data
            out, h = model(x, h)
            loss = criterion(out, y.long())

            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()

            avg_loss = sum(epoch_loss) / len(epoch_loss)
            print("Average loss at epoch %d: %.7f" % (epoch_nr, avg_loss))
        return model

def main():

    x_train, y_train = load_data('data/pre_processed/x_train_lemm.txt',
                                 'data/pre_processed/y_train_lemm.txt')
    x_test, y_test = load_data('data/pre_processed/x_test_lemm.txt',
                                 'data/pre_processed/y_test_lemm.txt')


    train_vocab = get_vocab(x_train)
    int2wor = dict(enumerate(set(train_vocab)))
    wor2int = {a : b for b, a in int2wor.items()}
    print('Encoding')
    train_encoded = encode(x_train,wor2int)
    test_encoded = encode(x_test, wor2int)
    print('Outputting Csv')
    '''train_df = output_csv(train_lemm ,y_train, 'data/processed/train_processed.csv')
    test_df = output_csv(test_lemm, y_test , 'data/processed/test_processed.csv')'''
    train_dataset = ProjDataset(train_encoded, y_train)
    test_dataset = ProjDataset(test_encoded, y_test)
    train_dl = DataLoader(train_dataset, batch_size=200, shuffle=True)
    '''print('Outputting Dataloader')
    output_dl(train_dl, 'train_dataloader.pkl')
    output_dl(test_dataset, 'test_dataset.pkl')'''

    model = RNN_GRU(vocab_size=len(wor2int), seq_len=174,
                   input_size=174, hidden_size=300,
                   num_layers=2, output_size=10,
                   device=torch.device('cpu'), dropout=0.01)

    print(model)
    return train_dl, test_dataset, train_vocab, wor2int, model

if __name__== '__main__':
    main()