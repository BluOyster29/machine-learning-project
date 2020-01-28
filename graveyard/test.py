import os, torch, pickle, torch, config, argparse,stats
from torch.utils.data import DataLoader, Dataset
from LangIdentDataset import RTDataset
from tqdm import tqdm
from GRUNetwork import RNN_GRU

def get_args():
    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument("-M", "--trained_model", dest='model_name', type=str,
                        help="select trained model")
    args = parser.parse_args()
    return args

def load_model(path, config):
    model = path
    if config['device'] == 'gpu':
        device = torch.device('cuda:01')
    else:
        device = torch.device('cpu')
    with open(path, 'rb') as input_model:
        data = torch.load(input_model)
    trained_model = RNN_GRU(vocab_size=config['vocab_size'], seq_len=100, input_size=100,
               hidden_size=256, num_layers=2, output_size=10, device=device, dropout=0.0)
    trained_model.load_state_dict(data)
    return trained_model

def get_test_loader(path, model_name):

    with open('{}/{}_testing_loader.pkl'.format(path,model_name), 'rb') as file:
        testing_loader = pickle.load(file)

    return testing_loader

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
    print('Testing Model')
    trained_model = load_model('trained_models/', args.model_name)
    test_dl = get_test_loader('dataloaders', args.model_name)

    evaluate(trained_model, test_dl)
