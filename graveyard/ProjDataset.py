from torch.utils.data import Dataset

class ProjDataset(Dataset):

    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

    def __getitem__(self, index):
        return (self.train_x[index], self.train_y[index])

    def __len__(self):
        return len(self.train_x)
