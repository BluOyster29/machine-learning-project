from torch import nn
import torch

class RNN_GRU(nn.Module):
    def __init__(self, vocab_size, seq_len, input_size, hidden_size, num_layers, output_size, device, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.emb = nn.Embedding(vocab_size, input_size).to(device)
        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers=self.num_layers, batch_first=True, dropout=dropout).to(device)
        self.fc = nn.Linear(hidden_size * seq_len, output_size).to(device)

    def forward(self, sequence, hidden_layer):
        output = self.emb(sequence).to(device)
        hidden_layer = hidden_layer.to(self.device)
        output, hidden_layer = self.gru(output, hidden_layer)
        output = output.contiguous().view(-1, self.hidden_size *
                                          len(sequence[0]))
        output = self.fc(output).to(device)

        return output, hidden_layer
    def forward(self, sequence, hidden_layer):
        output = self.emb(sequence)
        output, hidden_layer = self.gru(output, hidden_layer)
        output = output.contiguous().view(-1, self.hidden_size *
                                          len(sequence[0]))
        output = self.fc(output)
        return output, hidden_layer

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).float().to(self.device)
