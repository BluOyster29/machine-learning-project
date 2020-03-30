from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch import nn
import torch

class Rnn_Gru(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_hidden, n_out,
                    pretrained_vec, pretrained, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.device = device
        self.pretrained = pretrained
        if self.pretrained == True:
            self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)
            self.emb.weight.data.copy_(pretrained_vec) # load pretrained vectors
            self.emb.weight.requires_grad = False # make embedding non trainable
        else:
            self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.gru = nn.GRU(self.embedding_dim, self.n_hidden)
        self.out = nn.Linear(self.n_hidden*2, self.n_out)

    def forward(self, seq, lengths):
        bs = seq.size(1)
        self.h = self.init_hidden(bs)
        seq = seq.transpose(0,1).to(self.device)

        embs = self.emb(seq)
        embs = embs.transpose(0,1)
        embs = pack_padded_sequence(embs, lengths, enforce_sorted=False)

        gru_out, self.h = self.gru(embs, self.h)

        gru_out, lengths = pad_packed_sequence(gru_out)

        avg_pool = F.adaptive_avg_pool1d(gru_out.permute(1,2,0),1).view(bs,-1)
        max_pool = F.adaptive_max_pool1d(gru_out.permute(1,2,0),1).view(bs,-1)
        outp = self.out(torch.cat([avg_pool,max_pool],dim=1))
        return F.log_softmax(outp, dim=-1)

    def init_hidden(self, batch_size):
       return torch.zeros((1,batch_size,self.n_hidden)).cuda().to(self.device)
