import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(nn.Linear(2 * hidden_dim, 64), nn.ReLU(True), nn.Linear(64, 1))

    def forward(self, lstm_output):
        output = self.projection(lstm_output) / 8
        weights = F.softmax(output.squeeze(-1), dim=1)
        output = (lstm_output * weights.unsqueeze(-1)).sum(dim=1) * 0.001
        return output


class BiLSTM(nn.Module):
    def __init__(self, embeddings, hidden_dim, mid_dim, num_layers, max_len=600, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.emb = nn.Embedding(embeddings.weight.size(0), embeddings.weight.size(1), padding_idx=0)
        self.emb.weight = embeddings.weight
        self.emb.weight.requires_grad = False
        self.input_dim = embeddings.weight.size(1)
        self.hidden_dim = hidden_dim
        self.mid_dim = mid_dim
        self.num_layer = num_layers
        self.sen_len = max_len
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            dropout=dropout, batch_first=True, bidirectional=True)
        # self.linear1 = nn.Linear(hidden_dim * num_layers + 1, mid_dim)
        self.linear1 = nn.Linear(2 * hidden_dim * num_layers + 1, mid_dim)
        # self.linear1 = nn.Linear(2 * hidden_dim * num_layers, mid_dim)
        self.act = nn.Sigmoid()
        # self.act = nn.ReLU(True)
        self.linear2 = nn.Linear(mid_dim, 3)

    def attention(self, sen, h_state, batch_size):
        h_state = h_state.permute(1, 0, 2)
        h_state = h_state.contiguous().view(batch_size, 2 * self.hidden_dim, self.num_layer)
        # h_state = h_state.contiguous().view(batch_size, self.hidden_dim, self.num_layer)
        attention_weight = torch.bmm(sen, h_state)
        attention_weight = F.softmax(attention_weight, 1)
        sen = torch.bmm(sen.permute(0, 2, 1), attention_weight)
        return sen

    def forward(self, sen, sen_lengths):
        sen = self.emb(sen)
        batch_size = sen.size(0)
        sen, (h_state, c_state) = self.lstm(sen)
        sen = self.attention(sen, h_state, batch_size)
        # sen = self.attention(sen)
        sen = self.linear1(torch.cat((sen.view(batch_size, -1), sen_lengths.view(batch_size, 1)), 1))
        # sen = self.linear1(sen.view(batch_size, -1))
        sen = self.act(sen)
        sen = self.linear2(sen)
        return F.softmax(sen, 1)
