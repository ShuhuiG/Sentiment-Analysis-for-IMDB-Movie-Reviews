
# This is for Custom 1

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

class BOW_model2(nn.Module):
    def __init__(self, vocab_size, no_of_hidden_units):
        super(BOW_model2, self).__init__()

        self.embedding = nn.Embedding(vocab_size,no_of_hidden_units)

        #1
        self.fc_hidden1 = nn.Linear(no_of_hidden_units,no_of_hidden_units)
        self.bn_hidden1 = nn.BatchNorm1d(no_of_hidden_units)

        #2
        self.fc_hidden2 = nn.Linear(no_of_hidden_units,no_of_hidden_units)
        self.bn_hidden2 = nn.BatchNorm1d(no_of_hidden_units)

        #3
        self.fc_hidden3 = nn.Linear(no_of_hidden_units,no_of_hidden_units)
        self.bn_hidden3 = nn.BatchNorm1d(no_of_hidden_units)

        self.dropout = torch.nn.Dropout(p=0.5)

        self.fc_output = nn.Linear(no_of_hidden_units, 1)

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, t):

        bow_embedding = []
        for i in range(len(x)):
            lookup_tensor = Variable(torch.LongTensor(x[i])).cuda()
            embed = self.embedding(lookup_tensor)
            embed = embed.mean(dim=0)
            bow_embedding.append(embed)
        bow_embedding = torch.stack(bow_embedding)

        h = F.relu(self.bn_hidden1(self.fc_hidden1(bow_embedding)))
        h = F.relu(self.bn_hidden2(self.fc_hidden2(h)))
        h = F.relu(self.bn_hidden3(self.fc_hidden3(h)))
        h = self.dropout(h)
        #h = self.dropout(F.relu(self.bn_hidden(self.fc_hidden(bow_embedding))))
        h = self.fc_output(h)

        return self.loss(h[:,0],t), h[:,0]
