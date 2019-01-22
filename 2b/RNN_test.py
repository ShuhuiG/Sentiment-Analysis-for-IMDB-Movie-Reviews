
# This is for the given model

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist
import pandas as pd

import time
import os
import sys
import io

from RNN_model import RNN_model

glove_embeddings = np.load('../preprocessed_data/glove_embeddings.npy')
vocab_size = 100000

x_test = []
with io.open('../preprocessed_data/imdb_test_glove.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_test.append(line)
y_test = np.zeros((25000,))
y_test[0:12500] = 1

vocab_size += 1

model = RNN_model(500)
model.cuda()
model = torch.load('rnn.model')

batch_size = 200
no_of_epochs = 10

L_Y_test = len(y_test)

test_accu = []
test_loss = []
time_elapsed_save = []
sequence_length_save = []
for epoch in range(no_of_epochs):
    # ## test
    model.eval()

    epoch_acc = 0.0
    epoch_loss = 0.0
    epoch_counter = 0

    time1 = time.time()

    sequence_length = (epoch+1)*50
    sequence_length_save.append(sequence_length)

    I_permutation = np.random.permutation(L_Y_test)

    for i in range(0, L_Y_test, batch_size):
        x_input2 = [x_test[j] for j in I_permutation[i:i+batch_size]]
        x_input = np.zeros((batch_size,sequence_length),dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if(sl < sequence_length):
                x_input[j,0:sl] = x
            else:
                start_index = np.random.randint(sl-sequence_length+1)
                x_input[j,:] = x[start_index:(start_index+sequence_length)]
        x_input = glove_embeddings[x_input]
        y_input = y_test[I_permutation[i:i+batch_size]]

        data = Variable(torch.FloatTensor(x_input)).cuda()
        target = Variable(torch.FloatTensor(y_input)).cuda()

        with torch.no_grad():
            loss, pred = model(data,target)

        prediction = pred >= 0.0
        truth = target >= 0.5
        acc = prediction.eq(truth).sum().cpu().data.numpy()
        epoch_acc += acc
        epoch_loss += loss.data.item()
        epoch_counter += batch_size

    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    test_accu.append(epoch_acc)
    test_loss.append(epoch_loss)

    time2 = time.time()
    time_elapsed = time2 - time1
    time_elapsed_save.append(time_elapsed)

    print("  ", "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss)

save = pd.DataFrame({"elapsed time" : np.array(time_elapsed_save),
                        "test loss" : np.array(test_loss),
                        "test accuracy" : np.array(test_accu),
                        "sequence length" : np.array(sequence_length_save)})
save.to_csv("save1_rnntest.csv", index=False)
