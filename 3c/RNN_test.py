# Given model: python3 RNN_test.py --sequence_length_train 100 --sequence_length_test 500 --training_mode 0
# Custom1: python3 RNN_test.py --sequence_length_train 150 --sequence_length_test 500 --training_mode 0
# Custom2: python3 RNN_test.py --sequence_length_train 100 --sequence_length_test 500 --training_mode 1
# Custom3: python3 RNN_test.py --sequence_length_train 100 --sequence_length_test 500 --training_mode 2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist
import pandas as pd
import argparse
import time
import os
import sys
import io

from RNN_model import RNN_model

parser = argparse.ArgumentParser(description='NLP_test')
parser.add_argument('--sequence_length_train', '-t', default=100, type=int, help='training sequence length')
parser.add_argument('--sequence_length_test', '-s', default=500, type=int, help='testing sequence length')
parser.add_argument('--training_mode', '-m', default=0, type=int, help='choose the training mode')
args = parser.parse_args()
print('sequence length for training: %d | sequence length for testing: %d | training mode: %d' %
        (args.sequence_length_train, args.sequence_length_test, args.training_mode))


#imdb_dictionary = np.load('../preprocessed_data/imdb_dictionary.npy')
vocab_size = 8000

x_test = []
with io.open('../preprocessed_data/imdb_test.txt','r',encoding='utf-8') as f:
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

model = torch.load('rnn_len_{}_mode_{}.model'.format(args.sequence_length_train, args.training_mode))

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
        y_input = y_test[I_permutation[i:i+batch_size]]

        data = Variable(torch.LongTensor(x_input)).cuda()
        target = Variable(torch.FloatTensor(y_input)).cuda()

        with torch.no_grad():
            loss, pred = model(data,target,train=False)

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
save.to_csv('save_rnntest_len_{}_mode_{}.csv'.format(args.sequence_length_train, args.training_mode), index=False)
