from typing import Counter
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torchtext
import string
from collections import Counter
maxlen = 2000
PAD_TOK = '<pad>'
UNK_TOK = '<unk>'
class TextCNN(nn.Module):
    def __init__(self, vocab_size):
        filter_num = 100
        embedding_dim = 64
        kernel_list = [3, 4, 5]
        super(TextCNN, self).__init__()
        self.emed1 = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, filter_num, (kernel, embedding_dim)),
                          nn.ReLU(),
                          nn.MaxPool2d((maxlen - kernel + 1, 1)))
            for kernel in kernel_list
        ])
        self.flat = nn.Flatten()
        self.dropout = nn.Dropout(0.1)
        self.lin1 = nn.Linear(filter_num * len(kernel_list), 64)
        self.relu4 = nn.ReLU()
        self.lin2 = nn.Linear(64, 32)
        self.relu5 = nn.ReLU()
        self.lin3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1 * 317
        x = self.emed1(x)
        # print(x.shape)
        x = x.unsqueeze(1)
        # 1 * 64 * 317
        # print(x.shape)
        out = [conv(x) for conv in self.convs]
        x = torch.cat(out, dim = 1)
        # print(x.shape)
        # x = self.dropout1(x)
        # print(x.shape)
        x = self.flat(x)
        x = self.dropout(x)
        # print(x.shape)
        x = self.relu4(self.lin1(x))
        x = self.relu5(self.lin2(x))
        x = self.sigmoid(self.lin3(x))
        return x

def train(epoch, f):
    print("Training... Epoch = %d" % epoch)
    x_loader = []
    # fo = open("out.txt", "r+")
    for data, target in zip(traindata, trainlabel):
        # print(data, target)
        wd = [word2idx[i] for i in tokenizer(data)]
        if len(wd) <= maxlen:
            t = len(wd)
            for i in range(maxlen - t):
                wd.append(word2idx['<pad>'])
        else:
            wd = wd[0:maxlen]
        idata = torch.LongTensor([wd])
        # print(idata.shape)
        x = model(idata)
        tar = None
        if target == 'neg':
            tar = torch.Tensor([[0.]])
        else:
            tar = torch.Tensor([[1.]])
        # fo.write(str(x))
        # fo.write(str(tar))
        if f:
            print(x, tar)
        loss = nllloss(x, tar)
        if f:
            print(loss)


        optimizer4nn.zero_grad()
        loss.backward()
        optimizer4nn.step()
        # x_loader.append(x)

    # feat = torch.cat(x_loader, 0)
    # fo.write("finish")
    # fo.close()
    

train_iter = torchtext.datasets.IMDB(split='train', root='.data')
# train_loader = DataLoader(trainset, batch_size=128, num_workers=4)
tokenizer = torchtext.data.get_tokenizer('basic_english')
trainsetdata = list(train_iter)
# print(trainset)
traindata = []
trainlabel = []
for (label, data) in trainsetdata:
    traindata.append(data)
    trainlabel.append(label)
for i in range(len(traindata)):
    t = random.randint(i, len(traindata) - 1)
    tmp = traindata[i]
    traindata[i] = traindata[t]
    traindata[t] = tmp
    tmp = trainlabel[i]
    trainlabel[i] = trainlabel[t]
    trainlabel[t] = tmp
# traindata = traindata[0:1000]
# trainlabel = trainlabel[0:1000]
vocab = list(set(tokenizer(" ".join(traindata) + " " + UNK_TOK + " " + PAD_TOK)))
# vocab = torchtext.vocab.build_vocab_from_iterator(word_list)
# vocab = list(set(word_list))
# vocab = torchtext.vocab.Vocab(Counter(specials+word_list))
print(len(vocab))
word2idx = {w: i for i, w in enumerate(vocab)}
model = TextCNN(len(vocab) + 1)
nllloss = nn.BCELoss()
optimizer4nn = optim.SGD(model.parameters(),lr=0.001)
sheduler = lr_scheduler.StepLR(optimizer4nn,20,gamma=0.8)
for epoch in range(1000):
    # print optimizer4nn.param_groups[0]['lr']
    train(epoch+1, 0)
    sheduler.step()