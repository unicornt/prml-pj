from typing import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from  torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torchtext import datasets, data

PAD_TOK = '<pad>'
UNK_TOK = '<unk>'
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.lin = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout()
        self.flat = nn.Flatten(0,1)

    def forward(self, x, h, c):
        x = self.embed(x)
        x = x.unsqueeze(0)
        print(x.shape)
        x, (hn, cn) = self.lstm(x, (h, c))
        x = x.view(x.size(0)*x.size(1), x.size(2))
        # x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        x = self.lin(x)
        return x, (hn, cn)

def detach(states):
    return [state.detach() for state in states] 
def train(epoch, f):
    print("Training... Epoch = %d" % epoch)
    h = torch.zeros(num_layers, 1, hidden_size)
    c = torch.zeros(num_layers, 1, hidden_size)
    for di in range(0, len(traindata)):
        data = traindata[di]
        # print(data)
        idata = [word2idx[i] for i in tokenizer(data)]
        # print(wd)
        print(len(idata))
        states = (h, c)
        for j in range(len(idata) - seq_len - 1):
            (h, c) = detach(states)
            pred, states = model.forward(torch.LongTensor(idata[j: j + seq_len]), h, c)
            # target = []
            # for d in idata[j + 1: j+ seq_len + 1]:
            #     print(d, vocab_size)
            #     target.append(F.one_hot(torch.LongTensor([d]), vocab_size))
            target = torch.LongTensor(idata[j + 1: j+ seq_len + 1])
            print(pred.shape, target.shape)
            print(pred, target)
            lss = criterion(pred, target)
            print(lss)
            optimizer4nn.zero_grad()
            lss.backward(retain_graph=True)
            optimizer4nn.step()

torch.autograd.set_detect_anomaly(True)
train_iter, validdata, testdata = datasets.WikiText103(root='.data', split=('train', 'valid', 'test'))
traindata = []
for d in list(train_iter):
    traindata.append(d)
tokenizer = data.get_tokenizer('basic_english')
vocab = list(set(tokenizer(" ".join(traindata) + " " + UNK_TOK + " " + PAD_TOK)))
word2idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)
embed_size = 128
hidden_size = 1024
num_layers = 2
seq_len = 30
print(vocab_size)
model = RNNLM(vocab_size, embed_size, hidden_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer4nn = optim.SGD(model.parameters(),lr=0.001)
sheduler = lr_scheduler.StepLR(optimizer4nn,20,gamma=0.8)
for epoch in range(1000):
    # print optimizer4nn.param_groups[0]['lr']
    train(epoch+1, 0)
    sheduler.step()