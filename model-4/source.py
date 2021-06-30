from typing import Counter
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from  torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torchtext import datasets, data

n_hidden = 128
n_step = 2000

class Seq2Seq(nn.Module):
    def __init__(self, n_class):
        super(Seq2Seq, self).__init__()
        self.embed1 =nn.Embedding(len(ivocab), n_class)
        self.embed2 =nn.Embedding(len(ivocab), n_class)
        self.encoder = nn.RNN(input_size=n_class, num_layers=3, hidden_size=n_hidden, dropout=1, batch_first=True) # encoder
        self.decoder = nn.RNN(input_size=n_class, num_layers=3, hidden_size=n_hidden, dropout=1, batch_first=True) # decoder
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, enc_input, enc_hidden, dec_input):
        print(enc_input.shape, dec_input.shape)
        enc_input = self.embed1(enc_input)
        dec_input = self.embed2(dec_input)

        _, h_t = self.encoder(enc_input, enc_hidden)
        outputs, _ = self.decoder(dec_input, h_t)

        model = self.fc(outputs)
        return model

def train(epoch):
    print("Training... Epoch = %d" % epoch)
    for input, target in zip(iv, ov):
        idata = [iword2idx[i] for i in tokenizer1(input)]
        tmp = len(idata)
        for i in range(tmp, n_step - 1):
            idata.append(iword2idx[PAD_TOK])
        if tmp > n_step:
            print('bigger', tmp)
        ddata = [iword2idx[SOS_TOK]] + idata
        tmp = len(ddata)
        for i in range(tmp, n_step):
            ddata.append(iword2idx[PAD_TOK])
        ddata = torch.LongTensor([ddata])
        idata = torch.LongTensor([idata + [iword2idx[PAD_TOK]]])
        h_0 = torch.zeros(3, 1, n_hidden)
        # print(idata.shape, idata)
        pred = model(idata, h_0, ddata)
        target = [oword2idx[i] for i in tokenizer2(target)] + [oword2idx[EOS_TOK]]
        tmp = len(target)
        for i in range(tmp, n_step):
            target.append(oword2idx[PAD_TOK])
        target = torch.LongTensor(target)
        print(target.shape, target)
        loss = nllloss(pred[0], target)

        optimizer4nn.zero_grad()
        loss.backward()
        optimizer4nn.step()

iwslt, valid, test = datasets.IWSLT2017(root='.data', split=('train', 'valid', 'test'), language_pair=('de', 'en'))
data_loader = DataLoader(iwslt, batch_size=1)
iv = []
ov = []
for d in data_loader:
    iv.append(d[0][0])
    ov.append(d[1][0])
# iv = ['Er sagte: "Man möchte meinen, dass die Absicht zum Glücklichsein nicht Teil des Schöpfungsplans ist."']
# ov = ['He said, "One feels inclined to say that the intention that man should be happy is not included in the plan of creation."']

tokenizer2 = data.get_tokenizer('basic_english')
tokenizer1 = data.get_tokenizer('spacy', 'de_core_news_sm')
PAD_TOK = '<pad>'
UNK_TOK = '<unk>'
SOS_TOK = '<SOS>'
EOS_TOK = '<EOS>'
ivocab = list(set(tokenizer1(" ".join(iv)) + [UNK_TOK, PAD_TOK, SOS_TOK, EOS_TOK]))
ovocab = list(set(tokenizer2(" ".join(ov)) + [UNK_TOK, PAD_TOK, SOS_TOK, EOS_TOK]))
print(len(ivocab), len(ovocab))
iword2idx = {w: i for i, w in enumerate(ivocab)}
oword2idx = {w: i for i, w in enumerate(ovocab)}
nclass = len(ovocab)
# Model
model = Seq2Seq(nclass)

# NLLLoss
nllloss = nn.NLLLoss()


# optimzer4nn
optimizer4nn = optim.SGD(model.parameters(),lr=0.001,momentum=0.9, weight_decay=0.0005)
sheduler = lr_scheduler.StepLR(optimizer4nn,20,gamma=0.8)

for epoch in range(100):
    # print optimizer4nn.param_groups[0]['lr']
    train(epoch+1)
    sheduler.step()