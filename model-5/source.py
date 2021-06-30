import torch, torchtext
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from  torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets, transforms
import torch.distributed as dist
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import pickle

PAD_TOK = '<pad>'
UNK_TOK = '<unk>'
SOS_TOK = '<SOS>'
EOS_TOK = '<EOS>'
class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        # self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        # print(features.shape, embeddings.shape)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

class Net(nn.Module):
    def __init__(self, vocab_size):
        super(Net, self).__init__()
        self.encoder = EncoderCNN(256)
        self.decoder = DecoderRNN(256, 512, vocab_size, 1)
    
    def forward(self, x, captions, lengths):
        features = self.encoder(x)
        output = self.decoder(features, captions, lengths)
        return output


def train(epoch):
    print("Training... Epoch = %d" % epoch)
    for i, (images, captions) in enumerate(data_loader):
        # print(captions)
        idata = []
        for cap in captions:
            idata.append(vocab(SOS_TOK))
            for word in tokenizer(cap[0]):
                idata.append(vocab(word))
            idata.append(vocab(EOS_TOK))
        captions = torch.LongTensor([idata])
        lengths = torch.Tensor([len(idata)])
        target = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        # print(images.shape, captions.shape, lengths.shape)
        pred = model(images, captions, lengths)
        loss = nllloss(pred, target)

        optimizer4nn.zero_grad()
        loss.backward()
        optimizer4nn.step()

transform = transforms.Compose([ 
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
coco = datasets.CocoCaptions('.data', './.ann/captions_val2014.json', transform)
data_loader = DataLoader(coco, batch_size=1, shuffle=True)
tokenizer = torchtext.data.get_tokenizer('basic_english')
# Model
model = Net(len(vocab))

# NLLLoss
nllloss = nn.NLLLoss()


# optimzer4nn
optimizer4nn = optim.SGD(model.parameters(),lr=0.001,momentum=0.9, weight_decay=0.0005)
sheduler = lr_scheduler.StepLR(optimizer4nn,20,gamma=0.8)

for epoch in range(100):
    # print optimizer4nn.param_groups[0]['lr']
    train(epoch+1)
    sheduler.step()