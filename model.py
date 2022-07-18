import os
import re
import math
import numpy
import cProfile

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
nltk.download("stopwords")

from googletrans import Translator

import torch
from torch import nn
from torch import optim
from torch._C import device
from torch.nn import functional
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import matplotlib.pyplot as plt

BATCH_SIZE = 64
SEQ_LEN = 40

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
print("using device : " + str(device))
torch.cuda.empty_cache()

#tweet data class
class Record():
    def __init__(self, text, label):
        self.text = text
        self.label = label
        self.image = 0

class TextManipulator():
    def __init__(self):
        self.translator = Translator()
        self.stem = SnowballStemmer("english")
        self.stopwords = set(stopwords.words("english"))
        self.emojis = re.compile(u'[' u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF' u'\u2600-\u26FF\u2700-\u27BF]+', re.UNICODE)

    def preprocess(self, path, start_index=1, cache="cache.txt"):
        file = open(path, "r", encoding="utf-8")
        cache_file = open(cache, "a", encoding="utf-8")

        i = 1
        for record in file.readlines()[start_index:]:
            lst = record.split("\t")

            clean_text = self.clean(lst[1])
            label = lst[6]

            cache_file.write(clean_text + "\t" + label)
            
            print(i, end="\r")
            i += 1

            #reinitialise translator to avoid api limit
            self.translator = Translator()
            
        file.close()
        cache_file.close()

    def create_vocab(self, path):
        vocab = []
        
        cache1 = open("cache.txt", "r", encoding="utf-8")
        cache2 = open("testcache.txt", "r", encoding="utf-8")

        for record in cache1.readlines():
            lst = record.split("\t")
            for word in lst[0].split():
                if word not in vocab:
                    vocab.append(word)
        
        for record in cache2.readlines():
            lst = record.split("\t")
            for word in lst[0].split():
                if word not in vocab:
                    vocab.append(word)

        file = open(path, "w", encoding="utf-8")
        file.write(",".join(vocab))
        file.close()

        cache1.close()
        cache2.close()

    def clean(self, text):
        #remove url and tags
        text = re.sub(r"http: ?[\&\/\\.\d\w]*", "", text)
        text = re.sub(r"@\w*", "", text)

        #remove whitespace characters
        text = re.sub(r"\s", " ", text)
        text = re.sub(r"\\n", " ", text)

        text = self.emojis.sub("", text)

        #translate text
        text = self.translate(text).lower()

        #remove punctuation
        text = re.sub(r"[-\.\!\?\,()\"\“\”#\/;:\[\]<>\^{}`+=~|\_\«\»…\*¬]", " ", text)
        text = re.sub(r"\&", "and", text)

        #remove stopwords and get word stems
        text = " ".join([self.stem.stem(word) for word in text.split() if word not in self.stopwords])

        return text

    def translate(self, text:str):
        return self.translator.translate(text, src="auto", dest="en").text

class MEData(Dataset):
    def __init__(self, path, vocab_path):
        self.data = []
        self.vocab = [""]
        self.totals = [0, 0]

        file = open(path, "r", encoding="utf-8")
        for record in file.readlines():
            lst = record.split("\t")

            #sum the classes so the loss function can be weighted
            if lst[1][:4] == "real":
                self.totals[0] += 1
            else:
                self.totals[1] += 1

            self.data.append(Record(lst[0], lst[1]))
        file.close()

        vocab_file = open(vocab_path, "r", encoding="utf-8")
        for word in vocab_file.readline().split(","):
            self.vocab.append(word)
        vocab_file.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        r = self.data[idx]

        #sanitise text and return
        return (self.word2vec(r.text), self.encode_label(r.label))

    def encode_label(self, label:str):
        if label[:4] == "real":
            return 0
        else:
            return 1

    def word2vec(self, text:str):
        indexes = []

        for word in text.split():
            indexes.append(self.vocab.index(word))

        i = numpy.asarray(indexes)
        return (
            numpy.pad(i, (SEQ_LEN - len(indexes), 0), mode="constant", constant_values=0),
            numpy.pad(numpy.ones_like(i), (SEQ_LEN - len(indexes), 0), mode="constant", constant_values=0)
            )

#positional encoding from pytorch.org
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class Net(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.dmodel = 512

        self.src_embed = nn.Embedding(vocab_size, self.dmodel)
        self.encode = PositionalEncoding(self.dmodel, vocab_size)
        
        self.layer = nn.TransformerEncoderLayer(self.dmodel, 4, 1024, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.layer, num_layers=4)

        self.linear = nn.Linear(self.dmodel, 1)
        #self.linear = nn.Linear(vocab_size, 1)

    def forward(self, x, mask):
        x = self.src_embed(x.int()) * math.sqrt(self.dmodel)
        x = self.encode(x)

        x = self.transformer(x.float(), src_key_padding_mask=mask)

        #reduce output dims
        x = x.mean(dim=1)
        
        return self.linear(x)

def train(epochs:int, warmup:int):
    net.train()
    
    loss_list = []
    for epoch in range(epochs):
        epoch_loss = 0.

        for idx, ((src, mask), exp) in enumerate(train_loader):
            net.zero_grad() #reset gradient between batches

            src = src.float().to(device)
            exp = exp.float().to(device)
            mask = mask.bool().to(device)

            output = net(src, mask).flatten(start_dim=0)

            loss = functional.binary_cross_entropy_with_logits(output, exp, pos_weight=torch.tensor(0.6))
            loss.backward()

            print("BATCH: " + str(idx), end="\r")
            
            optimizer.step()
            epoch_loss += loss.item()
        
        print("EPOCH "+str(epoch)+" : total loss "+str(epoch_loss)+", LR: "+str(optimizer.param_groups[0]["lr"]))
        loss_list.append(float(epoch_loss))
        
        if epoch < warmup:
            warmup_sched.step()
        else:
            scheduler.step()

    torch.save({"state": net.state_dict(), "optim": optimizer.state_dict()}, r".\model.pt")

    plt.plot(loss_list)
    plt.show()

def test():
    net.eval()
    print("="*25)

    saved = torch.load(r".\model.pt")
    net.load_state_dict(saved["state"])
    optimizer.load_state_dict(saved["optim"])

    p = TextManipulator()
    test_dataset = MEData("testcache.txt", "vocab.txt")

    tp, fp, tn, fn = 0, 0, 0, 0

    for r in test_dataset.data:
        text = test_dataset.word2vec(r.text)
        source = torch.as_tensor(text[0]).float().to(device)
        mask = torch.as_tensor(text[1]).float().to(device)
        expect = torch.as_tensor(test_dataset.encode_label(r.label)).float().to(device)

        with torch.no_grad():
            output = net(source.unsqueeze(0), mask.unsqueeze(0))
            output = torch.sigmoid(output)
        
            if output.item() >= 0.5:
                if expect.item() == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if expect.item() == 0:
                    tn += 1
                else:
                    fn += 1
        
    # CALCULATE F1
    print("tp, fp, tn, fn =", tp, fp, tn, fn)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    print("F1: " + str((2 * p * r) / (p + r)))


print("Preparing data...")
mnpl = TextManipulator()
#mnpl.preprocess("trainset.txt", start_index=1, cache="cache.txt")       #sanitize and store datasets
#mnpl.preprocess("testset.txt", start_index=1, cache="testcache.txt")
#mnpl.create_vocab("vocab.txt")                                          #create vocab file

train_dataset = MEData("cache.txt", "vocab.txt")
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

net = Net(len(train_dataset.vocab))         #setup network and hyperparameters
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.0001)

warmup_sched = optim.lr_scheduler.StepLR(optimizer, 1, 1.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.9)

#train(40, warmup=5)     #train for 40 epochs w/ 5 epochs of lr warmup

test()                  #test and produce f1
