import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

def preprocess(raw_text):
    tokens = word_tokenize(raw_text)
    stop = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    porter = PorterStemmer()
    raw_text = [porter.stem(word) for word in tokens if word not in stop and word.isalpha()]
    return raw_text

def prepareData(text):
    data = []
    for i in range(2,len(text)-2):
        context = [text[i-2], text[i-1], text[i+1], text[i+2]]
        target = text[i]
        data.append((context,target))
    return data

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out)
        return log_probs
    
    def init_embedding(self,inputs):
        return self.embeddings(inputs).view(1,-1)

