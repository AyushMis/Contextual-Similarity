import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import nbimporter
import utilities
from utilities import CBOW
from utilities import preprocess, prepareData, make_context_vector

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

#file = open('rural.txt')
#raw_text1 = file.read()
raw_text1 = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold."""

raw_text = preprocess(raw_text1)
data = prepareData(raw_text)

vocab = set(raw_text)
vocab_size = len(vocab)
print(vocab_size)

word_to_ix = {word: i for i, word in enumerate(vocab)}

print(len(data))

losses = []
loss_function = nn.NLLLoss()
model = CBOW(len(vocab), EMBEDDING_DIM, 2*CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

count = 0
for epoch in range(10):
    total_loss = torch.Tensor([0])
    for context, target in data:

        context_var = make_context_vector(context, word_to_ix)
        
        model.zero_grad()

        log_probs = model(context_var)
        #print(log_probs)
        loss = loss_function(log_probs,  autograd.Variable(
            torch.LongTensor([word_to_ix[target]])))

        loss.backward()
        optimizer.step()

        total_loss += loss.data
        count+=1
    if count%20==0:
        print("total loss: ",total_loss)
    losses.append(total_loss)
print(losses)

torch.save(model,'modelNGram.pth')
