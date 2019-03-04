import torch
import torch.nn as nn
import torch.nn.functional as F
from vocab_utils import perform_lookups, batch_data

class BaseNeuralNetwork(nn.Module):

    def __init__(self, word2id):
        super(BaseNeuralNetwork, self).__init__()

        self.word2id = word2id
        self.embeddings = nn.Embedding(num_embeddings=len(self.word2id), embedding_dim=256)
        self.layer1 = nn.Linear(256, 100)
        self.layer2 = nn.Linear(100, 150)
        self.layer3 = nn.Linear(150, 2)

    def forward(self, x):

        output = self.embeddings(x)

        #need to add the embeddings to each other before here

        output = F.relu(self.layer1(output))
        output = F.relu(self.layer2(x))
        output = F.sigmoid(self.layer3(x))

        return output

sampleWordIndex = {"pineapple": 0, "aardvark": 1, "sponge": 2, "spider": 3, "blanket": 4}
example_sentences = ['pineapple aardvark sponge', 'spider blanket aardvark', 'sponge pineapple']

sentences = [["a"], ["hello there"], ["hi there"], ["incoming"], ["full sentence"], ["my"], ["nama"], ["jeff"]]
labels = [["funny"], ["funny"], ["not_funny"], ["funny"], ["not_funny"], ["funny"], ["not_funny"], ["funny"]]
batchSent, batchLab = batch_data(sentences,labels, 2)
print(batchSent)
print(batchLab)
net = BaseNeuralNetwork(sampleWordIndex)