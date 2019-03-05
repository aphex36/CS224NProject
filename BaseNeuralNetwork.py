import torch
import torch.nn as nn
import torch.nn.functional as F
from vocab_utils import perform_lookups, batch_data

class BaseNeuralNetwork(nn.Module):

    def __init__(self, word2id):
        super(BaseNeuralNetwork, self).__init__()

        self.word2id = word2id
        self.embeddings = nn.Embedding(num_embeddings=len(self.word2id), embedding_dim=256, padding_idx=0)
        self.layer1 = nn.Linear(256, 100)
        self.layer2 = nn.Linear(100, 150)
        self.layer3 = nn.Linear(150, 2)

    def forward(self, x):

        output = self.embeddings(x)
        output = torch.sum(output, dim=0)
        #need to add the embeddings to each other before here

        output = F.relu(self.layer1(output))
        output = F.relu(self.layer2(output))
        softmax = torch.nn.Softmax()
        output = softmax(self.layer3(output))
        return output