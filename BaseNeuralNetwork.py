import torch
import torch.nn as nn
import torch.nn.functional as F
from vocab_utils import perform_lookups, batch_data

class BaseNeuralNetwork(nn.Module):

    def __init__(self, word2id, id2word, idf):
        super(BaseNeuralNetwork, self).__init__()

        self.word2id = word2id
        self.id2word = id2word
        self.idf = idf
        self.embeddings = nn.Embedding(num_embeddings=len(self.word2id), embedding_dim=256, padding_idx=0)
        self.layer1 = nn.Linear(256, 100)
        self.layer2 = nn.Linear(100, 150)
        self.layer3 = nn.Linear(150, 2)

    def forward(self, x):

        output = self.embeddings(x)
        #print(output.size()) 
        # seq_len, batch, embed_size
        if self.idf is None:
            output = torch.sum(output, dim=0)
        else: 
            output = output.permute(1,2,0)
            nums = x.permute(1,0).contiguous() # batch, seq_len
            batch_size, seq_len = nums.size()
            tf_idf_scorer = []
            for i in range(batch_size):
                current_scores = []
                occurrences = {}
                for j in range(seq_len):
                    idxNum = nums.data.tolist()[i][j]
                    if idxNum not in occurrences:
                        occurrences[idxNum] = 0
                    occurrences[idxNum] += 1
                for j in range(seq_len):
                    idxNum = nums.data.tolist()[i][j]
                    if idxNum == 0:
                        current_scores.append(0)
                    else:
                        tf_part = occurrences[idxNum]
                        idf_part = self.idf[self.id2word[idxNum]]
                        current_scores.append(tf_part*idf_part)
                tf_idf_scorer.append(current_scores)
            tf_idf_scores = torch.tensor(tf_idf_scorer)
            tf_idf_scores = tf_idf_scores.unsqueeze(2)
            output = torch.bmm(output, tf_idf_scores).squeeze()

        output = F.relu(self.layer1(output))
        output = F.relu(self.layer2(output)) 
        softmax = torch.nn.Softmax()
        output = softmax(self.layer3(output))
        return output