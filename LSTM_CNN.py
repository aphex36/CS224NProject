import torch
import torch.nn as nn
import torch.nn.functional as F
from vocab_utils import perform_lookups, batch_data

class LSTM_CNN(nn.Module):

    def __init__(self, word2id):
        super(LSTM_CNN, self).__init__()

        self.word2id = word2id
        self.hidden_size = 256
        self.embed_size = 256
        self.embeddings = nn.Embedding(num_embeddings=len(self.word2id), embedding_dim=256, padding_idx=0)
        self.LSTM = nn.LSTM(self.embed_size, self.hidden_size, bias=True, bidirectional=True)
        self.layer_window_2 = nn.Conv1d(in_channels=2*self.hidden_size, out_channels=4, kernel_size=2)
        self.layer_window_3 = nn.Conv1d(in_channels=2*self.hidden_size, out_channels=4, kernel_size=3)
        self.layer_window_4 = nn.Conv1d(in_channels=2*self.hidden_size, out_channels=4, kernel_size=4)
        self.layer_window_5 = nn.Conv1d(in_channels=2*self.hidden_size, out_channels=4, kernel_size=5)
        self.layer1 = nn.Linear(16, 50)
        self.layer2 = nn.Linear(50, 2)

    def forward(self, x):


        #embeddings
        embeddings = self.embeddings(x) #seq_len, batch_size, embed_size
        
        #initializing h, c
        seq_len, batch_size, embed_size = embeddings.size()
        #h0 = torch.randn(seq_len, batch_size, hidden_size)
        #c0 = torch.randn(seq_len, batch_size, hidden_size)
        #hidden = (h0,c0)

        #lstm
        lstm_out, (hn, cn) = self.LSTM(embeddings) 
        # lstm_out: seq_len, batch_size, 2*hidden_size 
        # hn: 2, batch_size, hidden_size

        output = lstm_out.permute(1,2,0) # batch_size, embed_size seq_len
        output = output.contiguous()
        batch_size = output.size()[0] 
        embedding_dim = output.size()[1]

        window_2 = self.layer_window_2(output)
        max_pool_2 = nn.MaxPool1d(window_2.size()[-1])
        output_window_2 = max_pool_2(window_2)


        window_3 = self.layer_window_3(output)
        max_pool_3 = nn.MaxPool1d(window_3.size()[-1])
        output_window_3 = max_pool_3(window_3)

        window_4 = self.layer_window_4(output)
        max_pool_4 = nn.MaxPool1d(window_4.size()[-1])
        output_window_4 = max_pool_4(window_4)

        window_5 = self.layer_window_5(output)
        max_pool_5 = nn.MaxPool1d(window_5.size()[-1])
        output_window_5 = max_pool_5(window_5)

        final_window_2 = torch.squeeze(output_window_2)
        final_window_3 = torch.squeeze(output_window_3)
        final_window_4 = torch.squeeze(output_window_4)
        final_window_5 = torch.squeeze(output_window_5)

        joined_output = torch.cat((final_window_2, final_window_3, final_window_4, final_window_5), -1)
        linear_one = F.relu(self.layer1(joined_output))
        softmax = torch.nn.Softmax()
        final_output = softmax(self.layer2(linear_one))
        return final_output
