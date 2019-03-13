import torch
import torch.nn as nn
import torch.nn.functional as F
from vocab_utils import perform_lookups, batch_data

class RNN(torch.nn.Module):
	def __init__(self, word2id):
		super(RNN, self).__init__()
		self.embed_size = 256
		self.word2id = word2id
		self.hidden_size = 100 #?
		self.seq_len = 1002

		self.embeddings = nn.Embedding(num_embeddings=len(self.word2id), embedding_dim=self.embed_size, padding_idx=0)
		self.LSTM = nn.LSTM(self.embed_size, self.hidden_size, bias=True, bidirectional=True)
		self.fully_connected = nn.Linear(2*self.hidden_size*self.seq_len, 2, bias=False)
		self.softmax = nn.Softmax()

		#dropout?

		#self.ful = nn.Linear(hidden_size, hidden_size/2)

		# TODO: embedding --(embeddings)--> lstm --(hiddens)--> 
		# context-attention --(r)--> fully-connected --
	
	def forward(self, x): # x: seq_len, batch_size

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
		lstm_out = lstm_out.permute(1,0,2)
		lstm_out = lstm_out.contiguous()
		lstm_out = lstm_out.view((batch_size, seq_len * 2*self.hidden_size))
		linear_out = self.fully_connected(lstm_out)
		output = self.softmax(linear_out)

		return output



		#output = torch.sum(output, dim=0)
		
		#need to add the embeddings to each other before here
		#output = F.relu(self.layer1(output))
		#output = F.relu(self.layer2(output))
		#softmax = torch.nn.Softmax(dim=1)
		#output = softmax(self.layer3(output))
		#return output