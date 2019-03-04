import torch
import torch.nn as nn
from BaseNeuralNetwork import BaseNeuralNetwork as BNN
import vocab_utils
import numpy as np
import sys
#constants
EPOCHS = 10

word2id = vocab_utils.load_word2Id("vocab.txt", "<pad>")

#data
train_x, train_y = vocab_utils.load_train_data(word2id)
test_x, test_y = vocab_utils.load_test_data(word2id)
train_x = torch.tensor(train_x)
train_y = torch.tensor(train_y)
test_x = torch.tensor(test_x)
test_y = torch.tensor(test_y)

# initialize
bnn = BNN(word2id)
cel = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters())

# train
total_loss = 0.0
for epoch in range(EPOCHS): # for each epoch...
	for i in range(len(train_x)): # for each input/label pair...
		optimizer.zero_grad() # zero out parameter gradients
		preds = bnn(train_x[i]) # predict
		loss = cel(preds, train_y[i]) # calculate loss
		loss.backward()	# backprop
		optimizer.step()

		#statistics
		total_loss += loss.item()
		if i % 2000 == 1999:
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0

print('Finished Training')



# test
ouput = bnn(test_x)

