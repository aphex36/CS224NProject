import torch
import torch.nn as nn
from BaseNeuralNetwork import BaseNeuralNetwork as BNN
import vocab_utils
import numpy as np
import sklearn.metrics as metrics
import sys
import time

#constants
EPOCHS = 2

word2id = vocab_utils.load_word2Id("vocab.txt", "<pad>")

def train(word2id):
	
	# load data
	load_time = time.time()
	print("loading train data...")
	train_x, train_y = vocab_utils.load_train_data(word2id)
	train_x = torch.tensor(train_x)
	train_y = torch.tensor(train_y, dtype=torch.long)
	load_time = time.time() - load_time
	print("finished loading in %.2f seconds." % load_time)

	# initialize model
	bnn = BNN(word2id)
	cel = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(bnn.parameters(), lr=0.01)
	bnn.train()
	torch.save(bnn.state_dict(), "init_model.bin")

	# train
	train_time = time.time()
	print("training...")
	for epoch in range(EPOCHS): # for each epoch...
		running_loss = 0.0

		for i in range(len(train_x)): # for each input/label pair...
			optimizer.zero_grad() # zero out parameter gradients

			# Change the preds to appropriate shape

			# Need to shape preds into (1, 2) (num examples, num_classes)
			preds = bnn(train_x[i]).view(1, 2) # predict
			loss = cel(preds, train_y[i]) # calculate loss
			loss.backward()	# backprop
			optimizer.step()

			# printing statistics
			running_loss += loss.item()

			if i % 2000 == 1999:
				print('[%d, %5d] loss: %.3f, %.2f seconds in' % (epoch + 1, i + 1, running_loss / 2000, time.time() - train_time))
				running_loss = 0.0
	print('finished training.')

	#save model with trained parameters
	print('saving...')
	torch.save(bnn.state_dict(), "model.bin")
	print('finished saving.')

def test(word2id):

	#load test data
	print("loading test data...")
	test_x, test_y = vocab_utils.load_test_data(word2id)
	test_x = torch.tensor(test_x)
	test_y = torch.tensor(test_y, dtype=torch.long)
	print("finished loading.")

	#load model
	bnn = BNN(word2id)
	bnn.load_state_dict(torch.load("model.bin"))
	bnn.eval()

	#test
	print(test_x.size())
	print(test_y.size())
	output = bnn(test_x.permute(1,0))
	print(output.size())
	output = torch.argmax(output, dim=1)
	print(output.size())
	print(metrics.confusion_matrix(test_y, output))

def main():
	print(sys.argv[1])
	if sys.argv[1] == 'train':
		train(word2id)
	elif sys.argv[1] == 'test':
		test(word2id)

if __name__ == "__main__":
	main()