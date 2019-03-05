import torch
import torch.nn as nn
import torch.utils.data as data
from BaseNeuralNetwork import BaseNeuralNetwork as BNN
import vocab_utils
import numpy as np
import sklearn.metrics as metrics
import sys
import time

#constants
EPOCHS = 10
BATCH_SIZE = 4
word2id = vocab_utils.load_word2Id("vocab.txt", "<pad>")

def train(word2id):
	
	# load data
	load_time = time.time()
	print("loading train data...")
	train_x, train_y = vocab_utils.load_train_data(word2id)
	train_x = torch.tensor(train_x) #(20000, 1002)
	train_y = torch.tensor(train_y, dtype=torch.long) #(20000, 1)
	train_data = data.TensorDataset(train_x,train_y)
	train_loader = data.DataLoader(train_data,batch_size=BATCH_SIZE, shuffle=True)
	load_time = time.time() - load_time
	print("finished loading in %.2f seconds." % load_time)

	# initialize model
	bnn = BNN(word2id)
	cel = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(bnn.parameters(), lr=0.01)
	torch.save(bnn.state_dict(), "init_model.bin")

	# train
	bnn.train()
	train_time = time.time()
	print("training...")
	for epoch in range(EPOCHS): # for each epoch...
		running_loss = 0.0

		for batch_idx, (train_x, train_y) in enumerate(train_loader): # for each batch...
			
			# batch update
			optimizer.zero_grad() # zero out parameter gradients
			preds = bnn(train_x.permute(1,0)) # predict
			loss = cel(preds, train_y.squeeze()) # calculate loss
			loss.backward()	# backprop
			optimizer.step() # update parameters

			# printing statistics
			running_loss += loss.item()
			if batch_idx % 500 == 499:
				print('[%d, %5d] loss: %.3f, %.2f seconds in' % (epoch + 1, batch_idx + 1, running_loss / 500, time.time() - train_time))
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
	test_x = torch.tensor(test_x) # (2000, 964)
	test_y = torch.tensor(test_y, dtype=torch.long) # (2000,1)
	print("finished loading.")

	#load model
	bnn = BNN(word2id)
	bnn.load_state_dict(torch.load("model.bin"))
	bnn.eval()

	#test
	output = bnn(test_x.permute(1,0))
	output = torch.argmax(output, dim=1)
	print("confusion matrix:")
	print(metrics.confusion_matrix(test_y, output))
	print("accuracy:")
	print(metrics.accuracy_score(test_y,output))
	print("F1 score:")
	print(metrics.f1_score(test_y,output))
	print("precision:")
	print(metrics.precision_score(test_y, output))
	print("recall:")
	print(metrics.recall_score(test_y, output))


def main():
	print(sys.argv[1])
	if sys.argv[1] == 'train':
		train(word2id)
	elif sys.argv[1] == 'test':
		test(word2id)

if __name__ == "__main__":
	main()
