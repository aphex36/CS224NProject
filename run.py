#USAGE: python3 run.py {train, test} {bnn, rnn, cnn} model_savefile {cuda, cpu}
import torch
import torch.nn as nn
import torch.utils.data as data
import vocab_utils
import numpy as np
import sklearn.metrics as metrics
import sys
import time
import os
import pickle

#models
from BaseNeuralNetwork import BaseNeuralNetwork as BNN
from RNN import RNN
from CNN import CNN

#constants
EPOCHS = 10
BATCH_SIZE = 32
vocabfile = "./vocab.txt"
train_pkl = "./train.pkl"
test_pkl = "./test.pkl"


def train(model, word2id, savefile, device):
	
	# load data
	load_time = time.time()
	print("loading train data...")
	train_data = None
	if os.path.isfile(train_pkl):
		train_data = pickle.load(open(train_pkl, "rb"))
	else:
		train_data = vocab_utils.load_train_data(word2id)
		pickle.dump(train_data, open(train_pkl, "wb"))
	train_x, train_y = train_data
	train_x = torch.tensor(train_x, device=device) #(20000, 1002)
	train_y = torch.tensor(train_y, dtype=torch.long, device=device) #(20000, 1)
	train_data = data.TensorDataset(train_x,train_y)
	train_loader = data.DataLoader(train_data,batch_size=BATCH_SIZE, shuffle=True)
	load_time = time.time() - load_time
	print("finished loading in %.2f seconds." % load_time)

	# initialize model
	cel = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

	# train
	model.train()
	train_time = time.time()
	print("training...")
	for epoch in range(EPOCHS): # for each epoch...
		running_loss = 0.0

		for batch_idx, (train_x, train_y) in enumerate(train_loader): # for each batch...
			
			# batch update
			optimizer.zero_grad() # zero out parameter gradients
			preds = model(train_x.permute(1,0)) # predict
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
	torch.save(model.state_dict(), savefile)
	print('finished saving.')

def test(model, word2id, savefile, device):

	#load test data
	print("loading test data...")
	test_data = None
	if os.path.isfile(test_pkl):
		test_data = pickle.load(open(test_pkl, "rb"))
	else:
		test_data = vocab_utils.load_test_data(word2id)
		pickle.dump(test_data, open(test_pkl, "wb"))
	test_x, test_y = test_data
	test_x = torch.tensor(test_x, device=device) # (2000, 964)
	test_y = torch.tensor(test_y, dtype=torch.long, device=device) # (2000,1)
	print("finished loading.")

	#load model
	model.load_state_dict(torch.load(savefile))
	model.eval()

	#test
	output = model(test_x.permute(1,0))
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
	command = sys.argv[1]
	model_type = sys.argv[2]
	savefile = sys.argv[3]
	devicename = sys.argv[4]


	#initialize model
	word2id = vocab_utils.load_word2Id(vocabfile, "<pad>")
	model = None
	if model_type == "bnn":
		model = BNN(word2id)
	elif model_type == "cnn":
	    model = CNN(word2id)
	elif model_type == "rnn":
		model = RNN(word2id)

	# store model on correct device
	device = torch.device("cpu")
	if devicename == "cuda":
		device = torch.device("cuda:0")
	model.to(device)

	#train or test
	if command == 'train':
		train(model, word2id, savefile, device)
	elif command == 'test':
		test(model, word2id, savefile, device)

if __name__ == "__main__":
	main()
