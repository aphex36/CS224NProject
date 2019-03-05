import math
import json

def perform_lookups(listOfSentences, word2id):

	# Takes in a list of sentences
	listOfLookups = []
	counter = 0
	for sentence in listOfSentences:
		indices = []
		words = sentence.split(' ')
		for word in words:
			word = word.replace("\n","")
			word = word.replace("\r", "")
			word = word.replace(",", "")
			word = word.replace(":", "")
			word = word.replace(")", "")
			word = word.replace("(", "")
			word = word.replace("\"", "")
			word = word.replace("\\","")
			mini_words = word.split(".")
			for mini_word in mini_words:
				exclaim_points = mini_word.split("!")
				for wordFound in exclaim_points:
					question_marks = wordFound.split("?")
					for finalWord in question_marks:
						if len(finalWord) != 0:
							indices.append(word2id[finalWord.lower()])
		counter += 1
		listOfLookups.append(indices)
	return listOfLookups


def get_max_len(listOfSentences):

	# Takes in a list of sentences
	listOfLookups = []
	counter = 0
	max_size = 0
	sentenceToSize = dict()
	for sentence in listOfSentences:
		indicesSize = 0
		words = sentence.split(' ')
		for word in words:
			word = word.replace("\n","")
			word = word.replace("\r", "")
			word = word.replace(",", "")
			word = word.replace(":", "")
			word = word.replace(")", "")
			word = word.replace("(", "")
			word = word.replace("\"", "")
			word = word.replace("\\","")
			mini_words = word.split(".")
			for mini_word in mini_words:
				exclaim_points = mini_word.split("!")
				for wordFound in exclaim_points:
					question_marks = wordFound.split("?")
					for finalWord in question_marks:
						if len(finalWord) != 0:
							indicesSize += 1
		if max_size < indicesSize:
			max_size = indicesSize
		sentenceToSize[counter] = indicesSize
		counter += 1 
	return max_size, sentenceToSize

def pad_sentences(listOfSentences, padToken):
	maxLen, sentenceToSize = get_max_len(listOfSentences)
	counter = 0
	for sentence in listOfSentences:
		paddingNeeded = maxLen - sentenceToSize[counter]
		for i in range(paddingNeeded):
			sentence += " " + padToken
		listOfSentences[counter] = sentence
		counter += 1
	return listOfSentences

def load_word2Id(vocabFile, padToken):
	word2Id = dict()
	currFile = open(vocabFile, 'r')
	idCounter = 1
	for line in currFile:
		line = line.replace("\n","")
		word2Id[line] = idCounter
		idCounter += 1
	currFile.close()
	word2Id[padToken] = 0
	return word2Id

def load_train_data(word2id):
	labels = []
	train_data = []
	currFile = open("smaller_train_shuffled.json", 'r')
	for line in currFile:
		line = line.replace("\n","")
		review = json.loads(line)
		if review['funny'] > 0:
			labels.append([0])
		else:
			labels.append([1])
		train_data.append(review['text'])
	train_data = pad_sentences(train_data, "<pad>")
	train_data = perform_lookups(train_data, word2id)
	return train_data, labels

def load_test_data(word2id):
	labels = []
	test_data = []
	currFile = open("smaller_test_shuffled.json", 'r')
	for line in currFile:
		line = line.replace("\n","")
		review = json.loads(line)
		if review['funny'] > 0:
			labels.append([0])
		else:
			labels.append([1])
		test_data.append(review['text'])
	test_data = pad_sentences(test_data, "<pad>")
	test_data = perform_lookups(test_data, word2id)
	return test_data, labels


def batch_data(listOfSentences, labels, batches=32):
	assert(len(listOfSentences) == len(labels))

	newSentences = []
	newLabels = []
	numBatches = int(math.ceil(len(listOfSentences)*1.0/batches))
	for i in range(numBatches):
		newSentences.append(listOfSentences[batches*i:batches*(i+1)])
		newLabels.append(labels[batches*i:batches*(i+1)])
	return newSentences, newLabels

