import math

def perform_lookups(listOfSentences, word2id):

	# Takes in a list of sentences
	listOfLookups = []
	for sentence in listOfSentences:
		indices = []
		words = sentence.split(' ')
		for word in words:
			word = word.replace("\n","")
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
						indices.append(word2id[finalWord])
		listOfLookups.append(indices)
	return listOfLookups

def batch_data(listOfSentences, labels, batches=32):
	assert(len(listOfSentences) == len(labels))

	newSentences = []
	newLabels = []
	numBatches = int(math.ceil(len(listOfSentences)*1.0/batches))
	for i in range(numBatches):
		newSentences.append(listOfSentences[batches*i:batches*(i+1)])
		newLabels.append(labels[batches*i:batches*(i+1)])
	return newSentences, newLabels

