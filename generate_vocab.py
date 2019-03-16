import json
import sys

trainfile = sys.argv[1]
testfile = sys.argv[2]

vocab = set()
files = [trainfile, testfile]
for fileFound in files:
	currFile = open(fileFound, 'r')
	for line in currFile:
		words = json.loads(line)['text'].split(' ')
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
							vocab.add(finalWord.lower())
	currFile.close()

vocabFile = open('vocab.txt', 'w')
for word in vocab:
	vocabFile.write(word + "\n")
vocabFile.close()

