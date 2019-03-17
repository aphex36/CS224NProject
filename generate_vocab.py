import json
import sys
import collections
import pickle
import numpy as np
trainfile = sys.argv[1]
testfile = sys.argv[2]

vocab = set()
idf_counts = collections.defaultdict(int)
num_docs = 0
files = [trainfile, testfile]
for fileFound in files:
    currFile = open(fileFound, 'r')
    for line in currFile:
        num_docs += 1
        doc_words = set()
        words = json.loads(line)['text'].split(' ')
        for word in words:
            word = word.replace("\n","")
            word = word.replace("\r","")
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
                            finalWord = finalWord.lower()
                            vocab.add(finalWord)
                            doc_words.add(finalWord)
        # idf counts
        for word in doc_words:
            idf_counts[word] += 1
    
    currFile.close()
        
# idf calculation
idfs = {}
for word, count in idf_counts.items():
    idfs[word] = np.log(num_docs/float(count))
pickle.dump(idfs, open("idf.pkl", "wb"))

vocabFile = open('vocab.txt', 'w')
for word in vocab:
	vocabFile.write(word + "\n")
vocabFile.close()

