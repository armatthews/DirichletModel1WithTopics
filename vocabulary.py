class Vocabulary(object):
	def __init__(self):
		self.vocabulary = []
		self.wordToId = {}

		self.null_word = ''
		self.vocabulary.append(self.null_word)
		self.wordToId[self.null_word] = 0

	def getId(self, word):
		if word in self.wordToId:
			return self.wordToId[word]
		else:
			newId = len(self.vocabulary)
			self.vocabulary.append(word)
			self.wordToId[word] = newId
			return newId

	def getWord(self, Id):
		return self.vocabulary[Id]

	def __len__(self):
		return len(self.vocabulary)

	def __contains__(self, key):
		return key in self.vocabulary

	def __iter__(self):
		for word in self.vocabulary:
			yield word

