import numpy
import scipy
import sys
import argparse
import os
import cPickle as pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict, Counter
import math
numpy.random.seed(0)

class Vocabulary:
	def __init__(self, use_null=False):
		self.vocabulary = []
		self.wordToId = {}
		if use_null:
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

class Document:
	def __init__(self, text):
		self.counter = Counter()
		text = text.lower()	
		words = tokenize(text)
		self.length = len(words)

		for word in words:
			self.counter[word] += 1

	def __getitem__(self, word):
		return self.counter[word]

	def __len__(self):
		return self.length

	def __iter__(self):
		for key in self.counter.keys():
			for i in range(self.counter[key]):
				yield key

class DirichletMultinomial:
	def __init__(self, K, alpha):
		self.K = K
		self.alpha = 1.0 * alpha
		self.counts = Counter()
		self.N = 0

	def increment(self, k):
		assert k >= 0 and k < self.K
		self.counts[k] += 1
		self.N += 1

	def decrement(self, k):
		assert k >= 0 and k < self.K
		self.counts[k] -= 1
		self.N -= 1

	def probability(self, k):
		assert k >= 0 and k < self.K
		numerator = self.alpha + self.counts[k]
		denominator = self.alpha * self.K + self.N
		return numerator / denominator

class ChineseRestaurantProcess:
	def __init__(self):
		self.tables_by_dish = {}
		self.customers_by_dish = {}
		self.num_tables = 0
		self.num_customers = 0

	# Seat a customer at the table_index'th table
	# that is labeled with the given dish.
	# If table_index is None, then a new table is created
	# and labled with the given dish.
	# Return value is whether the customer was
	# seated at a new table.
	def seat_customer(self, dish, table_index):
		table_created = False

		if dish not in self.tables_by_dish:
			self.tables_by_dish[dish] = []
			self.customers_by_dish[dish] = 0

		if table_index == None:
			self.tables_by_dish[dish].append(0)
			self.num_tables += 1
			table_index = len(self.tables_by_dish[dish]) - 1
			table_created = True

		assert table_index >= 0
		assert table_index < len(self.tables_by_dish[dish])

		self.tables_by_dish[dish][table_index] += 1
		self.customers_by_dish[dish] += 1
		self.num_customers += 1
		return table_created

	# Eject a customer from the table_index'th table
	# that is labeled with the given dish.
	# Return value is whether the ejectee was the last
	# customer at his table.
	def eject_customer(self, dish, table_index):
		assert dish in self.tables_by_dish
		assert table_index >= 0
		assert table_index < len(self.tables_by_dish[dish])
		table_removed = False

		self.tables_by_dish[dish][table_index] -= 1
		self.customers_by_dish[dish] -= 1
		self.num_customers -= 1

		if self.tables_by_dish[dish][table_index] == 0:
			del self.tables_by_dish[dish][table_index]
			self.num_tables -= 1
			table_removed = True

		if self.customers_by_dish[dish] == 0:
			del self.customers_by_dish[dish]
			del self.tables_by_dish[dish]

		return table_removed

	def eject_random_customer(self, dish):
		i = numpy.random.randint(0, self.customers_by_dish[dish])
		for table_index, customers in enumerate(self.tables_by_dish[dish]):
			if i < customers:
				return self.eject_customer(dish, table_index)
			else:
				i -= customers
		raise Exception()
		
	def output(self):
		for dish in self.tables_by_dish.keys():
			print 'There are %d customers at %d tables serving %s with populations %s.' % \
				(self.customers_by_dish[dish], len(self.tables_by_dish[dish]), str(dish), ' '.join([str(n) for n in self.tables_by_dish[dish]]))

class DirichletProcess(ChineseRestaurantProcess):
	def __init__(self, strength, base):
		ChineseRestaurantProcess.__init__(self)
		self.strength = 1.0 * strength
		self.base = base

	def probability(self, dish):
		numerator = self.strength * self.base.probability(dish)
		numerator += self.customers_by_dish[dish] if dish in self.customers_by_dish else 0.0
		denominator = self.strength + self.num_customers
		assert numerator / denominator >= 0.0
		assert numerator / denominator <= 1.0
		return numerator / denominator

	def tables_serving_dish(self, dish):
		if dish in self.tables_by_dish:
			for table_index, num_customers in enumerate(self.tables_by_dish[dish]):
				yield table_index, num_customers
		yield None, self.strength * self.base.probability(dish)

	def increment(self, dish):
		table = draw_from_multinomial([num_customers for table_index, num_customers in self.tables_serving_dish(dish)])
		if dish not in self.tables_by_dish or table >= len(self.tables_by_dish[dish]):
			table = None
		updateBase = self.seat_customer(dish, table)
		if updateBase:
			self.base.increment(dish)

	def decrement(self, dish):
		updateBase = self.eject_random_customer(dish)
		if updateBase:
			self.base.decrement(dish)	

class DirichletModel1:
	def __init__(self, K, alpha, beta0, beta1, data, french_vocabulary, english_vocabulary, use_null=True):
		self.K = K
		self.alpha = alpha
		self.beta0 = beta0
		self.beta1 = beta1
		self.data = data
		self.french_vocabulary = french_vocabulary
		self.english_vocabulary = english_vocabulary
		self.use_null = use_null
		self.alignments_fixed = False

		if use_null:
			self.data = [(F + [0], E) for (F, E) in self.data]

		self.FV = len(french_vocabulary)
		self.EV = len(english_vocabulary)
		self.S = len(data)

		self.ttable = [DirichletMultinomial(self.EV, self.beta0) for f in range(self.FV)]
		self.topic_ttables = [[DirichletProcess(self.beta1, self.ttable[f]) for f in range(self.FV)] for k in range(self.K)]
		self.sentence_topics = [DirichletMultinomial(self.K, self.alpha) for s in range(self.S)]
		self.topic_assignments = []
		self.alignments = []

		for s, (F, E) in enumerate(self.data):
			self.alignments.append([None for e in E])
			self.topic_assignments.append([None for e in E])
			for n, e in enumerate(E):
				self.assign_topicless_alignment(s, n, F, e)
				self.assign_topic(s, n, F, e)

	def fix_alignments(self, alignments):
		self.alignments = alignments
		self.alignments_fixed = True

	def draw_topicless_alignment(self, s, n, F, e):
		probabilities = [self.ttable[f].probability(e) for f in F]
		return draw_from_multinomial(probabilities)

	def assign_topicless_alignment(self, s, n, F, e):
		a = self.draw_topicless_alignment(s, n, F, e)
		self.alignments[s][n] = a

	def draw_alignment(self, s, n, F, e):
		z = self.topic_assignments[s][n]
		probabilities = [self.topic_ttables[z][f].probability(e) for f in F]
		return draw_from_multinomial(probabilities)
		
	def assign_alignment(self, s, n, F, e):
		z = self.topic_assignments[s][n]
		a = self.draw_alignment(s, n, F, e)
		self.alignments[s][n] = a	
		self.topic_ttables[z][F[a]].increment(e)

	def remove_alignment(self, s, n, F, e):
		a = self.alignments[s][n]
		z = self.topic_assignments[s][n]
		self.topic_ttables[z][F[a]].decrement(e)

	def draw_topic_assignment(self, s, n, F, e):
		a = self.alignments[s][n]
		f = F[a]
		probabilities = [self.sentence_topics[s].probability(z) * \
				 self.topic_ttables[z][f].probability(e) \
				 for z in range(self.K)]

		return draw_from_multinomial(probabilities)

	def assign_topic(self, s, n, F, e):
		a = self.alignments[s][n]
		z = self.draw_topic_assignment(s, n, F, e)
		self.topic_assignments[s][n] = z
		self.topic_ttables[z][F[a]].increment(e)
		self.sentence_topics[s].increment(z)

	def remove_topic_assignment(self, s, n, F, e):
		a = self.alignments[s][n]
		z = self.topic_assignments[s][n]
		self.topic_ttables[z][F[a]].decrement(e)
		self.sentence_topics[s].decrement(z)

	def iterate(self, i):
		for s, (F, E) in enumerate(self.data):
			for n, e in enumerate(E):
				if not self.alignments_fixed:
					self.remove_alignment(s, n, F, e)	
					self.assign_alignment(s, n, F, e)
				self.remove_topic_assignment(s, n, F, e)
				self.assign_topic(s, n, F, e)

	def output(self):
		for s, (F, E) in enumerate(self.data):
			alignment = []
			topics = []
			for n, e in enumerate(E):
				best = (0.0, None, None)
				for z in range(self.K):
					for a, f in enumerate(F):
						p = self.topic_ttables[z][f].probability(e)
						if p > best[0]:
							best = (p, z, a)
				p, z, a = best
				alignment.append(a)
				topics.append(z)

			for n, e in enumerate(E):
				a = alignment[n]
				print '%d-%d' % (a, n),
			print

	def log_likelihood(self):
		log_likelihood = 0.0	

		S = len(self.data)
		for s, (F, E) in enumerate(self.data):	
			for n, e in enumerate(E):
				z = self.topic_assignments[s][n]
				a = self.alignments[s][n]
				f = F[a]

				log_likelihood += math.log(self.topic_ttables[z][f].probability(e))
				log_likelihood += math.log(self.sentence_topics[s].probability(z))
			log_likelihood += dirichlet_log_prob([self.sentence_topics[s].probability(k) for k in range(self.K)], self.alpha)

		for f in range(self.FV):
			for k in range(self.K):
				#log_likelihood += log(dp_prob(self.topic_ttable[f], self.topic.ttable[f], self.beta1)
				pass
			#log_likelihood += dirichlet_log_prob([self.ttable[f].probability(e) for e in range(self.EV)], self.beta0)
			pass

		return log_likelihood
		

	def perplexity(self):
		log_likelihood = self.log_likelihood()
		word_count = sum(len(E) for (F, E) in data)
		return math.exp(-log_likelihood / word_count)

def draw_from_multinomial(probabilities):
	prob_sum = sum(probabilities)
	probabilities = [p / prob_sum for p in probabilities]
	return numpy.random.multinomial(1, probabilities).argmax()

def dirichlet_log_prob(X, a):
	k = len(X)
	P = sum([(a - 1) * x for x in X])
	B = k * scipy.special.gammaln(a) - scipy.special.gammaln(k * a)
	return P - B

def load_data(filename, use_null):
	data = []
	french_vocabulary = Vocabulary(use_null)
	english_vocabulary = Vocabulary()

	stream = open(filename)
	line = stream.readline()
	while line:
		f, e = [part.split() for part in line.split('|||')]
		f = [french_vocabulary.getId(w) for w in f if len(w) != 0]
		e = [english_vocabulary.getId(w) for w in e if len(w) != 0]
		data.append((f, e))
		line = stream.readline()
	
	stream.close()
	return data, french_vocabulary, english_vocabulary

def load_sentence_alignment(stream, F, E):
	line = stream.readline().strip()
	link_dict = {}
	for link in line.split():
		link = link.split('-')
		s = int(link[0])
		t = int(link[1])
		link_dict[t] = s

	return [link_dict[n] if n in link_dict else len(F) for n in range(len(E))]
		

def load_alignment(filename, data):
	stream = open(filename)
	alignments = []
	for s, (F, E) in enumerate(data):
		alignment = load_sentence_alignment(stream, F, E)
		alignments.append(alignment)
	stream.close()
	return alignments

def output_iteration(i, model, outputDir):
	print >>sys.stderr, 'Iteration %d perplexity: %f' % (i, model.perplexity())
	sys.stderr.flush()

	model.output()
	print
	sys.stdout.flush()

	out_pickle = open(os.path.join(outputDir, 'model%d.pkl' % i), 'w')
	pickle.dump(model, out_pickle)
	out_pickle.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('corpus')
	parser.add_argument('output_dir')
	parser.add_argument('--num_iterations', type=int, default=100)
	parser.add_argument('--num_topics', type=int, default=3)
	parser.add_argument('--alpha', type=float, default=1.0)
	parser.add_argument('--beta0', type=float, default=0.002)
	parser.add_argument('--beta1', type=float, default=1.0)
	parser.add_argument('--aligns')
	parser.add_argument('--nonull', action='store_true')
	args = parser.parse_args()

	allow_null = not args.nonull
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	print >>sys.stderr, 'Loading data...'
	data, french_vocabulary, english_vocabulary = load_data(args.corpus, allow_null)

	print >>sys.stderr, 'Initializing model...'
	model = DirichletModel1(args.num_topics, args.alpha, args.beta0, args.beta1, data, french_vocabulary, english_vocabulary, allow_null)

	if args.aligns != None:
		print >>sys.stderr, 'Loading gold alignments...'
		gold_alignment = load_alignment(args.align, data)
		model.fix_alignments(gold_alignment)

	output_iteration(0, model, args.output_dir)
	for i in range(1, args.num_iterations + 1):
		model.iterate(i)
		output_iteration(i, model, args.output_dir)
