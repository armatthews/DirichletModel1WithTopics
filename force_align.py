import sys
import numpy
import argparse
import cPickle as pickle
from probability import DirichletMultinomial
from vocabulary import Vocabulary
from collections import Counter, namedtuple
from model1withtopics import DirichletModel1WithTopics

ParallelSentence = namedtuple('ParallelSentence', 'F, E, document_id')

parser = argparse.ArgumentParser()
parser.add_argument('model_pickle')
parser.add_argument('corpus_file') # corpus_file format: F ||| E ||| topic_id
args = parser.parse_args()

def showSentencePair(F, E):
	for f in F:
		print >>sys.stderr, french_vocabulary.getWord(f),
	print >>sys.stderr, '||| ',
	for e in E:
		print >>sys.stderr, english_vocabulary.getWord(e),
	print >>sys.stderr

def load_data(stream):
	global model
	line = stream.readline()
	while line:
		f, e, z = [part.strip() for part in line.split('|||')]
		f = [model.french_vocabulary.getId(w) for w in f.split() if len(w) != 0]
		e = [model.english_vocabulary.getId(w) for w in e.split() if len(w) != 0]
		z = int(z)
		yield (f, e, z)
		line = stream.readline()

print >>sys.stderr, 'Loading pickle...'
model = pickle.load(open(args.model_pickle, 'r'))
print >>sys.stderr, 'There seem to be %d French words' % len(model.ttable)
print >>sys.stderr, 'and %d English words' % model.ttable[0].K


data = load_data(open(args.corpus_file))

for f in range(len(model.ttable)):
	model.ttable[f].K = len(model.english_vocabulary)

print >>sys.stderr, 'Aligning...'
for s, (F, E, z) in enumerate(data):
	ttable = model.topic_ttables[z]
	log_prob = 0.0
	#showSentencePair(F, E)
	for n, e in enumerate(E):
		probabilities = []
		for f in F + [0]:
			if f >= len(ttable):
				#print >>sys.stderr, 'Translation not in ttable:', french_vocabulary.getWord(f), english_vocabulary.getWord(e)	
				p = 10.0 ** -40
			elif ttable[f].customers_by_dish.get(e, 0) > 0:
				ttable[f].decrement(e)
				p = ttable[f].probability(e)		
				ttable[f].increment(e)
			else:
				ttable[f].base.K = len(model.english_vocabulary)
				p = ttable[f].probability(e)		
			probabilities.append(p)

		a = numpy.argmax(probabilities)
		S = sum(probabilities)
		log_prob += numpy.log(S) if S > 0.0 else -float('inf')
		if a != len(F):
			print '%d-%d' % (a, n),
	print '|||', log_prob
	sys.stdout.flush()
