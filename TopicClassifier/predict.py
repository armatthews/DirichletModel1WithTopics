import creg
import numpy
import sys
import cPickle as pickle
import random
import argparse
sys.path.append('..')
from model1withtopics import *

def gen_data(model):
        for s, (F, E, d) in enumerate(model.data):
                #sys.stderr.write('Loaded %d/%d sentences\r' % (s, len(model.data)))
                features = {}
                for f in F:
                        features['f' + str(f)] = 1.0
                topic_probs = [model.sentence_topics[s].probability(k) for k in range(model.K)]
                label = numpy.array(topic_probs).argmax()
                #print label,
                #label = random.randint(0, 1)
                #print label
                yield features, label
        print >>sys.stderr

def parse_dict(s):
	def parse_kv(s):
		colon = s.find(':')
		k = s[:colon]
		n = s[colon + 1]
		if n == '{':
			v = parse_dict(s[colon + 1:])
		else:
			comma = s.find(',')
			if comma == -1:
				comma = s.find('}')
			v = s[colon + 1 : comma]
		print k, len(v)

	assert s[0] == '{'
	assert s[-1] == '}'
	parse_kv(s[1:])

if False:
	parser = argparse.ArgumentParser()
	parser.add_argument('pickle')
	args = parser.parse_args()

	print >>sys.stderr, 'Loading pickle...'
	with open(args.pickle) as f:
		model = pickle.load(f)

	print >>sys.stderr, 'Generating dataset...'
	train_data = creg.CategoricalDataset(gen_data(model))
	print train_data # <Dataset: 1000 instances, 3 features>
	#sys.exit(0)

print >>sys.stderr, 'Reading weights...'
model = creg.LogisticRegression()
weights = sys.stdin.readline().strip()
print weights[:100]
weights = parse_dict(weights)
sys.exit(0)
model.weights = weights

print >>sys.stderr, 'Predicting...'
test_data = train_data
predictions = model.predict(test_data)
truth = (y for x, y in test_data)
errors = sum(1 if pred != real else 0 for (pred, real) in zip(predictions, truth))
print 'Accuracy: %.3f' % (1-errors/float(len(test_data)))
