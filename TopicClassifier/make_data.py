import creg
import numpy
import sys
import cPickle as pickle
import random
import argparse
sys.path.append('..')
from model1withtopics import *

parser = argparse.ArgumentParser()
parser.add_argument('pickle')
parser.add_argument('tune')
parser.add_argument('test')
args = parser.parse_args()

print >>sys.stderr, 'Loading pickle...'
with open(args.pickle) as f:
	model = pickle.load(f)

def gen_data(model, data):
	for s, (F, E, d) in enumerate(data):
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

train_data = creg.CategoricalDataset(gen_data(model, model.data))
tune_data, model.french_vocabulary, model.english_vocabulary, document_ids = load_data(args.tune, True, model.french_vocabulary, model.english_vocabulary, model.document_ids)
tune_data  = creg.CategoricalDataset(gen_data(model, tune_data))
test_data, model.french_vocabulary, model.english_vocabulary, document_ids = load_data(args.test, True, model.french_vocabulary, model.english_vocabulary, model.document_ids)
test_data  = creg.CategoricalDataset(gen_data(model, test_data))
print >>sys.stderr, train_data
print >>sys.stderr, test_data
print >>sys.stderr, tune_data

model = creg.LogisticRegression()
model.fit(train_data, 0, 0, 40, 1e-2, 0)
print >>sys.stderr, model.weights
#sys.stdout.write(pickle.dumps(model.weights))
sys.stdout.flush()

for data in [train_data, tune_data, test_data]:
	predictions = model.predict(data)
	print >>sys.stderr, 'Predictions:', list(predictions)
	truth = (y for x, y in data)
	errors = sum(1 if pred != real else 0 for (pred, real) in zip(predictions, truth))
	print >>sys.stderr, 'Accuracy: %.3f' % (1-errors/float(len(data)))
