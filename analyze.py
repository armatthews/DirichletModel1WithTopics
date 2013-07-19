import pickle
import argparse
import os
import sys
import math
import numpy
import heapq
from operator import itemgetter
from model1withtopics import DirichletModel1WithTopics, DirichletMultinomial, Vocabulary, DirichletProcess, load_data, ParallelSentence

parser = argparse.ArgumentParser()
parser.add_argument('corpus')
parser.add_argument('pickle')
parser.add_argument('--nonull', action='store_true')

args = parser.parse_args()

print >>sys.stderr, 'Loading pickle file...'
model = pickle.load(open(args.pickle))
print >>sys.stderr, 'Loading corpus...'
data, french_vocabulary, english_vocabulary, document_ids = load_data(args.corpus, not args.nonull)
print >>sys.stderr, 'Done!'

show_ttables = True
show_document_topics = True
show_sentence_topics = True
show_most_different_words = False
show_topic_distances = True

def showTtable(ttable):
	for f in french_vocabulary:
		if f == '' and args.nonull:
			continue
		f_id = french_vocabulary.getId(f)
		print f if f != '' else '(null)'
		translations = [(e, ttable[f_id].probability(english_vocabulary.getId(e))) for e in english_vocabulary if e != '' or not args.nonull]
		translations = [(e, p) for (e, p) in translations if p > 0.0]
		translations = sorted(translations, key=itemgetter(1), reverse=True)
		for e, p in translations[:3]:
			if e == '':
				e = '(null)'
			print '\t%s: %g' % (e, p)

if show_ttables:
	print 'Overall ttable:'
	showTtable(model.ttable)
	print '=' * 60
	print

	for k in range(model.K):
		print 'Topic %d ttable:' % k
		showTtable(model.topic_ttables[k])
		print '=' * 60
		print

if show_document_topics:
	print 'Document topics:'
	for d in range(model.D):
		print ' '.join(['%0.2f' % model.document_topics[d].probability(k) for k in range(model.K)]) + '\t' + model.document_ids.getWord(d)

if show_sentence_topics:
	print 'Sentence topics:'
	for s, (F, E, d) in enumerate(data):
		sys.stdout.write(' '.join(['%0.2f' % model.sentence_topics[s].probability(k) for k in range(model.K)]) + '\t' + \
		    ' '.join([french_vocabulary.getWord(f) for f in F]) + '\t' + \
		    ' '.join([english_vocabulary.getWord(e) for e in E]) + '\t')

		for n, e in enumerate(E):
			a = model.alignments[s][n]
			f = F[a]
			t = numpy.array([model.topical_probs[f].probability(t) for t in range(2)]).argmax()
			if t == 0:
				z = None
			else:
				z = model.topic_assignments[s][n]

			if z != None:
				model.sentence_topics[s].decrement(z)
				sent_top_probs = [model.sentence_topics[s].probability(k) for k in range(model.K)]
				model.sentence_topics[s].increment(z)
		
				model.topic_ttables[z][f].decrement(e)
				ttable_probs = [model.topic_ttables[k][f].probability(e) for k in range(model.K)]
				model.topic_ttables[z][f].increment(e)

				probs = [sent_top_probs[k] * ttable_probs[k] for k in range(model.K)]
				probs = [p/sum(probs) for p in probs]
				z = numpy.array(probs).argmax()

				sys.stdout.write('%d ' % z)
			else:
				sys.stdout.write('N ')
		print

if show_most_different_words:
	print 'Words per topic that vary most from overall ttable:'
	for f in range(model.FV):
		for k in range(model.K):
			print french_vocabulary.getWord(f), k,
			translations = [k for k, v in model.ttable[f].most_common(5)]
			print [english_vocabulary.getWord(e) for e in translations],
			topical_translations = [model.topic_ttables[k][f].probability(e) for e in translations]
			print topical_translations,
			non_topical_translations = [model.ttable[f].probability(e) for e in translations]
			print non_topical_translations,
			print

if show_topic_distances:
	print 'Word p(topical) and cos distance of topical ttables from underlying ttable:'
	for f in french_vocabulary:
		id = french_vocabulary.getId(f)
		topic_distances = []
		underlying_ttable = [model.ttable[id].probability(e) for e in range(len(english_vocabulary))]
		for k in range(model.K):
			topic_ttable = [model.topic_ttables[k][id].probability(e) for e in range(len(english_vocabulary))]
			distance = sum(a * b for a, b in zip(underlying_ttable, topic_ttable))
			distance /= math.sqrt(sum(a ** 2 for a in underlying_ttable))
			distance /= math.sqrt(sum(b ** 2 for b in topic_ttable))
			distance = math.acos(distance - 1e-10)
			topic_distances.append('%.2f' % distance)
		print '%s\t%f' % (f, model.topical_probs[id].probability(1)), ' '.join(map(str, topic_distances))
