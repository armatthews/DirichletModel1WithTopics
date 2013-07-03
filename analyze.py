import pickle
import argparse
import os
import sys
import math
import heapq
from operator import itemgetter
from model1withtopics import DirichletModel1WithTopics, DirichletMultinomial, Vocabulary, DirichletProcess, load_data, ParallelSentence

parser = argparse.ArgumentParser()
parser.add_argument('corpus')
parser.add_argument('pickle')
parser.add_argument('no_null', action='store_true')

args = parser.parse_args()

print >>sys.stderr, 'Loading pickle file...'
model = pickle.load(open(args.pickle))
print >>sys.stderr, 'Loading corpus...'
data, french_vocabulary, english_vocabulary, document_ids = load_data(args.corpus, not args.no_null)
print >>sys.stderr, 'Done!'

show_ttables = False
show_sentence_topics = True
show_most_different_words = False

def showTtable(ttable):
	for f in french_vocabulary:
		f_id = french_vocabulary.getId(f)
		print f if f != '' else '(null)'
		translations = [(e, ttable[f_id].probability(english_vocabulary.getId(e))) for e in english_vocabulary]
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

if show_sentence_topics:
	print 'Document topics:'
	for d in range(model.D):
		print ' '.join(['%0.2f' % model.document_topics[d].probability(k) for k in range(model.K)]) + '\t' + model.document_ids.getWord(d)
	print 'Sentece topics:'
	for s, (F, E, d) in enumerate(data):
		print ' '.join(['%0.2f' % model.sentence_topics[s].probability(k) for k in range(model.K)]) + '\t' + ' '.join([english_vocabulary.getWord(e) for e in E]) + '\t' + ' '.join([french_vocabulary.getWord(f) for f in F])

if show_most_different_words:
	print 'Words per topic that vary most from overall ttable'
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

#for e in english_vocabulary:
#	id = english_vocabulary.getId(e)
#	print '%s\t%f' % (e, model.topical_prob[id].probability(1))
