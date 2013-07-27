import pickle
import argparse
import os
import sys
import math
import heapq
import numpy
from operator import itemgetter
from model1withtopics import DirichletModel1WithTopics, DirichletMultinomial, Vocabulary, DirichletProcess, load_data, ParallelSentence

parser = argparse.ArgumentParser()
#parser.add_argument('corpus')
parser.add_argument('pickle')
parser.add_argument('--nonull', action='store_true')
parser.add_argument('--map', action='store_true')

args = parser.parse_args()

print >>sys.stderr, 'Loading pickle file...'
model = pickle.load(open(args.pickle))
print >>sys.stderr, 'Loading corpus...'
#data, french_vocabulary, english_vocabulary, document_ids = load_data(args.corpus, not args.nonull)
data, french_vocabulary, english_vocabulary, document_ids = model.data, model.french_vocabulary, model.english_vocabulary, model.document_ids
print >>sys.stderr, 'Done!'

max_a_posteriori = args.map
show_ttables = True
show_sentence_topics = True
show_alignments = True
show_most_different_words = False

def showTtable(ttable):
	for f in french_vocabulary:
		f_id = french_vocabulary.getId(f)
		if f_id == 0 and args.nonull:
			continue
		print f if f != '' else '(null)'
		translations = [(e, ttable[f_id].probability(english_vocabulary.getId(e))) for e in english_vocabulary]
		translations = [(e, p) for (e, p) in translations if p > 0.0]
		translations = sorted(translations, key=itemgetter(1), reverse=True)
		for e, p in translations[:3]:
			if e == '':
				if args.nonull:
					continue
				else:
					e = '(null)'
			print '\t%s: %g' % (e, p)

if max_a_posteriori:
	changes = 0
	for s, (F, E, d) in enumerate(data):
		for n, e in enumerate(E):
			a = model.alignments[s][n]
			z = model.topic_assignments[s][n]
			f = F[a]
			model.topic_ttables[z][f].decrement(e)
			model.sentence_topics[s].decrement(z)

			best_p = model.topic_ttables[z][f].probability(e)
			best_za = (z, a)
			old_za = (z, a)
			for z in range(model.K):
				for a, f in enumerate(F):
					p = model.topic_ttables[z][f].probability(e)
					if best_p == None or p > best_p:
						best_p = p
						best_za = (z, a)
			if best_za != old_za:
				changes += 1
				print >>sys.stderr, 'Changed sentence %d word %d (%s)\'s alignment and topic from (%s, %d) to (%s, %d)' % (s, n, english_vocabulary.getWord(e), french_vocabulary.getWord(F[old_za[1]]), old_za[0], french_vocabulary.getWord(F[best_za[1]]), best_za[0])
			z, a = best_za
			f = F[a]
			model.topic_ttables[z][f].increment(e)
			model.sentence_topics[s].increment(z)
			model.alignments[s][n] = a
			model.topic_assignments[s][n] = z
	print >>sys.stderr, changes, 'values changed from last iteration to MAP'

if show_ttables:
	print 'Overall ttable:'
	showTtable(model.ttable)
	print '=' * 60
	print

	for k in range(model.K):
		print 'Topic %d ttable:' % k
		showTtable(model.topic_ttables[k])
		for f in range(model.FV):
			if f == 0 and args.nonull:
				continue
			print {english_vocabulary.getWord(k): v for k, v in model.topic_ttables[k][f].tables_by_dish.iteritems()}
		print '=' * 60
		print

if show_sentence_topics:
	print 'Document topics:'
	for d in range(model.D):
		print ' '.join(['%0.2f' % model.document_topics[d].probability(k) for k in range(model.K)]) + '\t' + model.document_ids.getWord(d)
	print
	print 'Sentece topics:'
	for s, (F, E, d) in enumerate(data):
		print ' '.join(['%0.2f' % model.sentence_topics[s].probability(k) for k in range(model.K)]) + '\t' + ' '.join([french_vocabulary.getWord(f) for f in F]) + '\t' + ' '.join([english_vocabulary.getWord(e) for e in E]) \
			+ '\t' + ' '.join([str(model.topic_assignments[s][n]) for n in range(len(E))])
	print

if show_alignments:
	print 'Alignments:'
	for s, (F, E, d) in enumerate(data):
		A = model.alignments[s]
		As = ' '.join([french_vocabulary.getWord(F[a]) for a in A])
		Es = ' '.join([english_vocabulary.getWord(e) for e in E])
		print '%s ||| %s' % (As, Es)
	print

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
