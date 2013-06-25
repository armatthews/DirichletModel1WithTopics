import pickle
import argparse
import os
from operator import itemgetter
from model1withtopics import DirichletMultinomial, DirichletModel1, Vocabulary, DirichletProcess, load_data

parser = argparse.ArgumentParser()
parser.add_argument('corpus')
parser.add_argument('output_dir')
parser.add_argument('iteration', type=int)
args = parser.parse_args()

model = pickle.load(open(os.path.join(args.output_dir, 'model%d.pkl' % args.iteration)))
data, french_vocabulary, english_vocabulary = load_data(args.corpus, model.use_null)

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

print 'Overall ttable:'
showTtable(model.ttable)
print '=' * 60
print

for k in range(model.K):
	print 'Topic %d ttable:' % k
	showTtable(model.topic_ttables[k])
	print '=' * 60
	print

print 'Sentece topics:'
for s, (F, E) in enumerate(data):
	print ' '.join(['%0.2f' % model.sentence_topics[s].probability(k) for k in range(model.K)]) + '\t' + ' '.join([english_vocabulary.getWord(e) for e in E]) + '\t' + ' '.join([french_vocabulary.getWord(f) for f in F])
