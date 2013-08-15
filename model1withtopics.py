import numpy
import scipy
import sys
import argparse
import os
import cPickle as pickle
from vocabulary import Vocabulary
from probability import DirichletMultinomial, DirichletProcess, draw_from_multinomial
from prior import SampledPrior, GammaPrior, FixedValue, BetaPrior
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict, Counter, namedtuple
import math
numpy.random.seed(0)

ParallelSentence = namedtuple('ParallelSentence', 'F, E, document_id')

class DirichletModel1WithTopics(object):
	def __init__(self, K, alpha0, alpha1, beta0, beta1, data, french_vocabulary, english_vocabulary, document_ids):
		self.K = K

		self.alpha0_prior = alpha0 if isinstance(alpha0, SampledPrior) else FixedValue(alpha0)
		self.alpha0_prior.tie(self)

		self.alpha1_prior = alpha1 if isinstance(alpha1, SampledPrior) else FixedValue(alpha1)
		self.alpha1_prior.tie(self)	

		self.beta0_prior = beta0 if isinstance(beta0, SampledPrior) else FixedValue(beta0)
		self.beta0_prior.tie(self)

		self.beta1_prior = beta1 if isinstance(beta1, SampledPrior) else FixedValue(beta1)
		self.beta1_prior.tie(self)

		self.tension_prior = GammaPrior(8.0, 1.0, 8.0)
		self.tension_prior.tie(self)

		self.p0_prior = BetaPrior(1.0, 99.0, 0.01)
		self.p0_prior.tie(self)

		self.use_alignment_prior = True

		self.data = data
		self.french_vocabulary = french_vocabulary
		self.english_vocabulary = english_vocabulary
		self.document_ids = document_ids
		self.alignments_fixed = False

		self.FV = len(french_vocabulary)
		self.EV = len(english_vocabulary)
		self.D = len(document_ids)
		self.S = len(data)

		self.ttable = [DirichletMultinomial(self.EV, self.beta0) for f in range(self.FV)]
		self.topic_ttables = [[DirichletProcess(self.beta1, self.ttable[f]) for f in range(self.FV)] for k in range(self.K)]
		self.document_topics = [DirichletMultinomial(self.K, self.alpha0) for d in range(self.D)]
		self.sentence_topics = [DirichletProcess(self.alpha1, self.document_topics[d]) for (F, E, d) in data]
		#self.sentence_topics = [DirichletMultinomial(self.K, self.alpha1) for (F, E, d) in data]
		self.topical_probs = [DirichletMultinomial(2, 0.000001) for f in range(self.FV)]
		self.is_topical = []
		self.topic_assignments = []
		self.alignments = []

		for s, (F, E, d) in enumerate(self.data):
			self.alignments.append([None for e in E])
			self.topic_assignments.append([None for e in E])
			self.is_topical.append([None for f in F])
			for m, f in enumerate(F):
				self.assign_topical(s, m, F)
			for n, e in enumerate(E):
				a = numpy.random.randint(0, len(F))
				f = F[a]
				t = self.is_topical[s][a]

				self.alignments[s][n] = a
				if t == 1:
					z = numpy.random.randint(0, self.K)
					self.topic_assignments[s][n] = z
					self.topic_ttables[z][f].increment(e)
					self.sentence_topics[s].increment(z)
				else:
					self.topic_assignments[s][n] = None
					self.ttable[f].increment(e)

	alpha0 = property(lambda self: self.alpha0_prior.x)
	alpha1 = property(lambda self: self.alpha1_prior.x)
	beta0 = property(lambda self: self.beta0_prior.x)
	beta1 = property(lambda self: self.beta1_prior.x)
	tension  = property(lambda self: self.tension_prior.x)
	p0  = property(lambda self: self.p0_prior.x)

	def fix_alignments(self, alignments):
		self.alignments = alignments
		self.alignments_fixed = True

	def draw_alignment(self, s, n, F, E):
		e = E[n]
		z = self.topic_assignments[s][n]
		ttable = self.topic_ttables[z] if z != None else self.ttable
		if self.use_alignment_prior:
			probabilities = []
			for a, f in enumerate(F):
				if f == 0:
					alignment_prob = self.p0
				else:
					alignment_prob = (1.0 - self.p0) * numpy.exp(-self.tension * numpy.abs(1.0 * a / len(F) - 1.0 * n / len(E)))
				translation_prob = self.ttable[f].probability(e)
				probabilities.append(alignment_prob * translation_prob)
		else:
			probabilities = [ttable[f].probability(e) for f in F]
		return draw_from_multinomial(probabilities)
		
	def assign_alignment(self, s, n, F, E):
		e = E[n]
		z = self.topic_assignments[s][n]
		a = self.draw_alignment(s, n, F, E)
		self.alignments[s][n] = a
		ttable = self.topic_ttables[z] if z != None else self.ttable
		ttable[F[a]].increment(e)

	def remove_alignment(self, s, n, F, e):
		a = self.alignments[s][n]
		z = self.topic_assignments[s][n]
		ttable = self.topic_ttables[z] if z != None else self.ttable
		ttable[F[a]].decrement(e)

	def draw_topical(self, s, m, F):	
		f = F[m]
		probabilities = [self.topical_probs[f].probability(t) for t in [0, 1]]	
		return draw_from_multinomial(probabilities)

	def assign_topical(self, s, m, F):
		f = F[m]
		t = self.draw_topical(s, m, F)
		self.is_topical[s][m] = t	
		self.topical_probs[f].increment(t)

	def remove_topical(self, s, m, F):
		f = F[m]
		t = self.is_topical[s][m]
		assert self.topical_probs[f].counts[t] >= 1
		self.topical_probs[f].decrement(t)

	def draw_topic_assignment(self, s, n, F, e):
		a = self.alignments[s][n]
		f = F[a]
		t = self.is_topical[s][a]
		if t == 0:
			return None
		probabilities = [self.sentence_topics[s].probability(z) * \
				 self.topic_ttables[z][f].probability(e) \
				 for z in range(self.K)]
		return draw_from_multinomial(probabilities)

	def assign_topic(self, s, n, F, e):
		a = self.alignments[s][n]
		z = self.draw_topic_assignment(s, n, F, e)
		self.topic_assignments[s][n] = z
		ttable = self.topic_ttables[z] if z != None else self.ttable
		ttable[F[a]].increment(e)
		if z != None:
			self.sentence_topics[s].increment(z)

	def remove_topic_assignment(self, s, n, F, e):
		a = self.alignments[s][n]
		z = self.topic_assignments[s][n]
		ttable = self.topic_ttables[z] if z != None else self.ttable
		ttable[F[a]].decrement(e)
		if z != None:
			self.sentence_topics[s].decrement(z)

	def iterate(self, i):
		if i % 1 == 0:
			resample_attempts = 5
			self.alpha0_prior.resample(resample_attempts)
			print >>sys.stderr, 'Alpha0 is now %f' % self.alpha0

			self.alpha1_prior.resample(resample_attempts)
			print >>sys.stderr, 'Alpha1 is now %f' % self.alpha1
	
			self.beta0_prior.resample(resample_attempts)
			print >>sys.stderr, 'Beta0 is now %f' % self.beta0

			self.beta1_prior.resample(resample_attempts)
			print >>sys.stderr, 'Beta1 is now %f' % self.beta1

			if self.use_alignment_prior:
				self.tension_prior.resample(resample_attempts)
				print >>sys.stderr, 'Tension is now %f' % self.tension

				self.p0_prior.resample(resample_attempts)
				print >>sys.stderr, 'P0 is now %f' % self.p0

		for s, (F, E, d) in enumerate(self.data):
			for m, f in enumerate(F):
				self.remove_topical(s, m, F)
				self.assign_topical(s, m, F)

			for n, e in enumerate(E):
				if not self.alignments_fixed:
					self.remove_alignment(s, n, F, e)	
					self.assign_alignment(s, n, F, E)
				self.remove_topic_assignment(s, n, F, e)
				self.assign_topic(s, n, F, e)

	def output(self):
		for s, (F, E, d) in enumerate(self.data):
			alignment = []
			topics = []
			for n, e in enumerate(E):
				best = (0.0, None, None)
				for z in range(self.K) + [None]:
					for a, f in enumerate(F):
						ttable = self.topic_ttables[z] if z != None else self.ttable
						p = ttable[f].probability(e)
						if p > best[0]:
							best = (p, z, a)
				p, z, a = best
				alignment.append(a)
				topics.append(z)

			for n, e in enumerate(E):
				a = alignment[n]
				if F[a] != 0:
					print '%d-%d' % (a, n),
			print

	def log_likelihood(self):
		def series_sum(l, g, r):
			"""Sum of the first l terms of a geometric series with ratio r, starting at g"""
			return g * (1.0 - r ** l) / (1.0 - r)

		def h(i, j, m, n):
			return -abs(1.0 * i / m - 1.0 * j / n)

		def delta(i, j, m, n):
			return exp(self.tension * h(i, j, m, n))

		log_likelihood = 0.0
		gammaln = scipy.special.gammaln
		log = numpy.log
		exp = numpy.exp
		floor = math.floor

		FV = len(self.french_vocabulary)
		EV = len(self.english_vocabulary)
		D = len(self.document_ids)
		K = self.K

		# Log likelihood of phi and phi0
		#log_likelihood += FV * gammaln(self.beta0 * EV) - FV * EV * gammaln(self.beta0)
		for f in range(FV):
			for k in range(K):
				log_likelihood += self.topic_ttables[k][f].num_tables * log(self.beta1) - gammaln(self.beta1 + self.topic_ttables[k][f].num_customers)

				total_num_tables = 0
				for dish, tables in self.topic_ttables[k][f].tables_by_dish.iteritems(): # dish = e
					num_customers = self.topic_ttables[k][f].customers_by_dish[dish]
					num_tables = len(tables)
					log_likelihood += num_customers * log(self.topic_ttables[k][f].probability(dish))
					log_likelihood += gammaln(num_tables)
					total_num_tables += num_tables

					for table_count in tables:
						log_likelihood += gammaln(table_count)

				log_likelihood -= gammaln(total_num_tables + 1)

		# Log likelihood of theta and theta0
		#log_likelihood += D * gammaln(self.alpha0 * K) - D * K * gammaln(self.alpha0)
		for s, (F, E, d) in enumerate(self.data):
			log_likelihood += self.sentence_topics[s].num_tables * log(self.alpha1) - gammaln(self.alpha1 + self.sentence_topics[s].num_customers)
			for k in range(K):
				log_likelihood += gammaln(len(self.sentence_topics[s].tables_by_dish.get(k, [])) + 1) + self.sentence_topics[s].customers_by_dish.get(k, 0) * log(self.sentence_topics[s].probability(k)) \
					- gammaln(self.sentence_topics[s].num_tables + 1)
			for dish, tables in self.sentence_topics[s].tables_by_dish.iteritems():
				for table_count in tables:
					log_likelihood += gammaln(table_count)

		# Log likelihood of alignment links
		for s, (F, E, d) in enumerate(self.data):
			lf = len([f for f in F if f != 0])
			for n, e in enumerate(E):
				z = self.topic_assignments[s][n]
				a = self.alignments[s][n]
				f = F[a]
				if self.use_alignment_prior:
					r = exp(-1.0 * self.tension / lf)
					j_up = floor(1.0 * n * lf / len(E))
					j_down = j_up + 1.0
					Z = series_sum(j_up + 1, delta(n + 1, j_up + 1, len(E), lf), r) \
						+ series_sum(lf - j_down, delta(n + 1, j_down + 1, len(E), lf), r)
					alignment_prob = (1.0 - self.p0) * delta(n + 1, a + 1, len(E), lf) / Z if f != 0 else self.p0
					log_likelihood += log(alignment_prob)
				else:
					log_likelihood += -log(len(F))

		return log_likelihood

	def old_log_likelihood(self):
		log_likelihood = 0.0	

		S = len(self.data)
		for s, (F, E, d) in enumerate(self.data):	
			for n, e in enumerate(E):
				z = self.topic_assignments[s][n]
				a = self.alignments[s][n]
				f = F[a]

				ttable = self.topic_ttables[z] if z != None else self.ttable
				log_likelihood += numpy.log(ttable[f].probability(e))
				if z != None:
					log_likelihood += numpy.log(self.sentence_topics[s].probability(z))
			#log_likelihood += dirichlet_log_prob([self.sentence_topics[s].probability(k) for k in range(self.K)], self.alpha)

		for f in range(self.FV):
			for k in range(self.K):
				#log_likelihood += dp_log_prob(self.topic_ttables[k][f], self.ttable[f], self.beta1)
				pass
			#log_likelihood += dirichlet_log_prob([self.ttable[f].probability(e) for e in range(self.EV)], self.beta0)

		return log_likelihood
		

	def perplexity(self):
		log_likelihood = self.log_likelihood()
		word_count = sum(len(E) for (F, E, d) in data)
		return math.exp(-log_likelihood / word_count)

	def get_parameters(self):
		return (self.alpha0, self.alpha1, self.beta0, self.beta1)

def load_data(filename, use_null):
	data = []
	french_vocabulary = Vocabulary()
	english_vocabulary = Vocabulary()
	document_ids = Vocabulary()

	stream = open(filename)
	line = stream.readline()
	useDocumentIds = len(line.split('|||')) == 3
	sentence_number = 1
	while line:
		parts = [part.strip() for part in line.split('|||')]
		if not useDocumentIds:
			document_id = document_ids.getId('sentence_%d' % sentence_number)
		elif len(parts) == 3:
			document_id = document_ids.getId(parts[0])
			parts = parts[1:]

		F, E = [part.split() for part in parts]
		F = [french_vocabulary.getId(w) for w in F if len(w) != 0]
		E = [english_vocabulary.getId(w) for w in E if len(w) != 0]

		if use_null:
			F += [0]

		data.append(ParallelSentence(F, E, document_id))
		line = stream.readline()
		sentence_number += 1
	
	stream.close()
	return data, french_vocabulary, english_vocabulary, document_ids

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
	parser.add_argument('--num_iterations', type=int, default=20)
	parser.add_argument('--num_topics', type=int, default=2)
	parser.add_argument('--alpha0', type=float, default=0.1)
	parser.add_argument('--alpha1', type=float, default=0.9)
	parser.add_argument('--beta0', type=float, default=0.00026)
	parser.add_argument('--beta1', type=float, default=0.1)
	parser.add_argument('--aligns')
	parser.add_argument('--nonull', action='store_true')
	args = parser.parse_args()

	allow_null = not args.nonull
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	print >>sys.stderr, 'Loading data...'
	data, french_vocabulary, english_vocabulary, document_ids = load_data(args.corpus, allow_null)

	print >>sys.stderr, 'Initializing model...'
	model = DirichletModel1WithTopics(args.num_topics, GammaPrior(1.0, 1.0, args.alpha0), GammaPrior(1.0, 1.0, args.alpha1), GammaPrior(1.0, 1.0, args.beta0), GammaPrior(1.0, 1.0, args.beta1), \
						data, french_vocabulary, english_vocabulary, document_ids)

	if args.aligns != None:
		print >>sys.stderr, 'Loading gold alignments...'
		gold_alignment = load_alignment(args.align, data)
		model.fix_alignments(gold_alignment)

	output_iteration(0, model, args.output_dir)
	for i in range(1, args.num_iterations + 1):
		model.iterate(i)
		output_iteration(i, model, args.output_dir)
	print >>sys.stderr, 'Final parameter values:', model.get_parameters()
