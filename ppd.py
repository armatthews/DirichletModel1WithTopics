import sys
import argparse
from math import log, exp
from collections import defaultdict
from analyze import ErrorFileParser

parser = argparse.ArgumentParser()
parser.add_argument('error_file')
args = parser.parse_args()

class DiagonalAlignmentPrior:
	def __init__(self, tension, p0):
		self.p0 = p0
		self.tension = tension

	def prob(self, i, j, m, n):
		# i is index into target sentence
		# j is index into source sentence
		# m is length of target sentence
		# n is length of source sentence (not counting NULL)
		return (1.0 - self.p0) * exp(-self.tension * abs(1.0 * i / m - 1.0 * j / n))

	def null_prob(self, i, m, n):
		return self.p0

def get_initial_topics_and_senses(error_file, document, source, target):
	if document in error_file.topic_probs:
		topic_probs = error_file.topic_probs[document]
	else:
		topic_probs = error_file.topic_probs['__underlying__']

	best_topic = max(range(len(topic_probs)), key=lambda i: topic_probs[i])
	topics = []
	for s in source:
		topics.append(best_topic)

	senses = []
	for s, topic in zip(source, topics):
		# topic = 0 is underlying
		sense_given_topic = error_file.word_info[s].sense_given_topic[topic + 1]
		if len(sense_given_topic) ==  0:
			sense = 0
		else:
			sense = max(sense_given_topic.iterkeys(), key=lambda k: sense_given_topic[k][0]) - 1
		senses.append(sense)

	return (topics, senses)

def get_alignments(error_file, document, source, target, senses, prior):
	def trans_prob(s, t, z):
		if s in error_file.word_info:
			if t in error_file.word_info[s].ttables[z + 1]:
				return error_file.word_info[s].ttables[z + 1][t][0]
			else:
				return error_file.word_info[s].ttables[z + 1]['[other]'][0]
		else:
			return 1.0e-1000
	alignment = []
	rev_alignment = defaultdict(set)
	for i, t in enumerate(target):
		alignment_probs = [prior.null_prob(i, len(target), len(source) - 1)] + [prior.prob(i, j + 1, len(target), len(source) - 1) for j in range(len(source) - 1)]
		# ttables[0] is underlying
		# TODO: Factor in diagonal prior
		a = max(range(len(source)), key=lambda j: trans_prob(source[j], t, senses[j]) * alignment_probs[j])
		rev_alignment[a].add(i)
		alignment.append(a)

	return (alignment, rev_alignment)

def get_topics_and_senses(error_file, document, source, target, rev_alignment, prior):
	topics = []
	senses = []
	total_score = 0.0
	for i, s in enumerate(source):
		best = None
		if document in error_file.topic_probs:
			topic_probs = error_file.topic_probs[document]
		else:
			topic_probs = error_file.topic_probs['__underlying__']
		for topic, topic_prob in enumerate(topic_probs):
			sense_probs = error_file.word_info[s].sense_given_topic[topic + 1].items() if s in error_file.word_info else [(1, (0, 0, 0))]
			if len(sense_probs) == 0:
				sense_probs = [(1, (0, 0, 0))]
			for sense, (sense_prob, _, __) in sense_probs:
				sense -= 1
				score = log(topic_prob) + log(sense_prob) if sense_prob > 0 else log(topic_prob) - 1000.0
				for a in rev_alignment[i]:
					ttable = error_file.word_info[s].ttables[sense + 1]
					link_prob = ttable[target[a]][0] if target[a] in ttable else ttable['[other]'][0]
					link_prob *= prior.prob(a, i, len(target), len(source) - 1) if i != 0 else prior.null_prob(a, len(target), len(source) - 1)
					score += log(link_prob) if link_prob > 0 else -1000.0
				#print s, topic, sense, sorted(list(rev_alignment[i])), score
				if best == None or score >= best[0]:
					best = (score, topic, sense)
			
		score, topic, sense = best
		topics.append(topic)	
		senses.append(sense)
		total_score += score
	return (topics, senses, total_score)

parser = ErrorFileParser()
parser.parse(open(args.error_file))

for line in sys.stdin:
	document, source, target = [part.strip() for part in line.split('|||')]
	source = [word.strip() for word in source.split() if len(word) > 0]
	source = ['<bad0>'] + source
	target = [word.strip() for word in target.split() if len(word) > 0]
	prior = DiagonalAlignmentPrior(parser.tension, parser.p0)

	topics, senses = get_initial_topics_and_senses(parser, document, source, target)
	alignment, rev_alignment = get_alignments(parser, document, source, target, senses, prior)

	for i in range(10):
		new_topics, new_senses, score = get_topics_and_senses(parser, document, source, target, rev_alignment, prior)
		new_alignment, new_rev_alignment = get_alignments(parser, document, source, target, senses, prior)
		if new_topics == topics and new_senses == senses and new_alignment == alignment:
			break
		else:
			topics = new_topics
			senses = new_senses
			alignment = new_alignment
			rev_alignment = new_rev_alignment

	print ' '.join(source[1:])
	print ' '.join(map(str, topics[1:]))
	print ' '.join(map(str, senses[1:]))
	print ' '.join(map(str, alignment))
	print ' '.join(target)
	print score
	print
