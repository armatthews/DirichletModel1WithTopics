import sys
from math import exp
from analyze import *

def parse_alignment(line):
	links = line.split()
	alignment = set()
	for link in links:
		i, j = link.split('-')
		i, j = int(i), int(j)
		alignment.add((i, j))
	return alignment

def alignment_prob(i, j, m, n, p0, tension):
	if i == 0:
		return p0
	return (1.0 - p0) * exp(-tension * abs(1.0 * i / m - 1.0 * j / n))

error_file = ErrorFileParser()
error_file.parse(open(sys.argv[1]))
topic_probs = error_file.topic_probs
word_info = error_file.word_info
num_senses = 5
p0 = .08008338235294117647
tension = 6.46652647058823529411
alignments_file = open(sys.argv[2])

for input_line, alignment_line in zip(sys.stdin, alignments_file):
	parts = input_line.strip().split(' ||| ')
	if len(parts) == 3:
		document_id, source, target = parts
	else:
		document_id = ''
		source, target = parts

	source = source.strip().split()
	target = target.strip().split()

	alignment_links = parse_alignment(alignment_line)
	alignment = [[j for (i, j) in alignment_links if i == w] for w, word in enumerate(source)]
	senses = []
	for w, word in enumerate(source):
		z_probs = [0.0 for z in range(num_senses)]
		for z in range(num_senses):
			for t, p in enumerate(topic_probs[document_id]):
				z_probs[z] += p * word_info[word].sense_given_topic[t][z + 1][0]
				print >>sys.stderr, 'p(t=%d|doc=%s)\t\t= %f' % (t, document_id, p)
				print >>sys.stderr, 'p(z=%d|t=%d)\t\t= %f' % (z, t, word_info[word].sense_given_topic[t][z][0])
			for a in alignment[w]:
				z_probs[z] *= alignment_prob(a + 1, w + 1, len(target), len(source), p0, tension)
				z_probs[z] *= word_info[word].ttables[z + 1][target[a]][0]
				print >>sys.stderr, 'p(a=%d|i=%d,m=%d,n=%d)\t= %f' % (a, w, len(target), len(source), alignment_prob(a + 1, w + 1, len(target), len(source), p0, tension))
				print >>sys.stderr, 'p(e_a=%s|f=%s,z=%d)\t= %f' % (target[a], word, z, word_info[word].ttables[z + 1][target[a]][0])
		z_probs = [p / sum(z_probs) for p in z_probs]
		print >>sys.stderr, word, z_probs
		senses.append(max(enumerate(z_probs), key=lambda (z, p): p)[0])

	for w, word in enumerate(source):
		print word,
	print
	for w, word in enumerate(source):
		print senses[w],
	print
