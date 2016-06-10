import sys
import random
import argparse
from collections import defaultdict

def build_sense_map(hyp_stream, ref_stream):
	sense_map = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
	for line_num, (hyp_line, ref_line) in enumerate(zip(hyp_stream, ref_stream)):
		hyp_line = hyp_line.strip()
		ref_line = ref_line.strip()
		if line_num % 2 == 0:
			sentence = hyp_line.split()
			continue

		hyp_senses = hyp_line.split()	
		ref_senses = ref_line.split()

		for w, h, r in zip(sentence, hyp_senses, ref_senses):
			if r == '-' or h == '-':	
				continue
			sense_map[w][r][h] += 1
	return sense_map

parser = argparse.ArgumentParser()
parser.add_argument('hyp_file')
parser.add_argument('ref_file')
parser.add_argument('-c', '--count_singletons', action='store_true')
args = parser.parse_args()

#sense_map = build_sense_map(open(args.hyp_file), open(args.ref_file))
inv_sense_map = build_sense_map(open(args.ref_file), open(args.hyp_file))

"""for word, senses in sense_map.iteritems():
	if len(senses.keys()) == 1:
		key = senses.keys()[0]
		if sum(senses[key].values()) == 1:
			#del senses[key]
			pass
		if len(senses[key].keys()) == 1:
			pass"""

total, total_correct = 0, 0
#for word, senses in sense_map.iteritems():
#	for r, hyp_counts in senses.iteritems():
#		for h, count in hyp_counts.iteritems():
for word, senses in inv_sense_map.iteritems():
	for h, ref_counts in senses.iteritems():
		for r, count in ref_counts.iteritems():
			assert h != '-'
			assert inv_sense_map[word][h][r] >= 1

			# We can pretend that we're looking at a single instance of this word/hyp combo
			# We want to forget that what we learned about this instance before doing ``inference''
			inv_sense_map[word][h][r] -= 1

			best_guess = max(inv_sense_map[word][h].keys(), key=lambda x: inv_sense_map[word][h][x])

			# If we only saw this word/hyp combo once, then subtracted off one,
			# then now the count is zero.
			is_singleton = inv_sense_map[word][h][best_guess] == 0

			# Add back in what we subtracted out
			inv_sense_map[word][h][r] += 1

			if not is_singleton or args.count_singletons:
				if r == best_guess:	
					total_correct += count
				total += count
print '%d/%d (%.2f%%)' % (total_correct, total, 100.0 * total_correct / total)
