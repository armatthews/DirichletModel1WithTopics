import sys
import argparse
from collections import defaultdict

def argmax(d):
	best = None
	for k, v in d.iteritems():	
		if best == None or v > best[1]:	
			best = (k, v)
	return best

parser = argparse.ArgumentParser()
parser.add_argument('sentence_count', type=int)
parser.add_argument('-b', '--burn_in', type=int, default=0)
parser.add_argument('-f', '--fast', action='store_true')
args = parser.parse_args()

iteration = 0
sentence_index = 0
line_num = 1

link_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

sys.stderr.write('Taking the mode over %d sentences with a burn-in of %d iterations\n' % (args.sentence_count, args.burn_in))
sys.stderr.write('Reading iteration %d\r' % 0)
for line in sys.stdin:
	if iteration >= args.burn_in:
		n = (line_num - 1) % (args.sentence_count)
		if n < 150 or not args.fast:
			line = line.strip()
			links = line.split(' ')
			links = [int(link) for link in links]
			for j, i in enumerate(links):
				link_counts[n][j][i] += 1

	if line_num % (args.sentence_count) == 0:
		sentence_index = 0
		iteration += 1
		sys.stderr.write('Reading iteration %d\r' % iteration)

	line_num += 1
sys.stderr.write('\n')

for n in range(args.sentence_count):
	for j in sorted(link_counts[n].keys()):
		i, v = argmax(link_counts[n][j])
		print i,
	print
