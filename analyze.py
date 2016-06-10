import sys
from math import log
from collections import namedtuple, defaultdict

WordInfo = namedtuple('WordInfo', 'sense_given_topic, ttables')

class ErrorFileParser:
	def __init__(self):
		self.reset()

	def reset(self):
		self.word_info = defaultdict(lambda: WordInfo(defaultdict(lambda: defaultdict(lambda: (0.0, 0.0, 0.0))), defaultdict(lambda: defaultdict(lambda: (0.0, 0.0, 0.0)))))
		self.topic_probs = {}

	def parse_ttable(self, stream, iteration):
		sense = 0
		for line in stream:
			if line.startswith('===') and 'END' in line:
				break
			elif line.startswith('==='):
				sense += 1
				continue
			elif line.startswith('\t'):
				parts = line.split('\t')
				if len(parts) == 5:
					_, target_word, prob, customers, tables = parts
				else:
					_, target_word, prob = parts
					customers = 0
					tables = 0
				prob = float(prob)
				customers = float(customers)
				tables = float(tables)

				prev_prob, prev_customers, prev_tables = self.word_info[source_word].ttables[sense][target_word]
				prev_prob *= (iteration - 1.0)
				prev_customers *= (iteration - 1.0)
				prev_tables *= (iteration - 1.0)

				new_prob = (prev_prob + prob) / iteration
				new_customers = (prev_customers + customers) / iteration
				new_tables = (prev_tables + tables) / iteration
				self.word_info[source_word].ttables[sense][target_word] = (new_prob, new_customers, new_tables)
			else:
				source_word, customers, tables = line.split('\t')
				customers = float(customers)
				tables = float(tables)
		self.num_senses = sense

	def parse_topic_sense_probs(self, stream, iteration):
		topic = 0
		for line in stream:
			if line.startswith('===') and 'END' in line:
				break
			elif line.startswith('==='):
				topic += 1
				continue
			elif line.startswith('\t'):
				_, sense, prob, customers, tables = line.split('\t')
				sense = int(sense[sense.find('#') + 1 :]) + 1
				prob = float(prob)
				customers = float(customers)
				tables = float(tables)

				prev_prob, prev_customers, prev_tables = self.word_info[source_word].sense_given_topic[topic][sense]
				prev_prob *= (iteration - 1.0)
				prev_customers *= (iteration - 1.0)
				prev_tables *= (iteration - 1.0)
				
				new_prob = (prev_prob + prob) / iteration
				new_customers = (prev_customers + customers) / iteration
				new_tables = (prev_tables + tables) / iteration
				self.word_info[source_word].sense_given_topic[topic][sense] = (new_prob, new_customers, new_tables)
			else:
				source_word, customers, tables = line.split('\t')
				customers = float(customers)
				tables = float(tables)

	def parse_document_topic_probs(self, stream, iteration):
		topic_num = 0
		doc_id = '__underlying__'
		self.topic_probs[doc_id] = []

		for line in stream:
			if line.startswith('===') and 'END' in line:
				break
			else:
				if line.startswith('\t'):
					_, _, prob = line.split('\t')
					self.topic_probs[doc_id].append(float(prob))
				else:
					doc_id = line.strip()
					self.topic_probs[doc_id] = []

	def get_underlying_ttable(self, source_word):
		word_info = self.word_info[source_word]
		return {target_word: prob for target_word, (prob, _, _) in word_info.ttables[0].iteritems()}

	def get_topical_ttables(self, source_word):
		word_info = self.word_info[source_word]
		underlying_ttable = self.get_underlying_ttable(source_word)
		topical_ttables = {}
		for topic, sense_info in word_info.sense_given_topic.iteritems():
			topical_ttable = {}
			for target_word in underlying_ttable.keys():
				topical_ttable[target_word] = 0.0
				for z in sense_info.keys():
					sense_prob = sense_info[z][0]
					if target_word in word_info.ttables[z]:
						word_prob = word_info.ttables[z][target_word][0]
					else:
						word_prob = 0.0
					topical_ttable[target_word] += sense_prob * word_prob
			topical_ttables[topic] = topical_ttable
		return topical_ttables

	def parse(self, stream):
		iteration = 0
		for line in stream:
			if line.startswith('===') and 'BEGIN' in line:
				if 'TTABLE' in line:
					iteration += 1	
					self.parse_ttable(stream, iteration)
				elif 'TOPIC-SENSE PROBS' in line:
					self.parse_topic_sense_probs(stream, iteration)
				elif 'DOCUMENT TOPIC PROBS' in line:
					self.parse_document_topic_probs(stream, iteration)
				else:
					continue
			elif line.startswith('Resampled diagonal alignment parameters '):
				line = line[line.find('(') + 1 : line.find(')') - 1]
				parts = line.split(',')
				self.p0 = float(parts[0].split('=')[1])
				self.tension = float(parts[1].split('=')[1])
			elif line.startswith('Number of topics:'):
				self.num_topics = int(line.split(':')[1])
		

def log2(f):
        return log(f) / log(2.0)

def H(P):
        h = 0.0
        for p in P.values():
                if p != 0.0:
                        h += p * log2(p)
        return -h

def JSD(D, weights = None):
	weights = weights if weights is not None else [1.0 / len(D) for d in D]
        epsilon = 1.0 / 80000

        keys = set()
        for d in D:
                keys = keys | set(d.keys())

        M = {}
        for k in keys:
                M[k] = sum([weights[i] * D[i][k] if k in D[i] else epsilon for i in range(len(D))])

        return H(M) - sum([weights[i] * H(D[i]) for i in range(len(D))])

if __name__ == "__main__":
	error_file = ErrorFileParser()
	error_file.parse(sys.stdin)
	word_info = error_file.word_info

	for source_word, word_info in error_file.word_info.iteritems():
		underlying_ttable = error_file.get_underlying_ttable(source_word)
		topical_ttables = error_file.get_topical_ttables(source_word)
		sense_given_topic = error_file.word_info[source_word].sense_given_topic

		table_counts = {d: sum(sense_given_topic[d][k][1] for k in sense_given_topic[d].keys()) for d in sense_given_topic.keys()}
		total_tables = sum(v for d, v in table_counts.iteritems() if d != 0)
		weights = {d: 0.5 if d == 0 else 0.5 * v / total_tables for d, v in table_counts.iteritems()}

		# This prints the target word's underlying ttable, as well as its marginal ttables per TOPIC
		sys.stdout.write('%s\t%.4f\tunderlying\t' % (source_word, JSD(topical_ttables.values(), weights.values()) * log(total_tables)))
		for topic in range(len(topical_ttables)):
			sys.stdout.write('topic #%d\t' % topic)
		sys.stdout.write('\n')
		for target_word in sorted(underlying_ttable.keys(), key=lambda k: underlying_ttable[k], reverse=True):
			sys.stdout.write('\t%s\t%f' % (target_word, underlying_ttable[target_word]))
			for topic, topical_ttable in topical_ttables.iteritems():
				sys.stdout.write('\t%f' % topical_ttable[target_word])
			sys.stdout.write('\n')
		sys.stdout.flush()
		sys.stdout.write('\n')

		ttables = [{key: value[0] for key, value in ttable.iteritems()} for ttable in error_file.word_info[source_word].ttables.values()]	
		sys.stdout.write('%s\t%.4f\tunderlying\t' % (source_word, JSD(ttables) * log(total_tables)))
		for sense in range(1, len(ttables)):
			sys.stdout.write('sense #%d\t' % sense)
		sys.stdout.write('\n')
		for target_word in sorted(ttables[0].keys(), key=lambda k: ttables[0][k], reverse=True):
			sys.stdout.write('\t%s' % (target_word))
			for sense, ttable in enumerate(ttables):
				sys.stdout.write('\t%f' % (ttable[target_word] if target_word in ttable else 0))
			sys.stdout.write('\n')
		sys.stdout.flush()
		sys.stdout.write('\n')
