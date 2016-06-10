import sys
import nltk

while True:
	sentence = sys.stdin.readline()
	if not sentence:
		break
	sentence = sentence.strip().split()
	senses = sys.stdin.readline().strip().split()
	pos_tags = [tag for (word, tag) in nltk.pos_tag(sentence)]
	senses = [sense if pos[0] == 'N' else '-' for (word, pos, sense) in zip(sentence, pos_tags, senses)]
	print ' '.join(sentence)
	print ' '.join(senses)
