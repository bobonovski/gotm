#-*- coding: utf-8 -*-
"""print topic K words of each topic"""
import argparse
from collections import defaultdict
import operator

def print_topics(model_file, vocab_file, topk):
    vocab = {}
    for i, word in enumerate(open(vocab_file, 'r')):
        vocab[i+1] = word.strip()

    topics = defaultdict(list)
    for i, line in enumerate(open(model_file, 'r')):
        if i == 0:
            continue
        info = line.strip().split(',')
        if int(info[0]) not in vocab:
            continue
        topics[info[1]].append((vocab[int(info[0])], float(info[2])))
    for k, v in topics.items():
        sorted_value = sorted(v, key=operator.itemgetter(1), reverse=True)
        top_words = map(lambda x: x[0]+':'+str(x[1]), sorted_value[0:topk])
        print k, ' '.join(top_words)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('print top k words of trained topic models')
    parser.add_argument('--model_file', help='word topic distribution file')
    parser.add_argument('--vocab_file', help='vocabulary file')
    parser.add_argument('--topk', type=int, default=10, help='top k')
    args = parser.parse_args()

    print_topics(args.model_file, args.vocab_file, args.topk)
