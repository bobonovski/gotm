#-*- coding: utf-8 -*-
"""convert common bag of words format to gotm format

   some popular dataset (for example, dataset from UCI repository) 
   has the following original format:
   ---
    D
    W
    NNZ
    docID wordID count
    docID wordID count
    docID wordID count
    docID wordID count
    ...
    docID wordID count
    docID wordID count
    docID wordID count
   ---

   after processing with this script, the output format will be:
   ---
    docID wordID:count wordID:count ... wordID:count
    docID wordID:count wordID:count ... wordID:count
   ---
   
   gotm assumes the docID start from zero, so the original docID could
   be changed after processing
"""
import argparse
from collections import defaultdict

def process(data_file, output_file):
    output = open(output_file, 'w')
    docs = defaultdict(list)
    for i, line in enumerate(open(data_file, 'r')):
        if i < 3:
            continue
        info = line.strip().split(' ')
        docs[int(info[0])-1].append('{}:{}'.format(info[1], info[2]))
    for k, v in docs.items():
        output.write('%s %s\n' % (k, ' '.join(v)))
    output.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('topic model dataset format converter')
    parser.add_argument('--data_file', help='data file with original BOW format')
    parser.add_argument('--output_file', help='converted output file')
    args = parser.parse_args()

    process(args.data_file, args.output_file)
