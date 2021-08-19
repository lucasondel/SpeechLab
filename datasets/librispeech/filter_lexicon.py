
'''Remove the lexical stress marker from the lexicon.'''

import argparse

LEXICAL_MARKERS = set(['0', '1', '2'])

def main(args):
    with open(args.lexicon, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            print(tokens[0], '\t', end='')
            prefix = ''
            for token in tokens[1:]:
                if token[-1] in LEXICAL_MARKERS:
                    print(prefix, token[:-1], sep='', end='')
                else:
                    print(prefix, token, sep='', end='')
                prefix = ' '
            print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lexicon', help='input lexicon')
    args = parser.parse_args()
    main(args)
