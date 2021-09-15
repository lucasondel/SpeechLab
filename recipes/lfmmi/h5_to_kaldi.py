'''Convert an HDF5 archive to a Kaldi archive.'''

import argparse
import h5py
import kaldi_io
import numpy as np

def main(args):
    with h5py.File(args.h5archive, 'r') as fin:
        with open(args.karchive, 'wb') as fout:
            for key in fin.keys():
                kaldi_io.write_mat(fout, np.array(fin[key]), key=key.replace('lbi-', ''))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('h5archive', help='input HDF5 archive')
    parser.add_argument('karchive', help='output Kaldi archive')
    args = parser.parse_args()
    main(args)

