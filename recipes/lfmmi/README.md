# Lattice Free - Maximum Mutual Information (LF-MMI) training

This recipe trains a neural network using sequence-discriminative
training based on the so-called [LF-MMI](https://www.isca-speech.org/archive/Interspeech_2016/pdfs/0595.PDF)
procedure.

## Data preparation

First, you need to prepare a dataset, say [Mini LibriSpeech](https://www.openslr.org/31/),
by doing:
```
$ ../../datasets/mini_librispeech/prepare.sh /path/to/datasets/mini_librispeech
```
This will prepare 3 folders:
```
$ ls /path/to/datasets/mini_librispeech
dev test train
```

