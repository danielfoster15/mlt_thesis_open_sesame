#!/usr/bin/env python

from __future__ import division

import sys
import numpy as np
from numpy import linalg as LA
from io import BufferedReader, FileIO
from itertools import takewhile, count
import mimetypes


class TKWV:
    def __init__(self, inputfile='', lowercase=False, normalizVector=1):
        '''
        word_vector object
        build from a eithor a wordvector binary or txt file
        '''
        self.lowercase = lowercase
        mimetype = mimetypes.guess_type(inputfile)
        # print("read word vection from a text file")
        self.read_embeddings_from_text(inputfile, normalizVector)

        self.vocab_to_index = {}
        for i, word in enumerate(self.vocab):
            self.vocab_to_index[word] = i

    def get_vocab_size(self):
        return self.vocab_size

    def get_vector_size(self):
        return self.vector_size

    def get_index(self, word):
        return self.vocab_to_index[word]

    def get_vector(self, word):
        vector_index = self.get_index(word)
        return self.vectors[vector_index]

    def get_word(self, index):
        return self.vocab[index]

    def update_embedding(self, embedding):
        self.vectors = embedding

    def __contains__(self, word):
        return word in self.vocab_to_index

    def cosine(self, inputword_vec, n=10):
        metrics = np.dot(self.vectors, inputword_vec.T)
        best = np.argsort(metrics)[::-1][1:n + 1]
        best_metrics = metrics[best]
        return best, best_metrics

    def read_embeddings_from_binary(self, filename, normalizVector, vocabUnicodeSize=78):
        with open(filename, 'rb') as fin:
            header = fin.readline()

            vocab_size, vector_size = list(map(int, header.split()))
            self.vocab_size = vocab_size
            self.vector_size = vector_size

            self.vocab = np.empty(vocab_size, dtype='<U%s' % vocabUnicodeSize)
            self.vectors = np.empty((vocab_size, vector_size), dtype=np.float)

            # print (vocab_size, vector_size)
            binary_len = np.dtype(np.float32).itemsize * vector_size
            for i in range(vocab_size):
                word = b''
                ch = fin.read(1)
                while ch != b' ':
                    word += ch
                    ch = fin.read(1)
                vector = np.fromstring(fin.read(binary_len), dtype=np.float32)
                fin.read(1)
                if self.lowercase:
                    self.vocab[i] = word.decode('utf-8').lower()
                else:
                    self.vocab[i] = word.decode('utf-8')

                if normalizVector:
                    self.vectors[i] = unitvec(vector)
                else:
                    self.vectors[i] = vector
        print("read {} tokens with vector size {} from {}".format(
            vocab_size, vector_size, filename))

    def read_embeddings_from_text(self, filename, normalizVector, vocabUnicodeSize=78):
        with open(filename, 'r') as fin:
            lines = fin.readlines()
            vocab_size = len(lines)
            vector_size = len(list(map(float, lines[0].split()[1:])))
            self.vocab_size = vocab_size
            self.vector_size = vector_size
        with open(filename, 'r') as fin:
            self.vocab = np.empty(vocab_size, dtype='<U%s' % vocabUnicodeSize)
            self.vectors = np.empty((vocab_size, vector_size), dtype=np.float)
            for i, line in enumerate(fin.readlines()):
                line = line.strip()
                parts = line.split(' ')
                word = parts[0]
                vector = np.array(parts[1:], dtype=np.float)

                if self.lowercase:
                    self.vocab[i] = word.lower()
                else:
                    self.vocab[i] = word

                if normalizVector:
                    self.vectors[i] = unitvec(vector)
                else:
                    self.vectors[i] = vector
        print("read {} tokens with vector size {} from {}".format(
            vocab_size, vector_size, filename))

    def write_embeddings_to_text(self, output_filename):
        with open(output_filename, 'w') as f:
            f.write(str(self.vocab_size) + ' ' + str(self.vector_size) + "\n")
            for i in range(self.vocab_size):
                f.write(
                    self.vocab[i] + " " + " ".join(list(map(str, self.vectors[i]))) + "\n")

    def write_embeddings_to_binary(self, output_filename):
        # TODO
        # Currently I am using textractor/TKSrc/word2vec/word2vec-r35/txt-to-bin to convert a txt to bin
        print("TODO, not implemented")


def unitvec(vec):
    return (1.0 / LA.norm(vec, ord=2)) * vec
