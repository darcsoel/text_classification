import os
import sys

import nltk
import sklearn
from nltk import word_tokenize


class DataParser:
    path = '20news-bydate-train'
    encodings = ('utf-8', 'windows-1250', 'windows-1252')

    def __init__(self):
        self.data = {}

    def parse(self):
        self._read_file('20news-bydate-train/alt.atheism/51060')

    def _read_file(self, file_path):
        for encoding in self.encodings:
            with open(file_path, encoding=encoding) as f:
                try:
                    tokens = nltk.sent_tokenize(f.read())
                except UnicodeDecodeError:
                    print(file_path, f'wrong encoding {encoding}, trying next')
                    continue



if __name__ == '__main__':
    parser = DataParser()
    parser.parse()
    sys.exit()
