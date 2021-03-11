import os
import re
import sys

import nltk
import pandas as pd
from nltk.corpus import stopwords


class DataParser:
    """
    Parsing datasets, base class
    Class param `path` should be overwritten
    """

    path = None
    encodings = ('utf-8', 'windows-1250', 'windows-1252')
    keywords = ('From', 'Subject', 'Summary', 'Keywords', 'Expires',
                'Distribution', 'Organization', 'Supersedes', 'Lines')

    email_regex = r'^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'
    parsed_data_csv_path = None
    topics_path = 'topics.csv'

    def __init__(self):
        if not self.path:
            raise NotImplementedError

        self.data = pd.DataFrame([], columns=['topic', 'text'])
        self.topics = []
        self._current_topic = None
        self._current_tokens = []

    def parse(self):
        """Parse article into words"""

        for topic in os.listdir(self.path):
            print(f'parsing topic {topic}')
            self._current_topic = topic
            self.topics.append(topic)

            articles = os.listdir(os.path.join(self.path, topic))
            for article in articles:
                self._read_file(os.path.join(self.path, topic, article))

        return self

    def save_to_csv(self):
        if not self.parsed_data_csv_path:
            raise NotImplementedError

        self.data.to_csv(self.parsed_data_csv_path, index=False)
        codes = pd.Categorical(self.topics).codes
        data = [(topic, int(code)) for topic, code in zip(self.topics, codes)]

        t = pd.DataFrame(data=data, columns=['topic', 't_id'])

        if os.path.isfile(self.topics_path):
            topics = pd.read_csv(self.topics_path, index_col=False)
            t = t.append(topics)
            t.drop_duplicates(inplace=True)

        t.to_csv(self.topics_path, index=False)

    def _read_file(self, file_path):
        for encoding in self.encodings:
            with open(file_path, encoding=encoding) as f:
                try:
                    self._current_tokens = nltk.sent_tokenize(f.read())
                    self._parse_words_from_tokens()
                    break
                except UnicodeDecodeError:
                    print(file_path, f'wrong encoding {encoding}, trying next')
                    self._current_tokens = []
                    continue

    def _parse_words_from_tokens(self):
        clean_tokens = []

        for sentence in self._current_tokens:
            if self._check_if_contain_keywords(sentence):
                continue

            sentence = self._replace_email(sentence)
            words = nltk.word_tokenize(sentence)
            words = [w for w in words if w not in stopwords.words('english')
                     and w.isalnum()]

            clean_tokens.extend(words)

        text = ' '.join(clean_tokens)
        new_text = pd.DataFrame(data=[[self._current_topic, text]],
                                columns=['topic', 'text'])
        self.data = self.data.append(new_text)

    def _check_if_contain_keywords(self, sentence):
        for k in self.keywords:
            if f'{k}:' in sentence:
                return True

        return False

    def _replace_email(self, sentence):
        return re.sub(self.email_regex, 'email', sentence)


if __name__ == '__main__':
    parser = DataParser()
    parser.parse()
    sys.exit()
