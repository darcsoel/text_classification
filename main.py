import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC

if __name__ == '__main__':
    topics = pd.read_csv('topics.csv', index_col='topic')

    train_data = pd.read_csv('train_data.csv', index_col='topic')
    train_data = train_data.dropna()
    train_result = pd.merge(train_data, topics, on='topic')

    test_data = pd.read_csv('test_data.csv', index_col='topic')
    test_data = test_data.dropna()
    test_result = pd.merge(test_data, topics, on='topic')

    vectorizer = CountVectorizer()
    vectorizer = vectorizer.fit(train_result['text'])
    x_count = vectorizer.transform(train_result['text'])

    tf_transformer = TfidfTransformer()
    tf_transformer = tf_transformer.fit(x_count)
    x_train_tfidf = tf_transformer.transform(x_count)

    clf = LinearSVC()
    clf.fit(x_train_tfidf, train_result['t_id'])

    x_count_test = vectorizer.transform(test_result['text'])
    x_test_tfidf = tf_transformer.transform(x_count_test)
    predict = clf.predict(x_test_tfidf)

    accuracy = accuracy_score(test_result['t_id'], predict)
    print(f'Accuracy = {accuracy}')

    matrix = confusion_matrix(test_result['t_id'], predict)
    sns.heatmap(matrix.T, fmt='d', cbar=False, square=True, annot=True)
    plt.xlabel('true')
    plt.ylabel('predicted')
    plt.show()

    docs_new = ['God is love', 'OpenGL on the GPU is fast']
    for doc, pred in zip(docs_new, predict):
        predicted = topics[topics['t_id'] == pred].index.values[0]
        print(f'{doc} -> {predicted}')

    sys.exit()
