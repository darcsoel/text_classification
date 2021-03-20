import multiprocessing
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def cpu_count():
    cpu_count_ = multiprocessing.cpu_count()
    if 2 < cpu_count_ <= 4:
        cpu_count_ -= 1
    elif 4 < cpu_count_ < 8:
        cpu_count_ -= 2
    else:
        cpu_count_ -= 4
    return cpu_count_


if __name__ == '__main__':
    topics = pd.read_csv('topics.csv', index_col='topic')

    train_data = pd.read_csv('train_data.csv', index_col='topic')
    train_data = train_data.dropna()
    train_result = pd.merge(train_data, topics, on='topic')

    test_data = pd.read_csv('test_data.csv', index_col='topic')
    test_data = test_data.dropna()
    test_result = pd.merge(test_data, topics, on='topic')

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    parameters = {'vect__ngram_range': ((1, 1), (2, 2), (3, 3)),
                  'clf__alpha': (0.001, 0.01, 0.1, 1, 2, 2.5)}
    grid = GridSearchCV(pipeline, param_grid=parameters, n_jobs=cpu_count())
    grid.fit(train_result['text'], train_result['t_id'])

    best_parameters = grid.best_estimator_.get_params()
    clf = grid.best_estimator_

    print(best_parameters, )
    print(f'Best score = {grid.best_score_}')
    print(f'Best estimator = {clf}')

    predict = clf.predict(test_result['text'])

    accuracy = accuracy_score(test_result['t_id'], predict)
    print(f'Accuracy = {accuracy}')

    matrix = confusion_matrix(test_result['t_id'], predict)
    sns.heatmap(matrix.T, fmt='d', cbar=False, square=True, annot=True)
    plt.xlabel('true')
    plt.ylabel('predicted')
    plt.show()

    docs_new = ('I ca remember many times game showed guy without puck',
                'game show lineman going running back turned corner '
                'touchdown Is Greg ESPN trying various things get away '
                'concept televising hockey',
                'OpenGL on the GPU  is fast')
    for doc, pred in zip(docs_new, predict):
        predicted = topics[topics['t_id'] == pred].index.values[0]
        print(f'{doc} -> {predicted}')

    sys.exit()
