import sklearn.linear_model
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from main import NN


def newsgroup_data_generation():
    categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']

    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

    num_train = len(newsgroups_train.data)
    num_test = len(newsgroups_test.data)

    # max_features is an important parameter. You should adjust it.
    vectorizer = TfidfVectorizer(max_features=40)

    X = vectorizer.fit_transform(newsgroups_train.data + newsgroups_test.data)
    X_train = X[0:num_train, :]
    X_test = X[num_train:num_train + num_test, :]

    y_train = newsgroups_train.target
    y_test = newsgroups_test.target

    return X_train, y_train, X_test, y_test


def test():
    X_train, y_train, X_test, y_test = newsgroup_data_generation()
    cls = NN(reg_lambda=.002, width=[12, 6], activation=['relu', 'relu'])
    cls.fit(X_train.toarray(), y_train, verbose=False, batch_size=16, epoch=30000, eta=4e-3, optimizer='Adam')
    cls.plot_losses(save_fig=True, file_name='relu.svg')
    cls.score(X_test.toarray(), y_test)


if __name__ == '__main__':
    test()
