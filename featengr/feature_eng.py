import numpy as np
import nltk
import re
import pandas as pd
import matplotlib.pylab as plt
import pickle

from imdbpie import Imdb
imdb = Imdb()

from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, auc, confusion_matrix, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from scipy import sparse
from scipy.sparse import csr_matrix

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk import ne_chunk, pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags


GENRE_LIST = ['Horror', 'Action', 'Thriller', 'Crime', 'Sport', 'Mystery']

# from pycorenlp import StanfordCoreNLP
# nlp = StanfordCoreNLP('http://localhost:9000')


from collections import Counter


def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]


def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]


def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]


def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]


def get_imdb_data():
    imdb_data = pickle.load(open("imdb.pickle", "rb"))
    return imdb_data


class FeatEngr:
    def __init__(self):
        from sklearn.feature_extraction.text import CountVectorizer

        self.vectorizer = CountVectorizer()

    def build_train_features(self, examples):
        """
        Method to take in training text features and do further feature engineering
        Most of the work in this homework will go here, or in similar functions
        :param examples: currently just a list of forum posts
        """
        return self.vectorizer.fit_transform(examples)

    def get_test_features(self, examples):
        """
        Method to take in test text features and transform the same way as train features
        :param examples: currently just a list of forum posts
        """
        return self.vectorizer.transform(examples)

    def show_top10(self):
        """
        prints the top 10 features for the positive class and the
        top 10 features for the negative class.
        """
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        top10 = np.argsort(self.logreg.coef_[0])[-10:]
        bottom10 = np.argsort(self.logreg.coef_[0])[:10]
        print("Pos: %s" % " ".join(feature_names[top10]))
        print("Neg: %s" % " ".join(feature_names[bottom10]))

    def train_model(self, random_state=1234):
        """
        Method to read in training data from file, and
        train Logistic Regression classifier.

        :param random_state: seed for random number generator
        """

        from sklearn.linear_model import LogisticRegression

        # load data
        dfTrain = pd.read_csv("../data/spoilers/train.csv")

        # get training features and labels
        self.X_train = self.build_train_features(list(dfTrain["sentence"]))
        self.y_train = np.array(dfTrain["spoiler"], dtype=int)

        # train logistic regression model.  !!You MAY NOT CHANGE THIS!!
        self.logreg = LogisticRegression(random_state=random_state)
        self.logreg.fit(self.X_train, self.y_train)

    def model_predict(self):
        """
        Method to read in test data from file, make predictions
        using trained model, and dump results to file
        """

        # read in test data
        dfTest = pd.read_csv("../data/spoilers/test.csv")

        # featurize test data
        self.X_test = self.get_test_features(list(dfTest["sentence"]))

        # make predictions on test data
        pred = self.logreg.predict(self.X_test)

        # dump predictions to file for submission to Kaggle
        pd.DataFrame({"spoiler": np.array(pred, dtype=bool)}).to_csv("prediction.csv", index=True, index_label="Id")


def submit_code():
    pass


def lemmatize(sentence):
    from nltk.stem import WordNetLemmatizer

    lemmer = WordNetLemmatizer()
    return ' '.join([lemmer.lemmatize(word) for word in word_tokenize(sentence)])


def stemmer(sentence):
    from nltk.stem.snowball import SnowballStemmer

    stemmer = SnowballStemmer('english')
    return ' '.join([stemmer.stem(word) for word in sentence.split(' ')])


def extract_emotion_features():
    feat = ['anger', 'anxiety', 'negative_affect', 'positive_affect', 'sadness', 'swear']
    features = {}
    for i in feat:
        with open("../data/LIWC/{}".format(i), 'r+') as f:
            data = f.readlines()
            features[i] = [line.strip()[:-1] if line.strip()[-1] == '*' else line.strip() for line in data]

    return features


def count_named_entities(examples):
    X = np.zeros((len(examples), 1))

    # Loop over examples and count words
    for ii, x in enumerate(examples):
        ne_tree = ne_chunk(pos_tag(word_tokenize(x)))
        iob_tagged = tree2conlltags(ne_tree)
        iob = [iob[2] for iob in iob_tagged]
        X[ii, 0] = iob.count('B-PERSON') + iob.count('B-ORGANIZATION')

    return X


def count_tense(examples):
    tense = [['VBN', 'VBD'], ['VBG', 'VBP', 'VBZ'], ['JJ']]
    X = np.zeros((len(examples), 4))

    # Loop over examples and count words
    for ii, x in enumerate(examples):
        pos_tagged = pos_tag(word_tokenize(x))
        pos_tagged = [tag[1] for tag in pos_tagged]
        past = sum(pos_tagged.count(i) for i in tense[0])
        present = sum(pos_tagged.count(i) for i in tense[1])
        adjectives = sum(pos_tagged.count(i) for i in tense[2])
        other = pos_tagged.count('VB')

        X[ii, :] = [past, present, other, adjectives]

    return X


def get_strong_word_metrics(examples):
    words = ['kill', 'die', 'end', '``', 'death', 'finale', 'turn', 'reveal', 'murder', 'episode']

    X = np.zeros((1, len(words)))

    # Loop over examples and count words
    for ii, x in enumerate(examples):
        X[0, :] += np.array([lemmatize(stemmer(x)).count(word) for word in words])

    return X


def get_metrics():
    dfTrain = pd.read_csv("../data/spoilers/train.csv")
    dfTrain_y = np.array(dfTrain["spoiler"], dtype=int)
    dfTest = pd.read_csv("../data/spoilers/test.csv")

    spoiler_train = dfTrain[dfTrain["spoiler"] == 1]
    non_spoiler_train = dfTrain[dfTrain["spoiler"] == 0]

    # print(count_named_entities(spoiler_train["sentence"]).mean())
    # print(count_named_entities(non_spoiler_train["sentence"]).mean())
    #
    # print(count_tense(spoiler_train["sentence"]).mean(axis=0))
    # print(count_tense(non_spoiler_train["sentence"]).mean(axis=0))

    words_spoiler = get_strong_word_metrics(spoiler_train["sentence"])
    words_non_spoiler = get_strong_word_metrics(non_spoiler_train["sentence"])
    x = np.arange(10)
    plt.plot(x, words_spoiler.T, x, words_non_spoiler.T)
    plt.title("Plot for word counts for spoiler and non-spoiler sentences")
    plt.show()


def get_imdb_stats(pages_data, test_pages_data):
    all_pages = np.concatenate((np.asarray(pages_data), np.asarray(test_pages_data)))
    counter = Counter(all_pages)
    imdb_data = {}
    for key in counter.keys():
        splitted = re.sub('(?!^)([A-Z][a-z]+)', r' \1', key).split()
        title = ' '.join(splitted)
        if key in imdb_data:
            continue
        try:
            data = imdb.search_for_title(title)
            id = data[0]['imdb_id']
            year = data[0]['year']
            genres = imdb.get_title_genres(id)['genres']
            imdb_data[key] = {'year': year, 'id': id, 'genres': genres}
        except Exception as e:
            imdb_data[key] = None

    import pickle
    pickle.dump(imdb_data, open("imdb.pickle", "wb"))


def store_data():
    dfTrain = pd.read_csv("../data/spoilers/train.csv")
    test = pd.read_csv("../data/spoilers/test.csv")
    get_imdb_stats(np.array(dfTrain['page']), np.array(test['page']))


def cross_val(features, imdb_data):
    dfTrain = pd.read_csv("../data/spoilers/train.csv")
    dfTrain_y = np.array(dfTrain["spoiler"], dtype=int)
    dfTest = pd.read_csv("../data/spoilers/test.csv")

    bag_of_words_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 9), stop_words='english',
                                              max_features=3000)
    word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=1200)
    vectorizer = FeatureUnion([
        ("bag-of-words", word_vectorizer),
        ("character-ngram", bag_of_words_vectorizer),
    ])

    trope_vector = FeatureUnion([
        ("character-ngrams", TfidfVectorizer(analyzer='char_wb', ngram_range=(5, 10), max_features=1200)),
        ("bag-of-words", TfidfVectorizer(max_features=500))
    ])

    vectorizer_unioned = FeatureUnion(
        transformer_list=[
            # Pipeline for pulling features from the post's subject line
            ('sentence', Pipeline([
                ('selector', ItemSelector(key='sentence')),
                ('vectorizer', vectorizer),
            ])),
            # # Pipeline for counting occurences of top feature words of sentence
            # ('wordCounts', Pipeline([
            #     ('selector', ItemSelector(key='sentence')),
            #     ('transformer', WordTransformer(lemmatize, stemmer, csr_matrix)),
            # ])),
            # # Pipeline for counting tense for sentence
            # ('NER', Pipeline([
            #     ('selector', ItemSelector(key='sentence')),
            #     ('transformer', NERTransformer()),
            # ])),
            # # Pipeline for counting tense for sentence
            # ('TenseVerbCounts', Pipeline([
            #     ('selector', ItemSelector(key='sentence')),
            #     ('transformer', TenseTransformer()),
            # ])),
            # Pipeline for standard bag-of-words model for trope
            ('trope', Pipeline([
                ('selector', ItemSelector(key='trope')),
                ('tfidf', trope_vector),
            ])),
            # Pipeline for genre
            ('genre', Pipeline([
                ('selector', ItemSelector(key='page')),
                ('transformer', GenreTransformer(imdb_data)),
            ])),
            # Pipeline for year of release
            ('year', Pipeline([
                ('selector', ItemSelector(key='page')),
                ('transformer', YearTransformer(imdb_data)),
            ])),
        ],
        # transformer_weights={
        #     'sentence': 0.8,
        #     'trope': 0.8,
        #     'genre': 0.8,
        #     'year': 1.0,
        # }
    )

    pipe = Pipeline([
        ('tfid', vectorizer_unioned),
        ('chi2', SelectKBest(chi2, 5000)),
        ('classifier', LogisticRegression(random_state=1234))
    ])

    # N_FEATURES_OPTIONS = ['all']
    # param_grid = [
    #     {
    #         'chi2__k': N_FEATURES_OPTIONS,
    #     },
    # ]

    scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
               'fp': make_scorer(fp), 'fn': make_scorer(fn), 'accuracy': make_scorer(accuracy_score)}

    kfold = KFold(n_splits=6, random_state=1234, shuffle=False)
    results = cross_validate(pipe, dfTrain, dfTrain_y, cv=kfold, scoring=scoring)
    print(results)

    # X_train, X_test, y_train, y_test = train_test_split(dfTrain, dfTrain_y, test_size=0.20, random_state = 42)
    # pipe.fit(X_train, y_train)
    # print(pipe.named_steps['classifier'].coef_[0])
    # misclassified = np.where(y_test != pipe.predict(X_test))
    # print(misclassified)
    pipe.fit(dfTrain, dfTrain_y)
    pred = pipe.predict(dfTest)
    # grid = GridSearchCV(pipe, cv=kfold, n_jobs=1, param_grid=param_grid)
    # grid.fit(list(dfTrain), list(dfTrain_y))
    # print(np.array(grid.cv_results_['mean_test_score']))

    pd.DataFrame({"spoiler": np.array(pred, dtype=bool)}).to_csv("prediction.csv", index=True, index_label="Id")


class WordTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lemmatizer, stemmer, csr_matrix):
        self.lemmatizer = lemmatizer
        self.stemmer = stemmer
        self.csr_matrix = csr_matrix

    def fit(self, examples, y=None):
        # return self and nothing else
        return self

    def transform(self, examples, y=None):
        import numpy as np

        words = ['kill', 'die', 'end', '``', 'death', 'finale', 'turn', 'reveal', 'murder', 'episode']

        X = np.zeros((len(examples), len(words)))

        # Loop over examples and count words
        for ii, x in enumerate(examples):
            X[ii, :] = np.array([self.lemmatizer(self.stemmer(x)).count(word) for word in words])

        return self.csr_matrix(X)


class NERTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, examples, y=None):
        # return self and nothing else
        return self

    def transform(self, examples, y=None):
        import numpy as np
        from scipy.sparse import csr_matrix
        X = count_named_entities(examples)
        # X = X / norm_constant
        return csr_matrix(X)


class LengthTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, examples, y=None):
        return self

    def transform(self, examples, y=None):
        X = np.zeros((len(examples), 1))
        for ii, x in enumerate(examples):
            X[ii, 0] = len(x)

        return csr_matrix(X)


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class TenseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, examples, y=None):
        # return self and nothing else
        return self

    def transform(self, examples, y=None):
        import numpy as np
        from scipy.sparse import csr_matrix

        X = count_tense(examples)
        # X = X / norm

        return csr_matrix(X)


class EmotionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, emotiondict):
        self.emotion_dict = emotiondict

    def fit(self, examples, y=None):
        # return self and nothing else
        return self

    def transform(self, examples, y=None):
        import numpy as np
        from scipy.sparse import csr_matrix

        emotions = extract_emotion_features()
        X = np.zeros((len(examples), len(emotions)))


        norm = 1
        # Loop over examples and count words
        for ii, x in enumerate(examples):

            X[ii, :] = [sum([x.count(word) for word in emotions[k]]) for k in emotions]
            # norm += np.sum(X[ii, :])

        # X = X / norm

        return csr_matrix(X)


class GenreTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, imdb_data):
        self.imdb_data = imdb_data

    def fit(self, examples, y=None):
        # return self and nothing else
        return self

    def transform(self, examples, y=None):
        import numpy as np
        from scipy.sparse import csr_matrix

        X = np.zeros((len(examples), len(GENRE_LIST)))

        # norm = 1
        # Loop over examples and count words
        for ii, x in enumerate(examples):
            for i, g in enumerate(GENRE_LIST):
                if self.imdb_data[x] and g in self.imdb_data[x]['genres']:
                    X[ii, i] = 1
                else:
                    X[ii, i] = 0

        return csr_matrix(X)


class YearTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, imdb_data):
        self.imdb_data = imdb_data
        self.current_year = 2018

    def fit(self, examples, y=None):
        # return self and nothing else
        return self

    def transform(self, examples, y=None):
        import numpy as np
        from scipy.sparse import csr_matrix

        X = np.zeros((len(examples), 1))

        # norm = 1
        # Loop over examples and count words
        for ii, x in enumerate(examples):
            if self.imdb_data[x] and self.imdb_data[x]['year'] and int(self.imdb_data[x]['year']) < self.current_year:
                X[ii, 0] = self.current_year - int(self.imdb_data[x]['year'])
            else:
                X[ii, 0] = 0

        return csr_matrix(X)


if __name__ == "__main__":
    emotion_features = extract_emotion_features()
    # store_data()
    imdb_data = pickle.load(open("imdb.pickle", "rb"))
    # get_metrics()
    cross_val(emotion_features, imdb_data)
    # feat = FeatEngr()
    #
    # # Train your Logistic Regression classifier
    # feat.train_model(random_state=1230)
    #
    # # Shows the top 10 features for each class
    # feat.show_top10()
    #
    # # Make prediction on test data and produce Kaggle submission file
    # # feat.model_predict()
