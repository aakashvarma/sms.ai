import os, csv, re, nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import snowballstemmer


class spam():

    def __init__(self):
        pass

    def data_input(self, loc, filename):
        try:
            os.chdir(loc)
            f = open(filename, 'r')
            file = csv.reader(f, delimiter = ',')

            df = pd.DataFrame(np.array(list(file)))
            df.columns = df.iloc[0]
            df = df[1:]

            le = preprocessing.LabelEncoder()
            le.fit(df['v1'])
            df['v1'] = le.transform(df['v1'])
            print (df.shape)

            self.df = df

        except IOError:
            print ('PROBLEM READING: ' + filename)

    def data_cleaning(self):
        stop = set(stopwords.words('english'))
        lmtzr = WordNetLemmatizer()
        stemmer = snowballstemmer.stemmer('english')
        c = np.array(self.df.v2)
        self.corpus = []
        for i in range(len(self.df.v2)):
            review = re.sub('[^a-zA-Z]', ' ', c[i])
            review = [i for i in review.lower().split() if i not in stop]
            l = [lmtzr.lemmatize(x) for x in review]
            s = stemmer.stemWords(l)
            review = ' '.join(s)
            self.corpus.append(review)
        print (self.corpus)

    def data_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df['v2'], self.df['v1'], test_size=0.33, random_state=42)

    def vectorizer(self):
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(np.array(self.X_train))
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        print (X_train_counts.shape)
        print (X_train_tfidf.shape)



if __name__ == '__main__':

    loc = os.getcwd() + '\data'
    filename = 'spam.csv'
    s = spam()
    s.data_input(loc, filename)
    s.data_split()
    s.vectorizer()
    s.data_cleaning()



























