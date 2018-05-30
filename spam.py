import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


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


            # X_train, X_test, y_train, y_test = train_test_split(df['v2'], df['v1'], test_size=0.33, random_state=42)
            # print (y_train)


            self.df = df
            # print (self.df.head())

        except IOError:
            print ('PROBLEM READING: ' + filename)

    def data_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df['v2'], self.df['v1'], test_size=0.33, random_state=42)
        # print (np.array(self.y_train))

    def vectorizer(self):
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(np.array(self.X_train))
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        print(X_train_tfidf.shape)



if __name__ == '__main__':

    loc = os.getcwd() + '\data'
    filename = 'spam.csv'
    s = spam()
    s.data_input(loc, filename)
    s.data_split()
    s.vectorizer()



























