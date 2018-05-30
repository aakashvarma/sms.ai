import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

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

            self.df = df
            # print (self.df.head())

        except IOError:
            print ('PROBLEM READING: ' + filename)

if __name__ == '__main__':

    loc = os.getcwd() + '\data'
    filename = 'spam.csv'
    s = spam()
    s.data_input(loc, filename)




























