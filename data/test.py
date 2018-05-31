import os, csv, re, nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import snowballstemmer

# dada input
f = open('spam.csv', 'r')
file = csv.reader(f, delimiter = ',')

df = pd.DataFrame(np.array(list(file)))
df.columns = df.iloc[0]
df = df[1:]

le = preprocessing.LabelEncoder()
le.fit(df['v1'])
df['v1'] = le.transform(df['v1'])

# Clean texts
stop = set(stopwords.words('english'))
lmtzr = WordNetLemmatizer()
stemmer = snowballstemmer.stemmer('english')
corpus = np.array(df.v2)
c = []
for i in range(len(df.v2)):
    review = re.sub('[^a-zA-Z]', ' ', corpus[i])
    review = [i for i in review.lower().split() if i not in stop]
    l = [lmtzr.lemmatize(x) for x in review]
    s = stemmer.stemWords(l)
    review = ' '.join(s)
    c.append(review)

# Creating the Bag of Words model
cv = CountVectorizer()
X = cv.fit_transform(c).toarray()
y = df.v1

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print (cm)



















