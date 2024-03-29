
# NLP-Spam-Ham-Classifier
A Machine learning classifier to predict whether the SMS is Spam or Ham by using Natural Language Processing(NLP)

### (Natural Language Toolkit)NLTK: 
NLTK is a popular open-source package in Python. Rather than building all tools from scratch, NLTK provides all common NLP Tasks.
## Installing NLTK
    !pip install nltk 
Type above code in the Jupyter Notebook or if it doesn’t work in cmd type conda install -c conda-forge nltk. This should work in most cases. 
Install NLTK: http://pypi.python.org/pypi/nltk
### Importing NLTK Library

After typing the above, we get an NLTK Downloader Application which is helpful in NLP Tasks
Stopwords Corpus is already installed in my system which helps in removing redundant repeated words. Similarly, we can install other useful packages.

## Reading and Exploring Dataset
### Reading in text data & why do we need to clean the text?
While reading data, we get data in the structured or unstructured format. A structured format has a well-defined pattern whereas unstructured data has no proper structure. In between the 2 structures, we have a semi-structured format which is a comparably better structured than unstructured format.

## Pre-processing Data
Cleaning up the text data is necessary to highlight attributes that we’re going to want our machine learning system to pick up on. Cleaning (or pre-processing) the data typically consists of a number of steps:
### 1. Remove punctuation
Punctuation can provide grammatical context to a sentence which supports our understanding. But for our vectorizer which counts the number of words and not the context, it does not add value, so we remove all special characters. eg: How are you?->How are you

In body_text_clean, we can see that all punctuations like I’ve-> I’ve are omitted.
### 2.Tokenization
Tokenizing separates text into units such as sentences or words. It gives structure to previously unstructured text. eg: Plata o Plomo-> ‘Plata’,’o’,’Plomo’.

In body_text_tokenized, we can see that all words are generated as tokens.
### 3. Remove stopwords
Stopwords are common words that will likely appear in any text. They don’t tell us much about our data so we remove them. eg: silver or lead is fine for me-> silver, lead, fine.

In body_text_nostop, all unnecessary words like been, for, the are removed.
## Preprocessing Data: Stemming
Stemming helps reduce a word to its stem form. It often makes sense to treat related words in the same way. It removes suffices, like “ing”, “ly”, “s”, etc. by a simple rule-based approach. It reduces the corpus of words but often the actual words get neglected. eg: Entitling,Entitled->Entitl
Note: Some search engines treat words with the same stem as synonyms.

In body_text_stemmed, words like entry,wkly is stemmed to entri,wkli even though don’t mean anything.
## Preprocessing Data: Lemmatizing
Lemmatizing derives the canonical form (‘lemma’) of a word. i.e the root form. It is better than stemming as it uses a dictionary-based approach i.e a morphological analysis to the root word.eg: Entitling, Entitled->Entitle
In Short, Stemming is typically faster as it simply chops off the end of the word, without understanding the context of the word. Lemmatizing is slower and more accurate as it takes an informed analysis with the context of the word in mind.

In body_text_stemmed, we can words like chances are lemmatized to chance whereas it is stemmed to chanc.
## Vectorizing Data
Vectorizing is the process of encoding text as integers i.e. numeric form to create feature vectors so that machine learning algorithms can understand our data.

### Vectorizing Data: Bag-Of-Words
Bag of Words (BoW) or CountVectorizer describes the presence of words within the text data. It gives a result of 1 if present in the sentence and 0 if not present. It, therefore, creates a bag of words with a document-matrix count in each text document.

BOW is applied on the body_text, so the count of each word is stored in the document matrix. (Check the repo).
### Vectorizing Data: N-Grams
N-grams are simply all combinations of adjacent words or letters of length n that we can find in our source text. Ngrams with n=1 are called unigrams. Similarly, bigrams (n=2), trigrams (n=3) and so on can also be used.

Unigrams usually don’t contain much information as compared to bigrams and trigrams. The basic principle behind n-grams is that they capture the letter or word is likely to follow the given word. The longer the n-gram (higher n), the more context you have to work with.

N-Gram is applied on the body_text, so the count of each group words in a sentence word is stored in the document matrix.(Check the repo).
### Vectorizing Data: TF-IDF
It computes “relative frequency” that a word appears in a document compared to its frequency across all documents. It is more useful than “term frequency” for identifying “important” words in each document (high frequency in that document, low frequency in other documents).
Note: Used for search engine scoring, text summarization, document clustering.
Check my previous post — In the TF-IDF Section, I have elaborated on the working of TF-IDF.

TF-IDF is applied on the body_text, so the relative count of each word in the sentences is stored in the document matrix. (Check the repo).
Note: Vectorizers outputs sparse matrices. Sparse Matrix is a matrix in which most entries are 0. In the interest of efficient storage, a sparse matrix will be stored by only storing the locations of the non-zero elements.
## Feature Engineering: Feature Creation
Feature engineering is the process of using domain knowledge of the data to create features that make machine learning algorithms work. It is like an art as it requires domain knowledge and it can tough to create features, but it can be fruitful for ML algorithm to predict results as they can be related to the prediction.

body_len shows the length of words excluding whitespaces in a message body.
punct% shows the percentage of punctuation marks in a message body.
## Check If Features are good or not

We can clearly see that Spams have a high number of words as compared to Hams. So it’s a good feature to distinguish.

Spam has a percentage of punctuations but not that far away from Ham. Surprising as at times spam emails can contain a lot of punctuation marks. But still, it can be identified as a good feature.

## NLP-Spam-Ham Classifier
All the above-discussed sections are combined to build a Spam-Ham Classifier.

Precision: 1.0 / Recall: 0.862 / F1-Score: 0.926 / Accuracy: 98.027%

