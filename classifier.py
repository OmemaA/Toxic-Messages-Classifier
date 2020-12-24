from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from autocorrect import Speller
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pandas import DataFrame
import itertools
import logreg_one

offence = 0

# A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}

def clean_data(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

# preprocess creates the term frequency matrix for the review data set
def preprocess(data):
    data = data.map(lambda c: clean_data(c))
    # data = data.fillna(" ")
    count_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    data = count_vectorizer.fit_transform(data)
    return data, count_vectorizer

def load_file(fileName):
    dataset = pd.read_table(fileName, header=0, sep=",", encoding="unicode_escape")
    return dataset

labels = ['obscene','insult','toxic','severe_toxic','identity_hate','threat']

dataset = load_file("train.csv")
data = dataset['comment_text'].fillna(" ")
target = dataset[labels]
word_vectors, vectorizer = preprocess(data)

training_input, test_input, training_labels, test_labels = train_test_split(word_vectors,target,test_size=0.3,random_state=43)


def buildModel():
    print("Learning model.....")
    models = [logreg_one.LogisticRegression() for i in range(len(labels))]
    for i,label in enumerate(labels):
        models[i].fit(training_input, training_labels[label])
    return models


def check(models, sentence):
    print("Classifying test data......")
    sentence = clean_data(sentence)
    sentence = pd.Series(sentence)
    sentence = vectorizer.transform(sentence)
    predicted = []
    for i,model in enumerate(models):
        y_pred_X = model.predict(sentence)
        predicted.append(int(y_pred_X[0]))
        print(labels[i],":", y_pred_X)
    return predicted
