import pandas as pd
from newspaper import Article
from nltk.tag import StanfordNERTagger
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import time
import requests
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
import datefinder

def stem_tokens(tokens):
    stemmer = nltk.stem.porter.PorterStemmer()
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

def cosine_sim(text1, text2):
    vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')
    tfidf = vectorizer.fit_transform([text1, text2])
    similarity = ((tfidf * tfidf.T).A)[0,1]
    print(similarity)

if __name__ == '__main__':
    cosine_sim('text1', 'text2')