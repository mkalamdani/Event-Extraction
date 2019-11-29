from flask import Flask, render_template, request,jsonify
from newspaper import Article
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import time
import requests
import json
import datefinder
import re
from datetime import datetime
import parsedatetime

def _news_article_extractor(query, DictofDict):
    '''
    we form url for hitting NEWS API with source and query and sorted by popularity.
    After getting the JSON response from the API, we'll extracts the articles.
    we'll iterate each article and give article url to _article_download where we can downlaod the article content.
    Processed results are stored in a Dictionary "Dict" and each result is appended in an array.
    Array and total number of articles downloaded are put in a Dictionary "DictofDict".
    '''
    url = ('https://newsapi.org/v2/everything?sources=the-times-of-india&q='+query+'&sortBy=popularity&apiKey=f2cdb3dcb01c4c588895c1fef3ea0bc0')
    response = requests.get(url)
    articles = response.json()["articles"]
    Dict = {}
    Array = []
    i = 1
    for article in articles:
        if(_article_download(article["url"], Dict) == False):
            Dict = {}
            continue
        Dict["TITLE"] = str(article["title"])
        Array.append(Dict)
        Dict = {}
        i += 1

    DictofDict['Results'] = Array
    DictofDict['NoOfArticles'] = i - 1 

def _create_frequency_table(text_string) -> dict:
    """
    we create a dictionary for the word frequency table.
    For this, we should only use the words that are not part of the stopWords array.
    Removing stop words and making frequency table
    Stemmer - an algorithm to bring words to its root word.
    :rtype: dict
    """
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable


def _score_sentences(sentences, freqTable) -> dict:
    """
    score a sentence by its words
    Basic algorithm: adding the frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = dict()

    for sentence in sentences:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        word_count_in_sentence_except_stop_words = 0
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                word_count_in_sentence_except_stop_words += 1
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]
        try:
            sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] / word_count_in_sentence_except_stop_words
        except:
            pass
        '''
        Notice that a potential issue with our score algorithm is that long sentences will have an advantage over short sentences. 
        To solve this, we're dividing every sentence score by the number of words in the sentence.
        
        Note that here sentence[:10] is the first 10 character of any sentence, this is to save memory while saving keys of
        the dictionary.
        '''

    return sentenceValue


def _find_average_score(sentenceValue) -> int:
    """
    Find the average score from the sentence value dictionary
    :rtype: int
    """
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original text
    #average = (sumValues / len(sentenceValue))
    try:
        average = (sumValues / len(sentenceValue))
    except ZeroDivisionError:
        average = 0

    return average


def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary


def run_summarization(text, Dict):
    # 1 Create the word frequency table
    freq_table = _create_frequency_table(text)

    '''
    We already have a sentence tokenizer, so we just need 
    to run the sent_tokenize() method to create the array of sentences.
    '''

    # 2 Tokenize the sentences
    sentences = sent_tokenize(text)

    # 3 Important Algorithm: score the sentences
    sentence_scores = _score_sentences(sentences, freq_table)

    # 4 Find the threshold
    threshold = _find_average_score(sentence_scores)

    # 5 Important Algorithm: Generate the summary
    summary = _generate_summary(sentences, sentence_scores, 1.05 * threshold)
    Dict["SUMMARY"] = summary
    return summary

def _tagger_util(text, Dict):
    '''
    first we'll tokenize the text into words. 
    Then using Stanford Libraries each word is tagged with corresponding parts of speech.
    Then classified text is passed to _person_tag, _organization_tag, _location_tag to get persons involved, organizations involved and Event Location.
    Most of the News Papers present Event Location at the start of the article. It is checked first. If it is not present then classified text is passed for _location_tag.
    '''
    st = StanfordNERTagger('/Users/abhinay/Downloads/stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz',
                           '/Users/abhinay/Downloads/stanford-ner-2018-10-16/stanford-ner-3.9.2.jar',
                           encoding='utf-8')
    tokenized_text = word_tokenize(text)
    classified_text = st.tag(tokenized_text)
    _person_tag(classified_text, Dict)
    _organization_tag(classified_text, Dict)
    position = Dict["FullContent"][0:30].find(':')
    if(position == -1):
        _location_tag(classified_text, Dict)
    else:
        Dict["LOCATION"] = Dict["FullContent"][0:position]
    
def _person_tag(classified_text, Dict):
    '''
    Here we iterate the classified text and compare whether its tag is "PERSON".
    We'll consider words with tags "PERSON" which are adjacent as single person.
    We also handled duplicates here.
    '''
    d = defaultdict(list)
    i=0
    length = len(classified_text)
    while(i < length):
        j=i
        if classified_text[i][1]=="PERSON":
            s=""
            s=classified_text[i][0]
            while(j + 1 < length and classified_text[j+1][1]=="PERSON"):
                s+=" "
                s+=classified_text[j+1][0]
                j+=1
            if s.lower() not in d["PERSON"]:    
                d["PERSON"].append(s.lower())
        if(i==j):
            i=i+1
        else:
            i=j+1
    tempstr = ""
    for person in d["PERSON"]:
        if re.search(re.escape(person), tempstr, re.IGNORECASE):
            print("")
        else:
            tempstr += person.capitalize() + ", "
    Dict["PERSON"] = tempstr[:-2]

def _organization_tag(classified_text, Dict):
    '''
    Here we iterate the classified text and compare whether its tag is "ORGANIZATION".
    We'll consider words with tags "ORGANIZATION" which are adjacent as single Organization.
    We also handled duplicates here.
    '''
    d = defaultdict(list)
    i=0
    length = len(classified_text)
    while(i < length):
        j=i
        if classified_text[i][1]=="ORGANIZATION":
            s = ""
            s = classified_text[i][0]
            while(j + 1 < length and classified_text[j+1][1]=="ORGANIZATION"):
                s+=" "
                s+=classified_text[j+1][0]
                j+=1
            if s.lower() not in d["ORGANIZATION"]:    
                d["ORGANIZATION"].append(s.lower())
        if(i==j):
            i=i+1
        else:
            i=j+1
    tempstr = ""
    for organization in d["ORGANIZATION"]:
        if re.search(re.escape(organization), tempstr, re.IGNORECASE):
            print("")
        else:
            tempstr += organization.capitalize()+ ", "
    Dict["ORGANIZATION"] = tempstr[:-2]
    
def _location_tag(classified_text, Dict):
    '''
    Here we iterate the classified text and compare whether its tag is "LOCATION".
    We'll consider words with tags "LOCATION" which are adjacent as single location.
    We also handled duplicates here.
    In case of multiple locations we are considering location in the highly ranked sentence as the Event Location.
    '''
    d = defaultdict(list)
    i=0
    length = len(classified_text)
    while(i < length):
        j=i
        if classified_text[i][1]=="LOCATION":
            s=""
            s=classified_text[i][0]
            while(j + 1 < length and classified_text[j+1][1]=="LOCATION"):
                s+=" "
                s+=classified_text[j+1][0]
                j+=1
            if s.lower() not in d["LOCATION"]:    
                d["LOCATION"].append(s.lower())
        if(i==j):
            i=i+1
        else:
            i=j+1
    tempstr = ""
    tempstr1 = ""
    for location in d["LOCATION"]:
        if re.search(re.escape(location), Dict["SUMMARY"], re.IGNORECASE):
            if re.search(re.escape(location), tempstr, re.IGNORECASE):
                print("")
            else:
                tempstr += location.capitalize() + ", "
        else:
            tempstr1 += location.capitalize() + ", "
    if(len(tempstr) == 0):
        Dict["LOCATION"] = tempstr1[:-2]
    else:
        Dict["LOCATION"] = tempstr[:-2]

def _date_extractor(notes, Dict):
    '''
    if the text contains date directly it will be extracted.
    Else, Any indirect reference will be compared with the article published date and given.
    '''
    cal = parsedatetime.Calendar()
    time_struct, parse_status = cal.parse(notes)
    date = str(datetime(*time_struct[:6]))
    Dict["EVENTDATE"] = date
    return True

def _article_download(article_url, Dict):
    '''
    Article url will used to download the full content of the article using Newspaper library.
    Any url in the artcle content will be removed.
    '''
    a = Article(article_url.strip(), language='en')
    a.download()
    try:
        a.parse()
    except:
        return False
    text = a.text
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    Dict["FullContent"] = text
    if(len(run_summarization(text, Dict)) == 0):
        return False
    _tagger_util(a.text, Dict)
    _date_extractor(a.text, Dict)
    return True

def _query_passer(query):
    '''
    "DictofDict" is a dictionary which will store the Array of results and total number of articles.
    Dictionary is converted into JSON.
    '''
    DictofDict = {} 
    _news_article_extractor(query, DictofDict)
    app_json = json.dumps(DictofDict) 
    return str(app_json)

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('home.html')

@app.route("/next", methods = ['GET', 'POST'])
def next():
    app_json = _query_passer(request.args.get("searchText"))
    return render_template('next.html', posts = app_json)


if __name__ == '__main__':
    app.run(debug=True, host = "0.0.0.0", port = 5000)
