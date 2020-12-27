import sqlite3
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import unicodedata
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle




class SqliteDBReader(object):
    """
    Provide streaming access to sqlite database records
    """
    def __init__ (self,path):
        self._cur=sqlite3.connect(path).cursor()
        
    def score_artist_album_reviews(self):
        """
        Reads the database and returns a DF with  score,artist name,artist album,review  as columns
        """
        
        sql = " SELECT R.score , A.artist , L.label , C.content FROM REVIEWS as R \
                JOIN ARTISTS as A on R.reviewid=A.reviewid \
                JOIN LABELS as L on R.reviewid=L.reviewid \
                JOIN CONTENT as C on R.reviewid=C.reviewid \
              " 
        data=[]
        for score,band,album,text in self._cur.execute(sql):
             data.append((score,band,album,text))
                
        return data

class preprocessor(object):
    """
    The preprocessor wrap the SqliteDBReader and does tokenization and part of speech tagging
    """
    def __init__(self,corpus):
        """
        the corpus is the extracted from the sqliteDB file using SqliteDBReader
        """
        self.corpus=corpus 
        
    def tokenize(self, text):
        """
        Segment a text into sentences, tokenize and tag the words in the corpus. Returns a list of sentences , which are lists
        tagged words
        """
        return [ 
            nltk.pos_tag(nltk.word_tokenize(sent))   
            for sent in nltk.sent_tokenize(text)
        ]
        
    def get_reviews(self):
        reviews=[]
        for (score,band,album,text) in self.corpus:
            reviews.append(self.tokenize(text))
            
        return reviews

    def get_scores(self):
        """
        bin the scores into 4 groups
        bad  : 0/0 < y < 3.0
        okay : 3.0 < y < 5.0
        good : 5.0 < y < 7.0
        great: 7.0 < y < 10.0
        """
        scores=[]
        for (score,band,album,text) in self.corpus:
            scores.append(score)
            
        return list(np.digitize(scores, [0.0,3.0,5.0,7.0,10.0]))

class TextNormalizer(BaseEstimator,TransformerMixin):
    def __init__ (self, language="english"):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords=set(nltk.corpus.stopwords.words(language))
    
        
    def is_stopword(self, token):
        return token.lower() in self.stopwords
        
    def is_punct(self, token):
        return all(
            unicodedata.category(char).startswith('P') for char in token
        )
    
    def lemmatize(self,token,pos_tag):
        
        tag = {
            "N" : wn.NOUN,
            "V" : wn.VERB,
            "R" : wn.ADV,
            "J" : wn.ADJ
        }.get(pos_tag[0],wn.NOUN)
        
        return self.lemmatizer.lemmatize(token,tag)
    
    def normalize(self,documents):
        return [
            self.lemmatize(token,tag) 
            for sent in documents 
            for token,tag in sent 
            if not self.is_stopword(token) 
            and not self.is_punct(token)
        ]
        
    def fit(self,documents,y=None):
        return self
    
    def transform(self,documents):
        return [" ".join(self.normalize(doc)) for doc in documents]


def main():
    path="./database.sqlite"
    corpus=SqliteDBReader(path).score_artist_album_reviews()
    y=preprocessor(corpus).get_scores()
    X=preprocessor(corpus).get_reviews()

    pipeline= Pipeline([
        ("Normalize",TextNormalizer()),
        ("Vectorizer",TfidfVectorizer()),
        ("Model",MultinomialNB())
        ])

    score=cross_val_score(pipeline,X,y,cv=12)
    print(score)
    with open("MultinomialNB_model.pkl","wb") as file:
        pickle.dump(pipeline,file)
    


if __name__ == "__main__":
    main()
    
    