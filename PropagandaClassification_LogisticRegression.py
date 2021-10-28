import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve
import re

#load the data using pandas
data_df=pd.read_csv(r"D:\Work\Class\Seminar\Propaganda_Dataset_tweets.csv")

#adding column headers
data_df.columns=["Source","Indicator","Tweet"]

#Visualizing the data
prop=data_df.loc[data_df['Indicator']==1]['Indicator'].value_counts()
non_prop=data_df.loc[data_df['Indicator']==0]['Indicator'].value_counts()
df_plot=pd.DataFrame([prop, non_prop])
df_plot.index=['propaganda','non_propaganda']
df_plot.plot(kind='bar',figsize=(10,10))

#Text Preprocessing 

def remove_unwanted(text):
    text=str(text)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    return(text)
 
 def remove_stopwords(text):
    text=str(text)
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return(text)
  
def remove_punctuations(text):
    punctuation_remover=RegexpTokenizer(r'\w+')
    punctuation_remover.tokenize(text)
    return (text)

def lowercase(text):
    return(text.lower())

def lemma(text):
    text_tokens=word_tokenize(text)
    word_lem=WordNetLemmatizer()
    #print(text_tokens)
    lemmatized_text=[]
    i=0
    for word in text_tokens:
        lemmatized_text.append(word_lem.lemmatize(word))
        i+=1
    return(lemmatized_text)
 
def convert_to_single_string(text):
    single_text=""
    for i in range(len(text)):
        if(i==0):
            single_text=single_text+text[i]
        else:
            single_text=single_text+" "+text[i]
    return(single_text)
 
#Applying the preprocessing steps to the Tweets column in dataframe

data_df['Tweet']=data_df['Tweet'].apply(lowercase)
data_df['Tweet']=data_df['Tweet'].apply(remove_unwanted)
data_df['Tweet']=data_df['Tweet'].apply(remove_stopwords)
data_df['Tweet']=data_df['Tweet'].apply(remove_punctuations)
data_df['Tweet']=data_df['Tweet'].apply(lemma)
data_df['Tweet']=data_df['Tweet'].apply(convert_to_single_string)

#Creating a final dataset dataframe with the preprocessed text
X=data_df['Tweet']
y=data_df['Indicator'].values
final_dataset=pd.DataFrame({"Indicator":y ,"Tweet":X})

#splitting the data into test and train sets
X_train,X_test, y_train,y_test=train_test_split(final_dataset["Tweet"].values,final_dataset["Indicator"].values,test_size=0.2)

#convert the test and train sets into machine readable numbers through tfidf vectorization
tfidfvectorizer = TfidfVectorizer(use_idf=True,sublinear_tf=True)
tfidfvectors_train = tfidfvectorizer.fit_transform(X_train)
tfidfvectors_test  = tfidfvectorizer.transform(X_test)

#initialze the model and score it
log_reg=LogisticRegression(solver = 'liblinear', C=10, penalty = 'l2')
log_reg.fit(tfidfvectors_train,y_train)
scores = cross_val_score(log_reg,tfidfvectors_train,y_train,scoring="accuracy",cv=5)
